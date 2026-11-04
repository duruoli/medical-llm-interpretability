"""Where downstream does a diff/tail conflict tip toward the winning side? (observation)

Same 5 hand-picked cases x 3 forms. We build the conflict states from followup A / D and,
WITHOUT touching the downstream positions, read every position's residual at every layer.

States (each one forward pass; the patch is at the diff or result_tail positions only):
  clean_P, clean_N                         references
  X@l  (diff says N, tail says P)          X_hostP: P run, diff residual after layer l <- N
                                           X_hostN: N run, tail residual after layer l <- P
  Y@l  (diff says P, tail says N)          Y_hostN: N run, diff residual after layer l <- P
                                           Y_hostP: P run, tail residual after layer l <- N
  MLP@L (inside the tail: token mixers say one side, MLPs the other)
                                           mlpN_hostP: P run, every MLP write into the tail over
                                                       layers 0..L <- N
                                           mlpP_hostN: N run, same with P's MLP writes
Two hosts per X/Y state = two estimates of the same state (they agreed in A).

Readings, per state x layer (0..31) x position (diff onward):
  s   = <h - h_N, h_P - h_N> / |h_P - h_N|^2      "P-ness": 0 = like clean N, 1 = like clean P
  off = |h - h_N - s (h_P - h_N)| / |h_P - h_N|   distance from the N-P line (large: a state
                                                  neither run has; then s means little)
  rel_gap = |h_P - h_N| / |h_P|                   how different P and N are there at all
Per segment, positions are pooled (sum of dot products / sum of squared gaps).
At the final position also the logit lens: the 4 answer letters (m) and the same s / off in
vocabulary space (only directions readable as words).

Output: results/conflict/<model_tag>/
  conflict_seg.jsonl    state x read_layer x segment: s, off, rel_gap
  conflict_final.jsonl  state x read_layer: m_lens, s_lens, off_lens; and m_out (the answer)
  perpos/<case>__<form>.pt   per-position dot / dd / xx / pp, to pool positions differently
  run_config.json
Check: m_out of X_hostP@l equals followup A's noise-diff at l (same intervention).

Run (repo root):
  sbatch conflict_posneg.slurm                                           # Quest
  python conflict_posneg.py --model Qwen/Qwen3.5-0.8B --cases lipase__28842673 --forms negation \
      --conflict-layers 5 6 --mlp-layers 6 --device cpu                  # local smoke test
"""

import argparse
import json
import time
from pathlib import Path

import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from followup_posneg import cache_writes, lens
from mechanism_posneg import DEFAULT_CASES
from patch_posneg import FORMS, LETTERS, cache_resid, encode, metric, segments

READ_SEGS = ["diff", "result_tail", "question", "template", "final"]


def build_states(conflict_layers, mlp_layers):
    """(name, host, kind, layer, where, source) per row; source = run the patch comes from."""
    rows = [("clean_P", "P", None, None, None, None), ("clean_N", "N", None, None, None, None)]
    for l in conflict_layers:
        rows += [("X_hostP", "P", "resid", l, "diff", "N"), ("X_hostN", "N", "resid", l, "result_tail", "P"),
                 ("Y_hostN", "N", "resid", l, "diff", "P"), ("Y_hostP", "P", "resid", l, "result_tail", "N")]
    for L in mlp_layers:
        rows += [("mlpN_hostP", "P", "mlp", L, "result_tail", "N"),
                 ("mlpP_hostN", "N", "mlp", L, "result_tail", "P")]
    return rows


@torch.no_grad()
def run_rows(lm, rows, ids, caches, writes, seg, read_pos, n_layers):
    """One batched forward of `rows`; returns residual at read_pos per layer [rows, pos, hidden]
    and the last-position logits [rows, vocab]."""
    batch = torch.tensor([ids[r[1]] for r in rows])
    rp = torch.tensor(read_pos)
    saved = []
    with lm.trace(batch, logits_to_keep=1):
        for l in range(n_layers):        # modules touched in execution order: mlp, then layer out
            for i, (_, _, kind, L, where, src) in enumerate(rows):
                if kind == "mlp" and l <= L:
                    p = torch.tensor(seg[where])
                    lm.model.layers[l].mlp.output[i, p, :] = writes[src]["mlp"][l][p, :]
            for i, (_, _, kind, L, where, src) in enumerate(rows):
                if kind == "resid" and l == L:
                    p = torch.tensor(seg[where])
                    lm.model.layers[l].output[i, p, :] = caches[src][l][p, :]
            saved.append(lm.model.layers[l].output[:, rp, :].save())
        logits = lm.lm_head.output[:, -1, :].save()
    return [h.float() for h in saved], logits.float()


def proj_stats(h, hp, hn):
    """Per-position dot, dd, xx for h relative to the N->P line (last dim = features)."""
    d, x = hp - hn, h - hn
    return (x * d).sum(-1), (d * d).sum(-1), (x * x).sum(-1)


def pooled(dot, dd, xx):
    s = dot.sum() / dd.sum()
    off2 = (xx - dot ** 2 / dd.clamp_min(1e-12)).sum().clamp_min(0)
    return s.item(), (off2 / dd.sum()).sqrt().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--pool", default="data/posneg_pool.jsonl")
    ap.add_argument("--out-root", default="results/conflict")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--forms", nargs="+", default=FORMS)
    ap.add_argument("--conflict-layers", type=int, nargs="+", default=list(range(4, 13)))
    ap.add_argument("--mlp-layers", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_root) / args.model.split("/")[-1]
    (out / "perpos").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    lm = LanguageModel(hf, tokenizer=tok)
    n_layers, layer_types = hf.config.num_hidden_layers, hf.config.layer_types
    letter_ids = tok.convert_tokens_to_ids(LETTERS)
    conflict_layers = [l for l in args.conflict_layers if l < n_layers]
    mlp_layers = [l for l in args.mlp_layers if l < n_layers]
    states = build_states(conflict_layers, mlp_layers)
    (out / "run_config.json").write_text(json.dumps(
        dict(vars(args), device=device, dtype=str(dtype), n_layers=n_layers, layer_types=layer_types,
             states=[dict(zip(["state", "host", "kind", "at_layer", "where", "source"], r)) for r in states]),
        indent=2))

    pool = {c["case_id"]: c for c in map(json.loads, open(args.pool))}
    f_seg = (out / "conflict_seg.jsonl").open("w")
    f_fin = (out / "conflict_final.jsonl").open("w")
    t0 = time.time()
    for cid in args.cases:
        c = pool[cid]
        ti = LETTERS.index(c["target_letter"])
        for form in args.forms:
            fm = c["forms"][form]
            _, p_ids, offs = encode(tok, fm["P"])
            _, n_ids, _ = encode(tok, fm["N"])
            seg = segments(tok, fm["P"], fm["P_result"], p_ids, n_ids, offs)
            if "result_tail" not in seg:
                print(f"  skip {cid} {form}: no result_tail", flush=True)
                continue
            read_pos = list(range(seg["diff"][0], len(p_ids)))
            idx = {s: [read_pos.index(p) for p in seg[s]] for s in READ_SEGS if s in seg}
            ids = {"P": p_ids, "N": n_ids}
            cp, lp = cache_resid(lm, p_ids, n_layers)
            cn, ln = cache_resid(lm, n_ids, n_layers)
            caches = {"P": cp, "N": cn}
            max_w = max(mlp_layers) if mlp_layers else 0
            writes = {"P": cache_writes(lm, p_ids, layer_types, max_w),
                      "N": cache_writes(lm, n_ids, layer_types, max_w)}
            m_p, m_n = metric(lp, letter_ids, ti).item(), metric(ln, letter_ids, ti).item()
            base = {"case_id": cid, "test_id": c["test_id"], "form": form, "m_P": m_p, "m_N": m_n}

            hp = [cp[l][read_pos].float() for l in range(n_layers)]
            hn = [cn[l][read_pos].float() for l in range(n_layers)]
            fpos = idx["final"][0]
            lens_p = [lens(hf, cp[l][read_pos[fpos]]) for l in range(n_layers)]
            lens_n = [lens(hf, cn[l][read_pos[fpos]]) for l in range(n_layers)]

            stats = torch.zeros(len(states), n_layers, len(read_pos), 4)   # dot, dd, xx, pp
            for b in range(0, len(states), args.batch):
                rows = states[b:b + args.batch]
                saved, logits = run_rows(lm, rows, ids, caches, writes, seg, read_pos, n_layers)
                m_out = metric(logits, letter_ids, ti).tolist()
                for i, r in enumerate(rows):
                    meta = dict(base, state=r[0], host=r[1], kind=r[2], at_layer=r[3], where=r[4])
                    for l in range(n_layers):
                        dot, dd, xx = proj_stats(saved[l][i], hp[l], hn[l])
                        pp = (hp[l] * hp[l]).sum(-1)
                        stats[b + i, l] = torch.stack([dot, dd, xx, pp], -1).cpu()
                        for sname, ix in idx.items():
                            ix_t = torch.tensor(ix)
                            s, off = pooled(dot[ix_t], dd[ix_t], xx[ix_t])
                            rel = (dd[ix_t].sum() / pp[ix_t].sum()).sqrt().item()
                            f_seg.write(json.dumps(dict(meta, read_layer=l, segment=sname,
                                                        s=s, off=off, rel_gap=rel)) + "\n")
                        z = lens(hf, saved[l][i, fpos].to(hf.lm_head.weight.dtype))
                        ld, ldd, lxx = proj_stats(z, lens_p[l], lens_n[l])
                        s_l, off_l = pooled(ld, ldd, lxx)
                        f_fin.write(json.dumps(dict(
                            meta, read_layer=l, m_out=m_out[i],
                            m_lens=metric(z, letter_ids, ti).item(),
                            m_lens_P=metric(lens_p[l], letter_ids, ti).item(),
                            m_lens_N=metric(lens_n[l], letter_ids, ti).item(),
                            s_lens=s_l, off_lens=off_l)) + "\n")
                del saved
            torch.save({"states": states, "read_pos": read_pos, "segments": idx,
                        "fields": ["dot", "dd", "xx", "pp"], "stats": stats},
                       out / "perpos" / f"{cid}__{form}.pt")
            f_seg.flush()
            f_fin.flush()
            print(f"  {cid} {form}  gap {m_p - m_n:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    f_seg.close()
    f_fin.close()
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
