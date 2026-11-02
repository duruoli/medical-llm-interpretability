"""Follow-up on the diff -> result_tail handoff (layers ~6-9), same hand-picked cases.

A. joint.jsonl - residual patching at diff alone, result_tail alone, and BOTH AT ONCE,
   layers 0..--max-layer, both directions. Joint patching is one intervention in one
   forward pass (not a sum of single patches): if joint ~ 1 in both directions, the P/N
   information at that layer lives in these two places only.

B. lens.jsonl - logit lens: each position's residual at each layer, pushed through the
   model's own final norm + unembedding ("if the model had to output a word from these
   notes now, which words?"). Per position (each diff and result_tail token, and the final
   token): the score of the 4 answer letters + metric m, for P and for N; and at selected
   layers the words whose score N raises most over P, and P over N.

C. heads.jsonl - do specific attention heads carry the N information?
   A head's output = its slice of the o_proj input (after the output gate).
   Head groups (layer.head): `neg` = heads that attend MORE to "no"/"normal" than to the
   positive word (from mechanism_posneg attention), `pos` = top heads preferring the
   positive word (control, same count, all after the veto layer).
     swap_denoise / swap_noise : only these heads' outputs (all positions from diff on)
                                 replaced by the other run's -> recovery
     zero_N / zero_P           : heads zeroed in the clean N / P run; recovery-scaled shift
                                 (zero_N: (m - m_N)/gap toward P; zero_P: (m_P - m)/gap toward N)
     veto / veto_clamped       : P run with N's residual at diff from --veto-layer (the state
                                 where "no" at diff wins against the positive tail); clamped =
                                 additionally the group's heads held at their clean-P outputs.
                                 If clamping removes most of the recovery, these heads carry
                                 the "no" information to the answer.

Run (repo root):
  sbatch followup_posneg.slurm                                           # Quest
  python followup_posneg.py --model Qwen/Qwen3.5-0.8B --cases lipase__28842673 --forms negation \
      --heads neg:3.1,7.2 pos:3.4,7.5 --device cpu                       # local smoke test
Output: results/followup/<model_tag>/{joint,lens,heads}.jsonl + run_config.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mechanism_posneg import DEFAULT_CASES
from patch_posneg import FORMS, LETTERS, cache_resid, encode, metric, segments

LENS_LAYERS = [4, 6, 8, 10, 12, 16, 20, 24, 28, 31]
TOPK = 15


def parse_heads(specs):
    """["neg:19.13,19.14"] -> {"neg": [(19, 13), (19, 14)]}"""
    out = {}
    for s in specs:
        name, hs = s.split(":")
        out[name] = sorted(tuple(int(x) for x in h.split(".")) for h in hs.split(","))
    return out


@torch.no_grad()
def patch_resid_layers(lm, run_ids, src, positions, layers, letter_ids, ti):
    """Batch row i = run_ids with the residual at `positions` after layers[i] taken from src."""
    batch = torch.tensor([run_ids] * len(layers))
    pos = torch.tensor(positions)
    with lm.trace(batch, logits_to_keep=1):
        for i, l in enumerate(layers):
            lm.model.layers[l].output[i, pos, :] = src[l][pos, :]
        logits = lm.lm_head.output[:, -1, :].save()
    return metric(logits, letter_ids, ti).tolist()


@torch.no_grad()
def cache_heads(lm, ids, layers):
    """o_proj input (= all heads' gated outputs) at the given full-attention layers."""
    out = {}
    with lm.trace(torch.tensor([ids]), logits_to_keep=1):
        for l in layers:
            out[l] = lm.model.layers[l].self_attn.o_proj.input[0].save()
    return out


@torch.no_grad()
def run_with(lm, ids, letter_ids, ti, hd, heads=(), head_src=None, positions=None,
             resid_layer=None, resid_src=None, resid_pos=None):
    """One forward of `ids` with optional (1) residual patch at resid_pos after resid_layer,
    (2) the listed heads set to head_src's outputs (or zero if head_src is None) at positions."""
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    pos = torch.tensor(positions) if positions is not None else None
    with lm.trace(torch.tensor([ids]), logits_to_keep=1):
        if resid_layer is not None:
            rp = torch.tensor(resid_pos)
            lm.model.layers[resid_layer].output[0, rp, :] = resid_src[resid_layer][rp, :]
        for l in sorted(by_layer):
            x = lm.model.layers[l].self_attn.o_proj.input
            for h in by_layer[l]:
                sl = slice(h * hd, (h + 1) * hd)
                x[0, pos, sl] = 0 if head_src is None else head_src[l][pos, sl]
        logits = lm.lm_head.output[0, -1, :].save()
    return metric(logits, letter_ids, ti).item()


@torch.no_grad()
def lens(hf, h):
    """Residual [..., hidden] -> vocabulary scores via the model's final norm + unembedding."""
    return hf.lm_head(hf.model.norm(h)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--pool", default="data/posneg_pool.jsonl")
    ap.add_argument("--out-root", default="results/followup")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--forms", nargs="+", default=FORMS)
    ap.add_argument("--max-layer", type=int, default=16)
    ap.add_argument("--heads", nargs="+", default=["neg:19.13,19.14,27.10", "pos:23.12,23.14,27.1"])
    ap.add_argument("--veto-layer", type=int, default=6)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_root) / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    lm = LanguageModel(hf, tokenizer=tok)
    n_layers, hd = hf.config.num_hidden_layers, hf.config.head_dim
    layer_types = hf.config.layer_types
    groups = parse_heads(args.heads)
    for g, hs in groups.items():
        for l, h in hs:
            assert layer_types[l] == "full_attention", f"{g}: layer {l} is not a full-attention layer"
            assert h < hf.config.num_attention_heads, f"{g}: head {h} out of range"
    head_layers = sorted({l for hs in groups.values() for l, _ in hs})
    joint_layers = list(range(min(args.max_layer, n_layers - 1) + 1))
    lens_layers = [l for l in LENS_LAYERS if l < n_layers] + ([n_layers - 1] if n_layers - 1 not in LENS_LAYERS else [])
    letter_ids = tok.convert_tokens_to_ids(LETTERS)
    (out / "run_config.json").write_text(json.dumps(
        dict(vars(args), device=device, dtype=str(dtype), n_layers=n_layers, layer_types=layer_types,
             head_groups={g: [f"{l}.{h}" for l, h in hs] for g, hs in groups.items()}), indent=2))

    pool = {c["case_id"]: c for c in map(json.loads, open(args.pool))}
    f_joint = (out / "joint.jsonl").open("w")
    f_lens = (out / "lens.jsonl").open("w")
    f_heads = (out / "heads.jsonl").open("w")
    t0 = time.time()
    for cid in args.cases:
        c = pool[cid]
        ti = LETTERS.index(c["target_letter"])
        for form in args.forms:
            fm = c["forms"][form]
            _, p_ids, offs = encode(tok, fm["P"])
            _, n_ids, _ = encode(tok, fm["N"])
            seg = segments(tok, fm["P"], fm["P_result"], p_ids, n_ids, offs)
            diff, tail = seg["diff"], seg.get("result_tail", [])
            base = {"case_id": cid, "test_id": c["test_id"], "form": form}

            cp, lp = cache_resid(lm, p_ids, n_layers)
            cn, ln = cache_resid(lm, n_ids, n_layers)
            m_p, m_n = metric(lp, letter_ids, ti).item(), metric(ln, letter_ids, ti).item()
            gap = m_p - m_n
            base_m = dict(base, m_P=m_p, m_N=m_n)

            # A. joint patching
            pos_sets = {"diff": diff, "result_tail": tail, "both": diff + tail}
            for pname, positions in pos_sets.items():
                if not positions:
                    continue
                for direction, run_ids, src in [("denoise", n_ids, cp), ("noise", p_ids, cn)]:
                    ms = patch_resid_layers(lm, run_ids, src, positions, joint_layers, letter_ids, ti)
                    for l, mp in zip(joint_layers, ms):
                        rec = (mp - m_n) / gap if direction == "denoise" else (m_p - mp) / gap
                        f_joint.write(json.dumps(dict(base_m, positions=pname, direction=direction,
                                                      layer=l, m_patched=mp, recovery=rec)) + "\n")

            # B. logit lens
            lens_pos = {**{f"diff{i}": p for i, p in enumerate(diff)},
                        **{f"tail{i}": p for i, p in enumerate(tail)}, "final": len(p_ids) - 1}
            for pname, p in lens_pos.items():
                tok_p, tok_n = tok.decode([p_ids[p]]), tok.decode([n_ids[p]])
                for l in range(n_layers):
                    sp, sn = lens(hf, cp[l][p]), lens(hf, cn[l][p])
                    row = dict(base, position=pname, token_P=tok_p, token_N=tok_n, layer=l,
                               letters_P=sp[letter_ids].tolist(), letters_N=sn[letter_ids].tolist(),
                               m_lens_P=metric(sp, letter_ids, ti).item(),
                               m_lens_N=metric(sn, letter_ids, ti).item())
                    if l in lens_layers:
                        d = sn - sp
                        row["top_N_over_P"] = [tok.decode([i]) for i in d.topk(TOPK).indices.tolist()]
                        row["top_P_over_N"] = [tok.decode([i]) for i in (-d).topk(TOPK).indices.tolist()]
                        row["top_P"] = [tok.decode([i]) for i in sp.topk(TOPK).indices.tolist()]
                        row["top_N"] = [tok.decode([i]) for i in sn.topk(TOPK).indices.tolist()]
                    f_lens.write(json.dumps(row) + "\n")

            # C. heads
            hp, hn = cache_heads(lm, p_ids, head_layers), cache_heads(lm, n_ids, head_layers)
            after = list(range(diff[0], len(p_ids)))
            veto = run_with(lm, p_ids, letter_ids, ti, hd, resid_layer=args.veto_layer,
                            resid_src=cn, resid_pos=diff)
            for g, hs in groups.items():
                res = {
                    "swap_denoise": (run_with(lm, n_ids, letter_ids, ti, hd, hs, hp, after) - m_n) / gap,
                    "swap_noise": (m_p - run_with(lm, p_ids, letter_ids, ti, hd, hs, hn, after)) / gap,
                    "zero_N": (run_with(lm, n_ids, letter_ids, ti, hd, hs, None, after) - m_n) / gap,
                    "zero_P": (m_p - run_with(lm, p_ids, letter_ids, ti, hd, hs, None, after)) / gap,
                    "veto": (m_p - veto) / gap,
                    "veto_clamped": (m_p - run_with(lm, p_ids, letter_ids, ti, hd, hs, hp, after,
                                                    resid_layer=args.veto_layer, resid_src=cn,
                                                    resid_pos=diff)) / gap,
                }
                for cond, rec in res.items():
                    f_heads.write(json.dumps(dict(base_m, group=g, heads=[f"{l}.{h}" for l, h in hs],
                                                  condition=cond, recovery=rec)) + "\n")
            print(f"  {cid} {form}  gap {gap:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    for f in (f_joint, f_lens, f_heads):
        f.close()
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
