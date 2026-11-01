"""Activation patching between positive (P) and negative (N) test evidence.

Stage 1 - behavior: score every pool case (from build_posneg_cases.py) under P / N / B
  for each surface form. Metric at the answer position:
      m = logit(target letter) - logsumexp(logits of the other three letters)
  Keep the first k eligible cases per test (default 5 x 4 tests = 20).
  Eligible = baseline P(target) not at floor/ceiling. Selection does NOT look at the P-N
  gap: a form where the negative result barely moves the model is itself the finding.
  The gap is only used downstream: each (case, form) is flagged gap_ok = gap >= --min-gap,
  and plot_posneg.py builds heatmaps from gap_ok pairs (recovery is unstable on tiny gaps).

Stage 2 - residual-stream patching, both directions, for every kept case x form:
  denoise : run N, overwrite the residual stream (decoder-layer output) at one segment of
            positions in one layer with P's activations.
            recovery = (m_patched - m_N) / (m_P - m_N)
            -> is the info at this site ENOUGH to move N's answer to P's?
  noise   : run P, overwrite with N's activations.
            recovery = (m_P - m_patched) / (m_P - m_N)
            -> does P's answer DEPEND on this site?
  Both are "fraction of the P-N gap moved", so the two directions share one scale.

  Segments (only positions at or after the first differing token can matter; earlier
  positions are identical in P and N):
      diff        tokens that differ between P and N ("980" vs "120", "acute" vs "no")
      result_tail rest of the result line after the differing tokens
      question    the question + answer options + instruction
      template    chat-template tokens before the final position
      final       the last token (where the answer is read)

  All layers are patched in ONE forward: the input is repeated n_layers times as a
  batch, and batch row i is patched at layer i.

Run (repo root), local smoke test:
  python patch_posneg.py \
      --model Qwen/Qwen3.5-0.8B --pool-per-test 3 --select-per-test 1
Full run on Quest: sbatch patch_posneg.slurm
Output: results/posneg/<model_tag>/{behavior.jsonl, selected.json, patching.jsonl, run_config.json}
"""
import argparse
import json
import time
from pathlib import Path

import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]
FORMS = ["numeric", "lexical", "negation"]
SEGMENTS = ["diff", "result_tail", "question", "template", "final"]


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def encode(tok, text):
    s = chat(tok, text)
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    return s, enc.input_ids, enc.offset_mapping


def metric(logits_last, letter_ids, target_idx):
    """logits_last: [..., vocab] -> m: [...]"""
    l4 = logits_last[..., letter_ids].float()
    others = torch.cat([l4[..., :target_idx], l4[..., target_idx + 1:]], dim=-1)
    return l4[..., target_idx] - torch.logsumexp(others, dim=-1)


@torch.no_grad()
def score(lm, ids, letter_ids, target_idx):
    with lm.trace(torch.tensor([ids])):
        logits = lm.lm_head.output[:, -1, :].save()
    logits = logits[0].float()
    l4 = logits[letter_ids]
    return {
        "m": metric(logits, letter_ids, target_idx).item(),
        "p4": torch.softmax(l4, -1).tolist(),                       # among the 4 letters
        "letter_mass": torch.softmax(logits, -1)[letter_ids].sum().item(),  # vs whole vocab
    }


def segments(tok, text, result, p_ids, n_ids, offsets):
    """Map each segment name -> list of token positions."""
    diff = [i for i, (a, b) in enumerate(zip(p_ids, n_ids)) if a != b]
    s = chat(tok, text)
    r0 = s.index("New test result:\n" + result) + len("New test result:\n")
    r1 = r0 + len(result)
    u1 = s.rindex("<|im_end|>")             # end of the user message
    L = len(p_ids)
    seg = {k: [] for k in SEGMENTS}
    seg["diff"] = diff
    for i, (a, b) in enumerate(offsets):
        if i <= diff[-1] or i == L - 1:
            continue
        if a < r1:
            seg["result_tail"].append(i)
        elif a < u1:
            seg["question"].append(i)
        else:
            seg["template"].append(i)
    seg["final"] = [L - 1]
    return {k: v for k, v in seg.items() if v}


@torch.no_grad()
def cache_resid(lm, ids, n_layers):
    hs = []
    with lm.trace(torch.tensor([ids])):
        # explicit loop: nnsight cannot trace inside a list comprehension's own scope
        for l in range(n_layers):
            hs.append(lm.model.layers[l].output.save())
        logits = lm.lm_head.output[:, -1, :].save()
    return [h[0].clone() for h in hs], logits[0]


@torch.no_grad()
def patch_all_layers(lm, run_ids, src_cache, positions, n_layers, letter_ids, target_idx):
    """Row l of the batch = run_ids with layer-l residual at `positions` taken from src."""
    batch = torch.tensor([run_ids] * n_layers)
    pos = torch.tensor(positions)
    with lm.trace(batch):
        for l in range(n_layers):
            lm.model.layers[l].output[l, pos, :] = src_cache[l][pos, :]
        logits = lm.lm_head.output[:, -1, :].save()
    return metric(logits, letter_ids, target_idx).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--pool", default="data/posneg_pool.jsonl")
    ap.add_argument("--out-root", default="results/posneg")
    ap.add_argument("--pool-per-test", type=int, default=10, help="score at most this many per test")
    ap.add_argument("--select-per-test", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=1.0,
                    help="(case, form) pairs with m_P - m_N below this are flagged gap_ok=False")
    ap.add_argument("--pB-range", type=float, nargs=2, default=[0.02, 0.95])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_root) / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    lm = LanguageModel(hf, tokenizer=tok)
    n_layers = hf.config.num_hidden_layers
    letter_ids = tok.convert_tokens_to_ids(LETTERS)
    layer_types = getattr(hf.config, "layer_types", None)
    (out / "run_config.json").write_text(json.dumps(
        dict(vars(args), device=device, dtype=str(dtype), n_layers=n_layers,
             layer_types=layer_types, segments=SEGMENTS), indent=2))

    pool = [json.loads(l) for l in open(args.pool)]
    per_test = {}
    for c in pool:
        per_test.setdefault(c["test_id"], []).append(c)
    pool = [c for cs in per_test.values() for c in cs[:args.pool_per_test]]

    # ---- stage 1: behavior ------------------------------------------------------
    t0 = time.time()
    beh = []
    with (out / "behavior.jsonl").open("w") as f:
        for c in pool:
            ti = LETTERS.index(c["target_letter"])
            row = {k: c[k] for k in ["case_id", "test_id", "hadm_id", "true_disease", "target_letter"]}
            row["B"] = score(lm, encode(tok, c["B"])[1], letter_ids, ti)
            for form in FORMS:
                for cond in ["P", "N"]:
                    row[f"{form}_{cond}"] = score(lm, encode(tok, c["forms"][form][cond])[1], letter_ids, ti)
            beh.append(row)
            f.write(json.dumps(row) + "\n")
    print(f"behavior: {len(beh)} cases in {time.time() - t0:.0f}s", flush=True)

    # ---- selection --------------------------------------------------------------
    selected = []
    for test_id in per_test:
        rows = [r for r in beh if r["test_id"] == test_id]
        for r in rows:
            ti = LETTERS.index(r["target_letter"])
            r["pB_target"] = r["B"]["p4"][ti]
            r["eligible"] = args.pB_range[0] <= r["pB_target"] <= args.pB_range[1]
        keep = [r for r in rows if r["eligible"]][:args.select_per_test]
        if len(keep) < args.select_per_test:      # fill with the least extreme priors, flagged
            rest = sorted((r for r in rows if not r["eligible"]), key=lambda r: abs(r["pB_target"] - 0.5))
            keep += rest[:args.select_per_test - len(keep)]
        selected += [{"case_id": r["case_id"], "eligible": r["eligible"], "pB_target": r["pB_target"],
                      "gap": {f: r[f"{f}_P"]["m"] - r[f"{f}_N"]["m"] for f in FORMS}} for r in keep]
    (out / "selected.json").write_text(json.dumps(selected, indent=2))
    n_elig = sum(s["eligible"] for s in selected)
    print(f"selected {len(selected)} cases ({n_elig} with non-extreme baseline prior)", flush=True)

    # ---- stage 2: patching ------------------------------------------------------
    by_id = {c["case_id"]: c for c in pool}
    t0 = time.time()
    with (out / "patching.jsonl").open("w") as f:
        for s in selected:
            c = by_id[s["case_id"]]
            ti = LETTERS.index(c["target_letter"])
            for form in FORMS:
                fm = c["forms"][form]
                _, p_ids, offs = encode(tok, fm["P"])
                _, n_ids, _ = encode(tok, fm["N"])
                assert len(p_ids) == len(n_ids)
                seg = segments(tok, fm["P"], fm["P_result"], p_ids, n_ids, offs)
                cache_p, lp = cache_resid(lm, p_ids, n_layers)
                cache_n, ln = cache_resid(lm, n_ids, n_layers)
                m_p = metric(lp, letter_ids, ti).item()
                m_n = metric(ln, letter_ids, ti).item()
                gap = m_p - m_n
                for seg_name, positions in seg.items():
                    for direction, run_ids, src in [("denoise", n_ids, cache_p), ("noise", p_ids, cache_n)]:
                        ms = patch_all_layers(lm, run_ids, src, positions, n_layers, letter_ids, ti)
                        if seg_name == "final":   # last-layer residual at the answer position fixes the logits
                            want = m_p if direction == "denoise" else m_n
                            if abs(ms[-1] - want) > 0.05 * max(1.0, abs(gap)):
                                print(f"  WARNING sanity: {c['case_id']} {form} {direction} "
                                      f"last-layer final patch gives {ms[-1]:.3f}, expected {want:.3f}")
                        for layer, mp in enumerate(ms):
                            rec = (mp - m_n) / gap if direction == "denoise" else (m_p - mp) / gap
                            f.write(json.dumps({
                                "case_id": c["case_id"], "test_id": c["test_id"],
                                "eligible": s["eligible"], "gap_ok": gap >= args.min_gap,
                                "form": form, "direction": direction,
                                "segment": seg_name, "n_pos": len(positions), "layer": layer,
                                "m_patched": mp, "m_P": m_p, "m_N": m_n, "recovery": rec,
                            }) + "\n")
            print(f"  patched {c['case_id']}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
