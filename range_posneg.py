"""Does the model compare the number with the stated reference range?

Behavior only (no patching). For every pool case, the numeric result line is rewritten with
the value held fixed and the reference range changed:

    value hi (the P value, e.g. 9 mm)  x  range orig   "normal under 6 mm"   -> abnormal
                                          range flip   "normal under 12 mm"  -> normal
                                          range none   (no range given)
    value lo (the N value, e.g. 4 mm)  x  range orig                         -> normal
                                          range flip   "normal under 3 mm"   -> abnormal
                                          range none

If the model uses the range, changing only the range (value fixed) moves the answer in the
same direction as changing the value: hi+flip looks like lo+orig, lo+flip like hi+orig.
If it ignores the range (e.g. falls back on what it knows about normal sizes), the range
effect is ~0. The flipped ranges are clinically wrong on purpose; a model that trusts its
own knowledge over the stated range will discount them, which is also an answer.

Run (repo root):
  python range_posneg.py --model Qwen/Qwen3.5-9B                          # Quest, a few min
  python range_posneg.py --model Qwen/Qwen3.5-0.8B --pool-per-test 1      # local smoke test
Output: results/range/<model_tag>/{range.jsonl, range_summary.md, run_config.json}
"""
import argparse
import json
import time
from pathlib import Path

import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from patch_posneg import LETTERS, encode, metric

# test_id -> numeric template with {v} and {range}; hi / lo values; range text per setting.
# "orig" is the range used in build_posneg_cases.py; "flip_hi" makes hi normal, "flip_lo"
# makes lo abnormal.
RANGES = {
    "lipase": {
        "tmpl": "Serum lipase: {v} U/L{range}.", "hi": "980", "lo": "120",
        "orig": " (reference interval 0-160 U/L)",
        "flip_hi": " (reference interval 0-1200 U/L)",
        "flip_lo": " (reference interval 0-60 U/L)",
    },
    "ct_appendix": {
        "tmpl": "CT abdomen/pelvis: appendix diameter {v} mm{range}.", "hi": "9", "lo": "4",
        "orig": " (normal under 6 mm)",
        "flip_hi": " (normal under 12 mm)",
        "flip_lo": " (normal under 3 mm)",
    },
    "us_gallbladder": {
        "tmpl": "Right upper quadrant ultrasound: gallbladder wall thickness {v} mm{range}.",
        "hi": "6", "lo": "2",
        "orig": " (normal up to 3 mm)",
        "flip_hi": " (normal up to 8 mm)",
        "flip_lo": " (normal up to 1 mm)",
    },
    "ct_sigmoid": {
        "tmpl": "CT abdomen/pelvis: sigmoid colon wall thickness {v} mm{range}.", "hi": "8", "lo": "2",
        "orig": " (normal up to 3 mm)",
        "flip_hi": " (normal up to 10 mm)",
        "flip_lo": " (normal up to 1 mm)",
    },
}
VARIANTS = [("hi", "orig"), ("hi", "flip"), ("hi", "none"),
            ("lo", "orig"), ("lo", "flip"), ("lo", "none")]


def result_line(spec, value, rng):
    text = {"orig": spec["orig"], "flip": spec[f"flip_{value}"], "none": ""}[rng]
    return spec["tmpl"].format(v=spec[value], range=text)


@torch.no_grad()
def score_m(hf, tok, text, letter_ids, ti, device):
    ids = torch.tensor([encode(tok, text)[1]], device=device)
    logits = hf(ids, logits_to_keep=1).logits[0, -1].float()
    return metric(logits, letter_ids, ti).item(), torch.softmax(logits[letter_ids], -1)[ti].item()


def summary(rows):
    """Per case: m relative to the pending baseline B, and the three contrasts."""
    out = ["# Value vs reference range (numeric form)\n",
           "m = logit(target) - logsumexp(other 3 letters), shown as m - m_B (B = result pending).",
           "value effect = hi_orig - lo_orig (the original P - N gap).",
           "range effect (hi) = hi_orig - hi_flip: how much calling 9 mm 'normal' lowers the target.",
           "range effect (lo) = lo_flip - lo_orig: how much calling 4 mm 'abnormal' raises it.",
           "If the range is used, both range effects are positive and comparable to the value effect.\n"]
    for test_id in RANGES:
        rs = [r for r in rows if r["test_id"] == test_id]
        if not rs:
            continue
        spec = RANGES[test_id]
        out.append(f"## {test_id}: hi={spec['hi']}, lo={spec['lo']}; orig{spec['orig']}, "
                   f"hi flip{spec['flip_hi']}, lo flip{spec['flip_lo']}\n")
        tab = []
        for r in rs:
            d = {k: r["m"][k] - r["m"]["B"] for k in r["m"] if k != "B"}
            tab.append([r["case_id"].split("__")[1], r["true_disease"][:6], round(r["p"]["B"], 2)]
                       + [round(d[f"{v}_{g}"], 2) for v, g in VARIANTS]
                       + [round(d["hi_orig"] - d["lo_orig"], 2),
                          round(d["hi_orig"] - d["hi_flip"], 2),
                          round(d["lo_flip"] - d["lo_orig"], 2)])
        out.append(tabulate(tab, headers=["case", "true dz", "pB"] + [f"{v}_{g}" for v, g in VARIANTS]
                            + ["value eff", "range eff hi", "range eff lo"], tablefmt="github"))
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--pool", default="data/posneg_pool.jsonl")
    ap.add_argument("--out-root", default="results/range")
    ap.add_argument("--pool-per-test", type=int, default=10)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_root) / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(dict(vars(args), device=device, dtype=str(dtype),
                                                         ranges=RANGES), indent=2))

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    letter_ids = tok.convert_tokens_to_ids(LETTERS)

    pool = [json.loads(l) for l in open(args.pool)]
    per_test = {}
    for c in pool:
        per_test.setdefault(c["test_id"], []).append(c)
    pool = [c for cs in per_test.values() for c in cs[:args.pool_per_test]]

    t0 = time.time()
    rows = []
    with (out / "range.jsonl").open("w") as f:
        for c in pool:
            spec = RANGES[c["test_id"]]
            ti = LETTERS.index(c["target_letter"])
            orig_p = c["forms"]["numeric"]["P"]
            orig_line = c["forms"]["numeric"]["P_result"]
            assert result_line(spec, "hi", "orig") == orig_line, (c["case_id"], orig_line)
            row = {k: c[k] for k in ["case_id", "test_id", "true_disease", "target_letter"]}
            row["m"], row["p"], row["lines"] = {}, {}, {}
            row["m"]["B"], row["p"]["B"] = score_m(hf, tok, c["B"], letter_ids, ti, device)
            for v, g in VARIANTS:
                line = result_line(spec, v, g)
                key = f"{v}_{g}"
                row["lines"][key] = line
                row["m"][key], row["p"][key] = score_m(hf, tok, orig_p.replace(orig_line, line),
                                                       letter_ids, ti, device)
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            print(f"  {c['case_id']}  ({time.time() - t0:.0f}s)", flush=True)
    (out / "range_summary.md").write_text(summary(rows))
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
