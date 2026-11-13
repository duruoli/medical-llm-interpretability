"""Score data/sigmoid_pool.jsonl and compare the model's belief moves with literature LRs.

Behavior only (one forward pass per prompt, last-position logits). For each case, test
(ct_sigmoid / ct_appendix), question format (mc4 / yesno) and wording pair:
    up   = m_P - m_B      how far a positive result pushes the target up
    down = m_B - m_N      how far a negative result pushes it down (< 0: wrong direction)
in log-odds units, where m is the target's log-odds (see build_sigmoid_cases.py).

An ideal Bayesian moves log-odds by ln(LR+) for P and -ln(LR-) for N, whatever the prior.
So the main number is the ratio mean(up) / mean(down) on mid-prior cases, compared with the
literature ratio. Mid prior = baseline P(target) in --pB-range (both ends saturate in the
model, which is itself un-Bayesian but would confound the ratio).

Literature (see results/PROGRESS_posneg.md / chat notes, pooled meta-analyses):
  CT for diverticulitis  sens 0.94, spec 0.99 (CI 0.90-1.00)  Lameris 2008 Eur Radiol
  CT for appendicitis    sens 0.95, spec 0.94 (contrast CT)   pooled meta-analyses

Run (repo root):
  python sigmoid_lr.py --model Qwen/Qwen3.5-9B                          # Quest, ~10-15 min
  python sigmoid_lr.py --model Qwen/Qwen3.5-0.8B --limit 3              # local smoke test
Output: results/sigmoid_lr/<model_tag>/{scores.jsonl, summary.md, run_config.json}
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from patch_posneg import LETTERS, encode, metric

LIT = {  # test -> (sens, spec, spec_low) ; spec_low = lower CI bound, gives the low LR+ end
    "ct_sigmoid": (0.94, 0.99, 0.96),
    "ct_appendix": (0.95, 0.94, 0.94),
}
PAIRS = ["lexical", "impression", "full_impression", "findings", "numeric"]  # numeric: appendix only
FORMATS = ["mc4", "yesno"]


def lit_moves(test):
    sens, spec, spec_lo = LIT[test]
    up_hi = math.log(sens / (1 - spec))
    up_lo = math.log(sens / (1 - spec_lo))
    down = math.log(spec / (1 - sens))
    return up_lo, up_hi, down


@torch.no_grad()
def logits_last(hf, tok, text, device):
    ids = torch.tensor([encode(tok, text)[1]], device=device)
    return hf(ids, logits_to_keep=1).logits[0, -1].float()


def m_mc4(lg, letter_ids, ti):
    return metric(lg, letter_ids, ti).item(), torch.softmax(lg[letter_ids], -1)[ti].item()


def m_yesno(lg, yn_ids):
    l2 = lg[yn_ids]
    return (l2[0] - l2[1]).item(), torch.softmax(l2, -1)[0].item()


def ratio_ci(up, down, rng, n_boot=2000):
    """Bootstrap over cases of mean(up)/mean(down)."""
    up, down = np.asarray(up), np.asarray(down)
    idx = rng.integers(0, len(up), size=(n_boot, len(up)))
    mu, md = up[idx].mean(1), down[idx].mean(1)
    r = np.where(md > 0, mu / np.where(md > 0, md, 1), 99.0)   # 99 = negative moved wrong way
    return np.percentile(r, 2.5), np.percentile(r, 97.5)


def summary(rows, pB_range):
    rng = np.random.default_rng(0)
    lines = ["# Sigmoid CT vs appendix CT: belief moves vs literature likelihood ratios\n",
             "up = m_P - m_B, down = m_B - m_N (log-odds; +2.3 = odds x10). "
             "ratio = mean(up)/mean(down), 95% bootstrap CI over cases. "
             f"Mid prior = baseline P(target) in {pB_range}. "
             "n_neg_raises = cases where the negative result RAISED the target (down < 0); "
             "n_pos_lowers = cases where the positive result LOWERED it (up < 0).\n"]
    for test in LIT:
        pairs = [p for p in PAIRS if p in rows[0][test]["mc4"]]
        up_lo, up_hi, down = lit_moves(test)
        lines.append(f"## {test}\n")
        lines.append(f"Literature ideal: up {up_lo:.1f}-{up_hi:.1f}, down {down:.1f}, "
                     f"ratio {up_lo / down:.2f}-{up_hi / down:.2f}\n")
        tab = []
        for fmt in FORMATS:
            for subset in ["mid", "all"]:
                rs = [r for r in rows if subset == "all" or
                      pB_range[0] <= r[test][fmt]["pB"] <= pB_range[1]]
                if not rs:
                    continue
                for pair in pairs:
                    up = [r[test][fmt][pair]["up"] for r in rs]
                    dn = [r[test][fmt][pair]["down"] for r in rs]
                    mu, md = float(np.mean(up)), float(np.mean(dn))
                    lo, hi = ratio_ci(up, dn, rng)
                    tab.append([fmt, subset, len(rs), pair, round(mu, 2), round(md, 2),
                                round(mu / md, 2) if md > 0 else "inf",
                                f"{lo:.2f}-{hi:.2f}", sum(d < 0 for d in dn), sum(u < 0 for u in up)])
        lines.append(tabulate(tab, headers=["format", "cases", "n", "pair", "up", "down", "ratio",
                                            "ratio 95% CI", "n_neg_raises", "n_pos_lowers"],
                              tablefmt="github"))
        lines.append("")

        # which cases go the wrong way (all cases; listed so they can be read one by one)
        wrong = []
        for fmt in FORMATS:
            for pair in pairs:
                for r in rows:
                    d = r[test][fmt][pair]
                    if d["down"] < 0 or d["up"] < 0:
                        wrong.append([fmt, pair, r["hadm_id"], r["true_disease"],
                                      round(r[test][fmt]["pB"], 2), round(d["up"], 2),
                                      round(d["down"], 2)])
        lines.append(f"### {test}: wrong-direction cases (all priors)\n")
        lines.append(tabulate(wrong, headers=["format", "pair", "hadm_id", "true dz", "pB",
                                              "up", "down"], tablefmt="github")
                     if wrong else "none")
        lines.append("")

        # does the negative fail more for some true diseases? (mid prior, mc4 + yesno)
        tab = []
        for fmt in FORMATS:
            for dz in sorted({r["true_disease"] for r in rows}):
                rs = [r for r in rows if r["true_disease"] == dz
                      and pB_range[0] <= r[test][fmt]["pB"] <= pB_range[1]]
                if rs:
                    tab.append([fmt, dz, len(rs)] + [round(float(np.mean(
                        [r[test][fmt][p]["down"] for r in rs])), 2) for p in pairs])
        lines.append(f"### {test}: mean down by true disease (mid prior)\n")
        lines.append(tabulate(tab, headers=["format", "true dz", "n"] + pairs, tablefmt="github"))
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--out-root", default="results/sigmoid_lr")
    ap.add_argument("--pB-range", type=float, nargs=2, default=[0.02, 0.95])
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke test)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_root) / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(
        dict(vars(args), device=device, dtype=str(dtype), literature=LIT), indent=2))

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    letter_ids = tok.convert_tokens_to_ids(LETTERS)
    yn_ids = tok.convert_tokens_to_ids(["Yes", "No"])

    pool = [json.loads(l) for l in open(args.pool)]
    if args.limit:
        pool = pool[:args.limit]

    t0, rows = time.time(), []
    with (out / "scores.jsonl").open("w") as f:
        for i, c in enumerate(pool):
            row = {"hadm_id": c["hadm_id"], "true_disease": c["true_disease"]}
            lg_B = logits_last(hf, tok, c["mc4_B"], device)       # shared by both tests
            for test, t in c["tests"].items():
                ti = LETTERS.index(t["target_letter"])
                res = {}
                for fmt in FORMATS:
                    if fmt == "mc4":
                        score = lambda text: m_mc4(logits_last(hf, tok, text, device), letter_ids, ti)
                        mB, pB = m_mc4(lg_B, letter_ids, ti)
                    else:
                        score = lambda text: m_yesno(logits_last(hf, tok, text, device), yn_ids)
                        mB, pB = score(t["yesno_B"])
                    d = {"mB": mB, "pB": pB}
                    for pair, pr in t["pairs"].items():
                        mP, pP = score(pr[f"{fmt}_P"])
                        mN, pN = score(pr[f"{fmt}_N"])
                        d[pair] = {"mP": mP, "mN": mN, "pP": pP, "pN": pN,
                                   "up": mP - mB, "down": mB - mN}
                    res[fmt] = d
                row[test] = res
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            print(f"  {i + 1}/{len(pool)} {c['hadm_id']} ({time.time() - t0:.0f}s)", flush=True)

    (out / "summary.md").write_text(summary(rows, args.pB_range))
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
