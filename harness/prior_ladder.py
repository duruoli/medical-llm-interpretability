"""Controlled-prior ladder (PLAN_crossmodel.md, question 2: the SHAPE of the compression curve).

Stage 1 §1c bins patients by the model's OWN no-result belief, so "where a model sits" and "how its curve
looks" are mixed (MedGemma sits on the floor; is its curve different, or only its position?). Here we SET
the starting point: every patient is scored once per level, with one fixed sentence stating a pre-test
probability, inserted after the history (before "New test result:"). The sentence is identical in the
no-result (B), positive (P) and negative (N) prompt of that level, so up = P - B and down = B - N compare
like with like. Levels: "none" (= the normal prompt) and the stated probabilities.

  python -m harness.prior_ladder --model Qwen/Qwen3-8B --limit-cases 60 --out /scratch/.../Qwen3-8B
  -> <out>/ladder[_q_<question>]/{REPORT.md, report.json, margins.jsonl, sanity.json}

REPORT.md:
  A  adoption   does the model's own no-result P(Yes) follow the stated probability? does the history
                still matter inside one level (AUC for the true disease)?  P(Yes)+P(No) per level.
  B  per level  up/ideal, down/ideal, asym per family at each stated level
  C  curve      all levels pooled, binned by the model's ACTUAL no-result belief (same patients in every
                bin, so bins are no longer filled by different kinds of patients)
  D  shape      per family: slope of asym per unit log-odds, balance point (prior where asym = 0),
                floor (down/ideal in the lowest bin) and ceiling (up/ideal in the highest bin)
"""
import argparse
import copy
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from harness.config import lit_moves
from harness.diagnose import (FAMILIES, PRIOR_BIN_LABELS, PRIOR_BINS, Q_RE, QUESTIONS, TEST_DISEASE, Runner, auc,
                              build_items, flatten, reword, unflatten)

LEVELS = [0.01, 0.05, 0.20, 0.50, 0.80]
SENTENCE = "Before this test result, the estimated probability of {d} for this patient is about {x}%."
ANCHOR = "\n\nNew test result:"


def lab(level):
    return "none" if level is None else f"{level * 100:g}%"


def with_level(text, level, question):
    """Insert the stated pre-test probability (if any), then apply the question wording."""
    if level is not None:
        d = Q_RE.search(text).group(1)
        if text.count(ANCHOR) != 1:
            raise ValueError("prompt does not have exactly one 'New test result:' block")
        text = text.replace(ANCHOR, "\n\n" + SENTENCE.format(d=d, x=f"{level * 100:g}") + ANCHOR)
    return text if question == "orig" else reword(text, question)


def ladder_items(base, levels, question):
    out = []
    for level in levels:
        items = copy.deepcopy(base)
        for it in items:
            for t in it["tests"].values():
                t["B"] = with_level(t["B"], level, question)
                for pr in t["pairs"].values():
                    pr["P"], pr["N"] = with_level(pr["P"], level, question), with_level(pr["N"], level, question)
        out.append(items)
    return out


# ---- analysis -------------------------------------------------------------------------------
def sig(m):
    return 1 / (1 + math.exp(-m))


def family_curve(rows, bins=(0.0, 1.0001), min_n=15):
    """Per family and bin of the model's no-result P(Yes): n, mean log-odds, up/ideal, down/ideal, asym."""
    out = {}
    for fam, tests in FAMILIES.items():
        row = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            ups, dns, ms = [], [], []
            for t in tests:
                lu, ld = lit_moves(t)
                for r in rows:
                    if lo <= sig(r[t]["B"]) < hi:
                        ps = [p for p in r[t] if isinstance(r[t][p], dict)]
                        ups.append(np.mean([r[t][p]["P"] - r[t]["B"] for p in ps]) / lu)
                        dns.append(np.mean([r[t]["B"] - r[t][p]["N"] for p in ps]) / ld)
                        ms.append(r[t]["B"])
            ok = len(ups) >= min_n
            row.append({"n": len(ups), "mean_logodds": float(np.mean(ms)) if ms else None,
                        "up_over_ideal": float(np.mean(ups)) if ok else None,
                        "down_over_ideal": float(np.mean(dns)) if ok else None,
                        "asym": math.log(max(np.mean(ups), 0.03) / max(np.mean(dns), 0.03)) if ok else None})
        out[fam] = row
    return out


def shape(curve):
    """Weighted line asym = a + b * log-odds over the populated bins of one family."""
    out = {}
    for fam, row in curve.items():
        pts = [(c["mean_logodds"], c["asym"], c["n"]) for c in row if c["asym"] is not None]
        res = {"n_bins": len(pts)}
        if len(pts) >= 3:
            x, y, w = map(np.array, zip(*pts))
            b, a = np.polyfit(x, y, 1, w=np.sqrt(w))
            x0 = -a / b if b < 0 else None      # balance only where the data are (no extrapolation)
            res.update(slope=float(b), intercept=float(a),
                       balance_p=float(sig(x0)) if x0 is not None and x.min() <= x0 <= x.max() else None)
        ok = [c for c in row if c["asym"] is not None]
        if ok:
            res.update(floor_down=ok[0]["down_over_ideal"], floor_bin=PRIOR_BIN_LABELS[row.index(ok[0])],
                       ceiling_up=ok[-1]["up_over_ideal"], ceiling_bin=PRIOR_BIN_LABELS[row.index(ok[-1])])
        out[fam] = res
    return out


def adoption(rows, items):
    """Per family: median no-result P(Yes), its IQR, and AUC for the true disease inside this level."""
    out = {}
    for fam, tests in FAMILIES.items():
        pb = [sig(r[t]["B"]) for r in rows for t in tests]
        pos, neg = [], []
        for it, r in zip(items, rows):
            for t in tests:
                if TEST_DISEASE.get(t):
                    (pos if it["true_disease"] == TEST_DISEASE[t] else neg).append(r[t]["B"])
        out[fam] = {"pB_median": float(np.median(pb)), "pB_iqr": [float(np.percentile(pb, 25)), float(np.percentile(pb, 75))],
                    "auc_true": auc(pos, neg) if pos and neg else None}
    return out


def report(levels, rows_by, items, mass_by, args, sanity):
    L = [f"# Controlled-prior ladder: {args.model}  (question: {QUESTIONS[args.question]})", "",
         f"{len(items)} patients x levels {[lab(l) for l in levels]}. Sentence (after the history, same in B/P/N): "
         f"\"{SENTENCE}\"", "",
         "asym = ln(up/ideal ÷ down/ideal); Bayes: 0 at every starting point. up/ideal, down/ideal: 1 = literature.", "",
         f"Yes/No tokens: {json.dumps(sanity['token_check'])}", "",
         "## A Does the model take the stated probability? (no-result prompt)", "",
         "median P(Yes) [IQR]; AUC = does the history still separate true cases inside one level (0.5 = no).", "",
         "| family | " + " | ".join(lab(l) for l in levels) + " |", "|---|" + "---|" * len(levels)]
    ad = {lab(l): adoption(rows_by[k], items) for k, l in enumerate(levels)}
    for fam in FAMILIES:
        cells = []
        for l in levels:
            a = ad[lab(l)][fam]
            au = "" if a["auc_true"] is None else f" AUC {a['auc_true']:.2f}"
            cells.append(f"{a['pB_median']:.3f} [{a['pB_iqr'][0]:.3f}–{a['pB_iqr'][1]:.3f}]{au}")
        L.append(f"| {fam} | " + " | ".join(cells) + " |")
    L += ["", "P(Yes)+P(No), mean over all prompts of a level: "
          + ", ".join(f"{lab(l)} {mass_by[k]:.3f}" for k, l in enumerate(levels))
          + "  (well below 0.9 = the sentence breaks the answer format; read that level with care)", "",
          "## B By stated level", "", "cell = asym (up/ideal, down/ideal)", "",
          "| family | " + " | ".join(lab(l) for l in levels) + " |", "|---|" + "---|" * len(levels)]
    per_level = {lab(l): family_curve(rows_by[k]) for k, l in enumerate(levels)}
    for fam in FAMILIES:
        cells = []
        for l in levels:
            c = per_level[lab(l)][fam][0]
            cells.append("–" if c["asym"] is None else
                         f"{c['asym']:+.2f} ({c['up_over_ideal']:.2f}, {c['down_over_ideal']:.2f})")
        L.append(f"| {fam} | " + " | ".join(cells) + " |")
    pooled = [r for rs in rows_by for r in rs]
    curve = family_curve(pooled, PRIOR_BINS)
    L += ["", "## C Curve: all levels pooled, binned by the model's ACTUAL no-result P(Yes)", "",
          "cell = asym (up/ideal, down/ideal; n)", "",
          "| family | " + " | ".join(PRIOR_BIN_LABELS) + " |", "|---|" + "---|" * len(PRIOR_BIN_LABELS)]
    for fam, row in curve.items():
        L.append(f"| {fam} | " + " | ".join(
            "–" if c["asym"] is None else f"{c['asym']:+.2f} ({c['up_over_ideal']:.2f}, {c['down_over_ideal']:.2f}; {c['n']})"
            for c in row) + " |")
    sh = shape(curve)
    L += ["", "## D Shape (from C)", "",
          "slope = change of asym per +1 log-odds of the starting point (more negative = stronger compression); "
          "balance = starting P(Yes) where asym = 0; floor = down/ideal in the lowest populated bin; "
          "ceiling = up/ideal in the highest.", "",
          "| family | slope | balance P(Yes) | floor down/ideal (bin) | ceiling up/ideal (bin) |", "|---|---|---|---|---|"]
    f = lambda v, fmt: "–" if v is None else format(v, fmt)
    for fam, s in sh.items():
        L.append(f"| {fam} | {f(s.get('slope'), '+.2f')} | {f(s.get('balance_p'), '.3f')} | "
                 f"{f(s.get('floor_down'), '.2f')} ({s.get('floor_bin', '–')}) | "
                 f"{f(s.get('ceiling_up'), '.2f')} ({s.get('ceiling_bin', '–')}) |")
    return L, {"adoption": ad, "per_level": per_level, "curve": curve, "shape": sh}


# ---- run ------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--out", default=None, help="model directory (default results/diagnose/<model name>)")
    ap.add_argument("--question", default="orig", choices=list(QUESTIONS))
    ap.add_argument("--levels", type=float, nargs="+", default=LEVELS, help="stated pre-test probabilities (0-1)")
    ap.add_argument("--limit-cases", type=int, default=60, help="stratified subset (0 = all 188)")
    ap.add_argument("--batch", type=int, default=1, help="1 = exact (default); >1 drifts up to 0.25 on Llama bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from-margins", action="store_true", help="no GPU: re-analyse <out>/ladder*/margins.jsonl")
    args = ap.parse_args()
    out = Path(args.out or f"results/diagnose/{args.model.split('/')[-1]}")
    out = out / ("ladder" if args.question == "orig" else f"ladder_q_{args.question}")
    out.mkdir(parents=True, exist_ok=True)
    levels = [None] + sorted(args.levels)

    base = build_items(args.pool, args.limit_cases, "orig")
    by_level = ladder_items(base, levels, args.question)
    if args.from_margins:
        recs = [json.loads(l) for l in open(out / "margins.jsonl")]
        sanity = json.loads((out / "sanity.json").read_text())
        rows_by = [[next(r["tests"] for r in recs if r["hadm_id"] == it["hadm_id"] and r["level"] == lab(l))
                    for it in base] for l in levels]
        mass_by = sanity["mass_by_level"]
    else:
        t0 = time.time()
        keys, texts = [], []
        for k, items in enumerate(by_level):
            kk, tt = flatten(items)
            keys += [(k, key) for key in kk]
            texts += tt
        print(f"{len(base)} patients x {len(levels)} levels = {len(texts)} prompts; loading {args.model}", flush=True)
        R = Runner(args.model, args.device, args.batch)
        tc = R.token_check()
        res = R.margins(texts)
        print(f"scored ({time.time() - t0:.0f}s)", flush=True)
        rows_by, mass_by = [], []
        for k, items in enumerate(by_level):
            idx = [j for j, (kk, _) in enumerate(keys) if kk == k]
            rows_by.append(unflatten(items, [keys[j][1] for j in idx], [res[j] for j in idx]))
            mass_by.append(float(np.mean([res[j]["mass"] for j in idx])))
        with (out / "margins.jsonl").open("w") as fh:
            for k, l in enumerate(levels):
                for it, r in zip(base, rows_by[k]):
                    fh.write(json.dumps({"hadm_id": it["hadm_id"], "true_disease": it["true_disease"],
                                         "level": lab(l), "tests": r}) + "\n")
        sanity = {"token_check": tc, "mass_by_level": mass_by, "n_prompts": len(texts), "batch": args.batch,
                  "seconds": round(time.time() - t0), "levels": [lab(l) for l in levels]}
        (out / "sanity.json").write_text(json.dumps(sanity, indent=1))

    L, rep = report(levels, rows_by, base, mass_by, args, sanity)
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps({"model": args.model, "question": args.question,
                                                 "sentence": SENTENCE, **rep}, indent=1, default=str))
    print(f"-> {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
