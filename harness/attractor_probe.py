"""Where do the attractors come from?  Where is the patient difference lost once a result is added?

Behaviour (results/diagnose, 9 models): after a result, the final margin F = a + s*B across patients of one
(test, wording) cell, with s well below 1 (pull = 1 - s). Bayes would keep s = 1: the result ADDS a fixed amount,
so two patients 2 apart stay 2 apart. With s < 1 part of the patient difference is erased.

This script splits that erased part over the model's components. Batch 1 (exact), one forward per prompt.

1. Direct logit attribution at the answer position, final norm linearised per prompt:
       F = sum_c (w_yes - w_no) * g * x_c / rms(h_final),  c = embedding, attn_0, mlp_0, ... attn_L, mlp_L
   (checked: the sum reproduces the real logit margin, see `decomp_err` in the report).
   For one cell, regress each component's contribution on the patient's no-result margin B:
       slope_c(B prompt) sums to exactly 1 (it IS B); slope_c(result prompt) sums to s.
       delta_c = slope_c(result) - slope_c(B) sums to s - 1 = -pull.
   delta_c < 0: component c erases patient difference when a result is present  ("eraser").
   Cumulative delta over layers = how much of the difference is left at each depth.
   Also: s_unnorm = the same slope before the final norm (h . (w g)). If pull is mostly gone there, the
   final RMS norm (a large result vector shrinking everything else) is the cause, not a component.
2. Attention share of the answer position on the HISTORY tokens, per layer (mean over heads), no-result
   prompt vs result prompt. A drop = the result draws attention away from the history ("not read in").
   Needs eager attention; layers without softmax attention (linear / DeltaNet) are skipped.
3. Blank history ("No history available."): one prompt per cell and side. Consistency check only:
   does the history-free point fall on the cell's line, and where does it sit relative to the fixed point
   L = a / (1 - s)?

Mid-prior patients (no-result P(Yes) in PB_RANGE) are chosen from this run's own B, per test.

  python -m harness.attractor_probe --model meta-llama/Llama-3.1-8B-Instruct --out /projects/p53430/medllm/attractor/llama
  python -m harness.attractor_probe --model Qwen/Qwen3.5-9B --out /projects/p53430/medllm/attractor/qwen35
  python -m harness.attractor_probe --model <same> --out <same> --from-saved      # report only, no GPU

Output <out>/records.jsonl (ids + numbers only, no patient text; appended, a rerun resumes), REPORT.md, report.json.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from harness.config import PB_RANGE
from harness.diagnose import Runner, build_items

BLANK = "No history available."
LO, HI = math.log(PB_RANGE[0] / (1 - PB_RANGE[0])), math.log(PB_RANGE[1] / (1 - PB_RANGE[1]))
IMG = ["ct_sigmoid", "ct_appendix", "us_appendix", "us_gallbladder", "ct_kidney_stone"]


# ---- model hooks ------------------------------------------------------------------------------
class Probe:
    def __init__(self, R):
        self.R, hf = R, R.hf
        cands = [n for n, _ in hf.named_modules() if re.fullmatch(r".*\.layers\.0", n) and "vision" not in n]
        self.prefix = cands[-1][:-2]
        base = hf.get_submodule(self.prefix.rsplit(".", 1)[0])
        self.norm, self.embed = base.norm, base.embed_tokens
        self.nL = R.cfg.num_hidden_layers
        self.names = ["embed"]
        mods = []
        for L in range(self.nL):
            layer = hf.get_submodule(f"{self.prefix}.{L}")
            att = next(s for s in ("self_attn", "linear_attn", "attn", "mixer") if hasattr(layer, s))
            mods += [(f"a{L}", getattr(layer, att)), (f"m{L}", layer.mlp)]
            self.names += [f"a{L}", f"m{L}"]
        self.cap = {}
        for name, mod in [("embed", self.embed)] + mods:
            mod.register_forward_hook(self._hook(name))
        self.norm.register_forward_pre_hook(lambda m, a: self.cap.__setitem__("h_final", a[0][:, -1].detach().float()))
        d = R.cfg.hidden_size
        dev = next(self.norm.parameters()).device
        with torch.no_grad():
            g = self.norm(torch.ones(1, 1, d, device=dev, dtype=next(self.norm.parameters()).dtype))[0, 0].float()
        W = hf.get_output_embeddings().weight
        y, n = R.yn
        self.wg = ((W[y].float() - W[n].float()).to(dev) * g)
        self.eps = float(getattr(self.norm, "variance_epsilon", getattr(self.norm, "eps", 1e-6)))
        self.attn_ok = False
        try:
            hf.set_attn_implementation("eager")
            self.attn_ok = True
        except Exception as e:      # older transformers: no attention shares, the rest still works
            print(f"WARNING: cannot switch to eager attention ({e}); attention shares skipped", flush=True)

    def _hook(self, name):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            self.cap[name] = h[:, -1].detach().float()
        return hook

    @torch.no_grad()
    def run(self, text, spans):
        """spans: {'hist': (char_start, char_end)} in the RAW prompt text. Returns numbers for one prompt."""
        R = self.R
        chat = R.chat(text)
        off = chat.find(text[:200])          # raw prompt sits inside the chat template
        enc = R.tok(chat, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc.input_ids
        tok_span = {}
        for k, (a, b) in spans.items():
            a, b = a + off, b + off
            idx = [i for i, (s, e) in enumerate(enc.offset_mapping) if s < b and e > a]
            tok_span[k] = (idx[0], idx[-1] + 1) if idx else None
        if off < 0:
            tok_span = {}
        x = torch.tensor([ids], device=R.device)
        kw = {"output_attentions": True} if self.attn_ok else {}
        try:
            out = R.hf(input_ids=x, logits_to_keep=1, **kw)
        except TypeError:
            out = R.hf(input_ids=x, **kw)
        lg = out.logits[0, -1].float()
        y, n = R.yn
        m = (lg[y] - lg[n]).item()
        h = self.cap["h_final"][0]
        rms = torch.sqrt((h * h).mean() + self.eps)
        contrib = [float((self.cap[c][0] * self.wg).sum() / rms) for c in self.names]
        rec = {"m": m, "rms": float(rms), "c": contrib, "m_unnorm": float((h * self.wg).sum()),
               "decomp_err": abs(sum(contrib) - m)}
        if self.attn_ok and out.attentions is not None and tok_span.get("hist"):
            a0, a1 = tok_span["hist"]
            atts = list(out.attentions)
            # hybrid models may return attentions only for softmax-attention layers
            lt = getattr(R.cfg, "layer_types", None)
            layer_ids = list(range(self.nL)) if len(atts) == self.nL else \
                [i for i, t in enumerate(lt or []) if "full" in str(t)]
            sh = [None] * self.nL
            if len(layer_ids) == len(atts):
                for l, A in zip(layer_ids, atts):
                    if A is not None:
                        sh[l] = float(A[0, :, -1, a0:a1].float().sum(-1).mean())
            rec["att_hist"] = sh
        return rec


def hist_span(text, hist):
    i = text.find(hist)
    return {"hist": (i, i + len(hist))} if i >= 0 and hist else {}


# ---- analysis ---------------------------------------------------------------------------------
def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.cov(x, y, bias=True)[0, 1] / x.var()) if x.var() > 1e-9 else float("nan")


def analyse(recs, names, blank):
    cells = {}
    for t in sorted({t for r in recs for t in r["tests"]}):
        rs = [r["tests"][t] for r in recs if t in r["tests"] and "pairs" in r["tests"][t]]
        if len(rs) < 15:
            continue
        mB = np.array([r["B"]["m"] for r in rs])
        if mB.std() < 0.3:
            continue
        CB = np.array([r["B"]["c"] for r in rs])
        for w in rs[0]["pairs"]:
            for side in "PN":
                S = [r["pairs"][w][side] for r in rs]
                mS = np.array([s["m"] for s in S])
                CS = np.array([s["c"] for s in S])
                s_ = slope(mB, mS)
                a_ = float(mS.mean() - s_ * mB.mean())
                delta = [slope(mB, CS[:, j]) - slope(mB, CB[:, j]) for j in range(len(names))]
                uB = np.array([r["B"]["m_unnorm"] for r in rs]); uS = np.array([s["m_unnorm"] for s in S])
                c = {"test": t, "wording": w, "side": side, "n": len(rs), "s": s_, "pull": 1 - s_, "a": a_,
                     "L": a_ / (1 - s_) if s_ < 0.98 else None, "s_unnorm": slope(uB, uS),
                     "rms_ratio": float(np.mean([s["rms"] for s in S]) / np.mean([r["B"]["rms"] for r in rs])),
                     "delta": delta, "cum": list(np.cumsum(delta))}
                if "att_hist" in rs[0]["B"]:
                    ab = np.array([[np.nan if v is None else v for v in r["B"]["att_hist"]] for r in rs])
                    as_ = np.array([[np.nan if v is None else v for v in s["att_hist"]] for s in S])
                    c["att_B"], c["att_S"] = list(np.nanmean(ab, 0)), list(np.nanmean(as_, 0))
                bk = f"{t}|{w}|{side}"
                if bk in blank and f"{t}|B" in blank:
                    b0, f0 = blank[f"{t}|B"]["m"], blank[bk]["m"]
                    c["blank"] = {"B0": b0, "F0": f0, "line_pred": a_ + s_ * b0, "resid": f0 - (a_ + s_ * b0)}
                cells[bk] = c
    return cells


def report(cells, names, meta, out):
    f2 = lambda x: "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2f}"
    nL = (len(names) - 1) // 2
    L = [f"# Attractor probe: {meta['model']}", "",
         f"{meta['n_patients']} patients scored; mid prior {PB_RANGE}; {len(cells)} cell-sides. "
         f"Decomposition check: max |sum of contributions − real margin| = {meta['decomp_err_max']:.3g}.", "",
         "s = slope of final margin on no-result margin B (1 = Bayes, 0 = full attractor); pull = 1 − s. "
         "delta_c = component c's slope with result − without; sums to −pull. Negative = eraser.", ""]
    for side, lab in (("N", "negative results"), ("P", "positive results")):
        cs = [c for c in cells.values() if c["side"] == side]
        if not cs:
            continue
        D = np.array([c["delta"] for c in cs])
        med = np.median(D, 0)
        cum = np.median(np.array([c["cum"] for c in cs]), 0)
        L += [f"## {lab} ({len(cs)} cells)", "",
              f"- median pull {np.median([c['pull'] for c in cs]):.2f}; median pull before final norm "
              f"(1 − s_unnorm) {np.median([1 - c['s_unnorm'] for c in cs]):.2f}; "
              f"rms(result) / rms(no result) median {np.median([c['rms_ratio'] for c in cs]):.2f}", "",
              "Top erasers (median delta over cells; fraction of cells where it is negative):", "",
              "| component | median delta | cells < 0 | imaging median | lab median |", "|---|---|---|---|---|"]
        img = np.array([c["test"] in IMG for c in cs])
        for j in np.argsort(med)[:10]:
            L.append(f"| {names[j]} | {med[j]:+.3f} | {np.mean(D[:, j] < 0):.0%} | "
                     f"{np.median(D[img, j]) if img.any() else float('nan'):+.3f} | "
                     f"{np.median(D[~img, j]) if (~img).any() else float('nan'):+.3f} |")
        L += ["", "Biggest restorers (positive delta):", ""]
        L += [f"- {names[j]} {med[j]:+.3f}" for j in np.argsort(med)[::-1][:5]]
        L += ["", "Patient difference left by depth (1 + cumulative delta, median over cells; 1 = all kept):", "",
              "| after layer | " + " | ".join(str(l) for l in range(0, nL, max(1, nL // 16))) + " | end |",
              "|---|" + "---|" * (len(range(0, nL, max(1, nL // 16))) + 1),
              "| kept | " + " | ".join(f"{1 + cum[2 + 2 * l]:.2f}" for l in range(0, nL, max(1, nL // 16)))
              + f" | {1 + cum[-1]:.2f} |", ""]
        if "att_B" in cs[0]:
            aB = np.nanmedian(np.array([c["att_B"] for c in cs], float), 0)
            aS = np.nanmedian(np.array([c["att_S"] for c in cs], float), 0)
            ls = [l for l in range(nL) if not math.isnan(aB[l])]
            step = max(1, len(ls) // 16)
            L += ["Attention share on history at the answer position (median over cells):", "",
                  "| layer | " + " | ".join(str(l) for l in ls[::step]) + " |", "|---|" + "---|" * len(ls[::step]),
                  "| no result | " + " | ".join(f"{aB[l]:.2f}" for l in ls[::step]) + " |",
                  "| with result | " + " | ".join(f"{aS[l]:.2f}" for l in ls[::step]) + " |", ""]
        bl = [c for c in cs if "blank" in c]
        if bl:
            L += ["Blank history (consistency check): history-free point vs the cell's line and fixed point L", "",
                  "| cell | pull | B0 | F0 | line pred | resid | L |", "|---|---|---|---|---|---|---|"]
            for c in bl:
                b = c["blank"]
                L.append(f"| {c['test']}/{c['wording']} | {c['pull']:.2f} | {b['B0']:+.2f} | {b['F0']:+.2f} | "
                         f"{b['line_pred']:+.2f} | {b['resid']:+.2f} | {f2(c['L'])} |")
            L.append(f"\nmedian |resid| {np.median([abs(c['blank']['resid']) for c in bl]):.2f}\n")
    L += ["## Per cell", "", "| cell | side | n | pull | pull pre-norm | top eraser | its delta |", "|---|---|---|---|---|---|---|"]
    for k, c in cells.items():
        j = int(np.argmin(c["delta"]))
        L.append(f"| {c['test']}/{c['wording']} | {c['side']} | {c['n']} | {c['pull']:.2f} | {1 - c['s_unnorm']:.2f} | "
                 f"{names[j]} | {c['delta'][j]:+.2f} |")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps({"meta": meta, "names": names, "cells": cells}, default=float))
    print("\n".join(L[:80]))


# ---- run ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--question", default="orig")
    ap.add_argument("--max-per-test", type=int, default=80, help="mid-prior patients per test (cap)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from-saved", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rpath, bpath, npath = out / "records.jsonl", out / "blank.json", out / "names.json"

    if not args.from_saved:
        items = build_items(args.pool, 0, args.question)
        R = Runner(args.model, args.device, 1)
        P = Probe(R)
        npath.write_text(json.dumps({"names": P.names, "model": args.model, "attn": P.attn_ok}))
        t0 = time.time()
        # phase 1: no-result prompt for everyone -> mid-prior patients per test
        sel_path = out / "selection.json"
        if sel_path.exists():
            sel = json.loads(sel_path.read_text())
        else:
            sel = {}
            for t in items[0]["tests"]:
                mids = []
                for it in items:
                    m = R.margins([it["tests"][t]["B"]])[0]["m"]
                    if LO < m < HI:
                        mids.append(it["hadm_id"])
                sel[t] = mids[:args.max_per_test]
                print(f"{t}: {len(mids)} mid-prior, using {len(sel[t])} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            sel_path.write_text(json.dumps(sel))
        # phase 3 first (cheap): blank history
        if not bpath.exists():
            blank, it = {}, items[0]
            for t, T in it["tests"].items():
                rep = lambda s: s.replace(it["history"], BLANK)
                blank[f"{t}|B"] = P.run(rep(T["B"]), {})
                for w, pr in T["pairs"].items():
                    for side in "PN":
                        blank[f"{t}|{w}|{side}"] = P.run(rep(pr[side]), {})
            bpath.write_text(json.dumps(blank))
        # phase 2: per patient, all selected tests, B + every wording P/N with decomposition
        done = {json.loads(l)["hadm_id"] for l in open(rpath)} if rpath.exists() else set()
        todo = [it for it in items if it["hadm_id"] not in done and any(it["hadm_id"] in v for v in sel.values())]
        errs = []
        for k, it in enumerate(todo):
            rec = {"hadm_id": it["hadm_id"], "tests": {}}
            for t, T in it["tests"].items():
                if it["hadm_id"] not in sel.get(t, []):
                    continue
                sp = lambda s: hist_span(s, it["history"])
                row = {"B": P.run(T["B"], sp(T["B"])), "pairs": {}}
                for w, pr in T["pairs"].items():
                    row["pairs"][w] = {side: P.run(pr[side], sp(pr[side])) for side in "PN"}
                errs += [row["B"]["decomp_err"]]
                rec["tests"][t] = row
            with rpath.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            el = time.time() - t0
            print(f"{k + 1}/{len(todo)} patients, {el / 60:.1f} min, ~{el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min left; "
                  f"decomp err max {max(errs):.3g}", flush=True)
    names = json.loads(npath.read_text())["names"]
    recs = [json.loads(l) for l in open(rpath)]
    blank = json.loads(bpath.read_text()) if bpath.exists() else {}
    errs = [r["tests"][t]["B"]["decomp_err"] for r in recs for t in r["tests"]]
    meta = {"model": json.loads(npath.read_text())["model"], "n_patients": len(recs),
            "decomp_err_max": max(errs) if errs else float("nan")}
    report(analyse(recs, names, blank), names, meta, out)


if __name__ == "__main__":
    main()
