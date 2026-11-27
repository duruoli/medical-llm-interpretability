"""Attractor, step 2: which HEADS / NEURONS erase the patient difference, and does removing that erasure
lower the pull?  Follow-up to harness/attractor_probe.py (CROSSMODEL_batch1.md §13.5).

Part A  unit-level direct attribution (one forward per prompt, answer position only):
  head h of attention layer L : contribution = x_L[h-slice] . (W_o^T w)[h-slice] / rms(h_final)
                                (x_L = input of o_proj / out_proj, i.e. per-head outputs after any gate)
  neuron n of MLP layer L     : contribution = act_n * (W_down^T w)_n / rms(h_final)
  with w = (w_yes - w_no) * final-norm gain. Per cell: delta_u = slope(unit | result prompt) - slope(unit | no-result
  prompt), regressed on the patient's no-result margin B. Sum over all units of a component = that component's
  delta in attractor_probe. Negative = eraser.
Part B  causal test, "patient-mean patch": at the answer position, replace the chosen units' input values on the
  RESULT prompt by their mean over the cell's patients (same cell, same side). The unit keeps its average push
  but can no longer depend on the patient. If it is an eraser, the pull (1 - slope) should FALL and the mean move
  should stay roughly the same. Conditions per patched component: whole component, its top-k erasers, k random
  units (control). Plus all top-k sets together.
  RESULT (first run, biased 40-patient subset): did NOT lower the pull — with a result these units carry little patient
  information anyway, so replacing it by the mean removes nothing.
Part C  causal test, "add patient information back" (PLAN_attractor 3.1 step 2): on the RESULT prompt set the units to
      (cell mean of the unit on the result prompt) + (this patient's no-result value - mean no-result value over the
       cell's patients)
  i.e. the unit keeps the average push it has with a result but varies over patients as it did without one. If the
  pull is lost in these units (eraser), the pull should FALL while the mean move stays about the same. If the pull does
  not change, the patient information was not lost there (it no longer reaches the answer by another route).
  Conditions: addback_whole_<c>, addback_top_<c>, addback_rand_<c> (same number of random units), addback_top_all.
Also --track m24:14331,...: per-patient raw value of single units on the no-result and every result prompt
  (is the unit pushed into saturation?) -> track.json.
Every summary reports pull (1 - slope) AND spread = SD(F)/SD(B) (< 1 compression, ~1 reordering).

  python -m harness.attractor_units --model meta-llama/Llama-3.1-8B-Instruct --out /projects/p53430/medllm/attractor/llama_units \
      --selection /projects/p53430/medllm/attractor/llama/selection.json --units m24,m29,m21,m28,a24 --patch m24,m29
  python -m harness.attractor_units --model Qwen/Qwen3.5-9B --out /projects/p53430/medllm/attractor/qwen35_units \
      --selection /projects/p53430/medllm/attractor/qwen35/selection.json --units a27,m27,a23,m24,m26,m28 --patch a27,m27
  ... --from-saved      # report only

Batch 1, sdpa attention. Default 3 wordings per test (impression / findings / lexical / numeric first).
Output <out>/unitsA.npz (+ unitsA.json meta), patch_<cond>.json (one per condition; a rerun skips finished ones), REPORT.md.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from harness.attractor_probe import IMG
from harness.cancel_bins import pick_pairs
from harness.config import PB_RANGE
from harness.diagnose import Runner, build_items

LO, HI = math.log(PB_RANGE[0] / (1 - PB_RANGE[0])), math.log(PB_RANGE[1] / (1 - PB_RANGE[1]))


class Units:
    def __init__(self, R, comps):
        self.R, hf = R, R.hf
        cands = [n for n, _ in hf.named_modules() if re.fullmatch(r".*\.layers\.0", n) and "vision" not in n]
        prefix = cands[-1][:-2]
        base = hf.get_submodule(prefix.rsplit(".", 1)[0])
        norm = base.norm
        dev = next(norm.parameters()).device
        d = R.cfg.hidden_size
        with torch.no_grad():
            g = norm(torch.ones(1, 1, d, device=dev, dtype=next(norm.parameters()).dtype))[0, 0].float()
            W = hf.get_output_embeddings().weight
            y, n = R.yn
            wg = (W[y].float() - W[n].float()).to(dev) * g
        self.eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
        self.wg, self.prefix = wg, prefix
        self.cap, self.patch = {}, {}          # patch: comp -> (index tensor, value tensor) for the current prompt
        norm.register_forward_pre_hook(lambda m, a: self.cap.__setitem__("h_final", a[0][:, -1].detach().float()))
        self.mods, self.read, self.slices, self.n_units = {}, {}, {}, {}
        for c in comps:
            kind, L = c[0], int(c[1:])
            layer = hf.get_submodule(f"{prefix}.{L}")
            if kind == "m":
                proj = layer.mlp.down_proj
                self.n_units[c] = proj.in_features
            else:
                att = next(getattr(layer, s) for s in ("self_attn", "linear_attn", "attn", "mixer") if hasattr(layer, s))
                proj = next(getattr(att, s) for s in ("o_proj", "out_proj") if hasattr(att, s))
                hd = getattr(att, "head_v_dim", None) or getattr(att, "head_dim", None) or \
                    getattr(R.cfg, "head_dim", None) or R.cfg.hidden_size // R.cfg.num_attention_heads
                self.n_units[c] = proj.in_features // hd
                self.slices[c] = hd
            with torch.no_grad():
                self.read[c] = (proj.weight.float().t().to(dev) @ wg)       # (in_features,)
            proj.register_forward_pre_hook(self._hook(c))
            self.mods[c] = proj

    def _hook(self, c):
        def hook(mod, args):
            x = args[0]
            self.cap[c] = x[:, -1].detach().float().clone()
            if c in self.patch:
                idx, val = self.patch[c]
                x = x.clone()
                x[:, -1, idx] = val.to(x.dtype)
                return (x,) + tuple(args[1:])
            return None
        return hook

    def expand(self, c, units):
        """unit ids -> input-feature indices (heads cover head_dim features each)."""
        if c[0] == "m":
            return torch.as_tensor(units, dtype=torch.long)
        hd = self.slices[c]
        return torch.as_tensor([u * hd + j for u in units for j in range(hd)], dtype=torch.long)

    @torch.no_grad()
    def run(self, ids):
        R = self.R
        x = torch.tensor([ids], device=R.device)
        try:
            lg = R.hf(input_ids=x, logits_to_keep=1).logits[0, -1].float()
        except TypeError:
            lg = R.hf(input_ids=x).logits[0, -1].float()
        y, n = R.yn
        h = self.cap["h_final"][0]
        rms = torch.sqrt((h * h).mean() + self.eps)
        contrib = {}
        for c in self.mods:
            xin = self.cap[c][0]
            per = xin.to(self.read[c].device) * self.read[c] / rms
            contrib[c] = per if c[0] == "m" else per.view(self.n_units[c], self.slices[c]).sum(-1)
        return (lg[y] - lg[n]).item(), contrib, {c: self.cap[c][0] for c in self.mods}


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.cov(x, y, bias=True)[0, 1] / x.var()) if x.var() > 1e-9 else float("nan")


def vslope(x, Y):
    """slope of every column of Y on x."""
    x = np.asarray(x, float); Y = np.asarray(Y, float)
    xc = x - x.mean()
    v = (xc * xc).mean()
    return (xc[:, None] * (Y - Y.mean(0))).mean(0) / v if v > 1e-9 else np.full(Y.shape[1], np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", required=True, help="selection.json of the attractor_probe run (same patients)")
    ap.add_argument("--units", required=True, help="components to split, e.g. a27,m27 (a = heads, m = neurons)")
    ap.add_argument("--patch", required=True, help="components to patch in part B (subset of --units)")
    ap.add_argument("--k-heads", type=int, default=4)
    ap.add_argument("--k-neurons", type=int, default=32)
    ap.add_argument("--rank-side", default="N", choices=["N", "P", "both"], help="side used to rank top erasers")
    ap.add_argument("--pairs", type=int, default=3, help="wordings per test (0 = all)")
    ap.add_argument("--track", default="", help="single units to record per patient, e.g. m24:14331,m21:13111")
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from-saved", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    comps = args.units.split(",")
    pcomps = args.patch.split(",")
    track = [(x.split(":")[0], int(x.split(":")[1])) for x in args.track.split(",") if x]
    assert all(c in comps for c, _ in track), "--track components must be in --units"
    assert all(c in comps for c in pcomps), "--patch components must be in --units"
    sel = json.loads(Path(args.selection).read_text())
    apath = out / "unitsA.npz"

    items = build_items(args.pool, 0, "orig")
    by_id = {it["hadm_id"]: it for it in items}
    tests = [t for t in items[0]["tests"] if sel.get(t)]
    pairs = {t: (pick_pairs(items[0]["tests"][t]["pairs"], args.pairs) if args.pairs else list(items[0]["tests"][t]["pairs"]))
             for t in tests}
    cells = [(t, w, s) for t in tests for w in pairs[t] for s in "PN"]

    U = None
    if not args.from_saved:
        R = Runner(args.model, args.device, 1)
        U = Units(R, comps)
        tok = lambda s: R.tok(R.chat(s), add_special_tokens=False).input_ids

    # ---------------- part A
    if not apath.exists():
        t0 = time.time()
        mB, cB, rec = {t: [] for t in tests}, {t: {c: [] for c in comps} for t in tests}, {}
        xB = {t: {c: [] for c in pcomps} for t in tests}          # per-patient no-result input (for part C)
        tr = {f"{c}:{u}": {"B": {t: [] for t in tests}, "S": {"|".join(cell): [] for cell in cells}} for c, u in track}
        for cell in cells:
            rec[cell] = {"m": [], "c": {c: [] for c in comps}, "x": {c: None for c in pcomps}, "n": 0}
        for t in tests:
            for k, hid in enumerate(sel[t]):
                T = by_id[hid]["tests"][t]
                m, con, xin = U.run(tok(T["B"]))
                mB[t].append(m)
                for c in comps:
                    cB[t][c].append(con[c].cpu().numpy().astype(np.float32))
                for c in pcomps:
                    xB[t][c].append(xin[c].cpu().numpy().astype(np.float32))
                for c, u in track:
                    tr[f"{c}:{u}"]["B"][t].append(float(xin[c][u]))
                for w in pairs[t]:
                    for s in "PN":
                        m, con, xin = U.run(tok(T["pairs"][w][s]))
                        r = rec[(t, w, s)]
                        r["m"].append(m)
                        for c in comps:
                            r["c"][c].append(con[c].cpu().numpy().astype(np.float32))
                        for c in pcomps:
                            v = xin[c].cpu().numpy().astype(np.float64)
                            r["x"][c] = v if r["x"][c] is None else r["x"][c] + v
                        for c, u in track:
                            tr[f"{c}:{u}"]["S"]["|".join((t, w, s))].append(float(xin[c][u]))
                        r["n"] += 1
            print(f"A {t}: {len(sel[t])} patients, {(time.time() - t0) / 60:.1f} min", flush=True)
        if track:
            (out / "track.json").write_text(json.dumps({"tracks": tr, "mB": mB, "sel": {t: sel[t] for t in tests}}))
        save = {}
        for t in tests:
            save[f"mB|{t}"] = np.array(mB[t])
            for c in pcomps:
                save[f"xB|{c}|{t}"] = np.stack(xB[t][c])
        for c in comps:
            sB = {t: vslope(mB[t], cB[t][c]) for t in tests}
            save[f"slopeB|{c}"] = np.stack([sB[t] for t, w, s in cells])
            save[f"slopeS|{c}"] = np.stack([vslope(mB[t], rec[(t, w, s)]["c"][c]) for t, w, s in cells])
            save[f"meanS|{c}"] = np.stack([np.array(rec[cell]["c"][c]).mean(0) for cell in cells])
        for c in pcomps:
            save[f"xmean|{c}"] = np.stack([rec[cell]["x"][c] / rec[cell]["n"] for cell in cells]).astype(np.float32)
        save["mS"] = np.array([rec[cell]["m"] for cell in cells], dtype=object)
        np.savez(apath, **save)
        (out / "unitsA.json").write_text(json.dumps({"model": args.model, "cells": cells, "comps": comps,
                                                     "pcomps": pcomps, "minutes": (time.time() - t0) / 60}))
    A = np.load(apath, allow_pickle=True)
    side = np.array([s for _, _, s in cells])
    rank_mask = np.ones(len(cells), bool) if args.rank_side == "both" else side == args.rank_side
    delta = {c: A[f"slopeS|{c}"] - A[f"slopeB|{c}"] for c in comps}
    top = {}
    rng = np.random.default_rng(0)
    for c in pcomps:
        med = np.median(delta[c][rank_mask], 0)
        k = args.k_heads if c[0] == "a" else args.k_neurons
        top[c] = [int(u) for u in np.argsort(med)[:k]]
        top[f"rand_{c}"] = [int(u) for u in rng.choice(len(med), k, replace=False)]

    # ---------------- part B
    conds = {}
    for c in pcomps:
        conds[f"whole_{c}"] = {c: "all"}
        conds[f"top_{c}"] = {c: top[c]}
        conds[f"rand_{c}"] = {c: top[f"rand_{c}"]}
    conds["top_all"] = {c: top[c] for c in pcomps}
    # part C: add patient information back (name prefix addback_)
    for c in pcomps:
        conds[f"addback_whole_{c}"] = {c: "all"}
        conds[f"addback_top_{c}"] = {c: top[c]}
        conds[f"addback_rand_{c}"] = {c: top[f"rand_{c}"]}
    conds["addback_top_all"] = {c: top[c] for c in pcomps}
    for name, spec in conds.items():
        pth = out / f"patch_{name}.json"
        if pth.exists() or args.from_saved:
            continue
        addback = name.startswith("addback_")
        t0 = time.time()
        res = {}
        for i, (t, w, s) in enumerate(cells):
            idxs = {c: torch.arange(U.mods[c].in_features) if units == "all" else U.expand(c, units)
                    for c, units in spec.items()}
            mean_S = {c: torch.as_tensor(A[f"xmean|{c}"][i]).float() for c in spec}
            if addback:
                XB = {c: torch.as_tensor(A[f"xB|{c}|{t}"]).float() for c in spec}
                devB = {c: XB[c] - XB[c].mean(0) for c in spec}       # patient's no-result deviation
            ms = []
            for k, hid in enumerate(sel[t]):
                U.patch = {}
                for c in spec:
                    v = mean_S[c] + devB[c][k] if addback else mean_S[c]
                    U.patch[c] = (idxs[c].to(R.device), v[idxs[c]].to(R.device))
                ms.append(U.run(tok(by_id[hid]["tests"][t]["pairs"][w][s]))[0])
            U.patch = {}
            res["|".join((t, w, s))] = ms
        pth.write_text(json.dumps({"spec": {c: v for c, v in spec.items()}, "margins": res}))
        print(f"{'C' if addback else 'B'} {name}: {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---------------- report
    L = [f"# Attractor units: {args.model}", "",
         f"Cells: {len(cells)} (tests {len(tests)}, {args.pairs or 'all'} wordings each, P and N). Patients per test from "
         f"{args.selection}. delta < 0 = unit erases patient difference (sums over a component to its delta in "
         "attractor_probe).", "", "## A. Top erasing units (median delta over cells)", ""]
    for c in comps:
        L += [f"### {c} ({A[f'slopeS|{c}'].shape[1]} units)", "", "| unit | N median | N cells<0 | P median | P cells<0 | mean push N | mean push P |",
              "|---|---|---|---|---|---|---|"]
        dN, dP = delta[c][side == "N"], delta[c][side == "P"]
        mN, mP = A[f"meanS|{c}"][side == "N"], A[f"meanS|{c}"][side == "P"]
        tot = np.median(delta[c].sum(1)[side == "N"]), np.median(delta[c].sum(1)[side == "P"])
        order = np.argsort(np.median(delta[c][rank_mask], 0))[:12]
        for u in order:
            L.append(f"| {u} | {np.median(dN[:, u]):+.3f} | {np.mean(dN[:, u] < 0):.0%} | {np.median(dP[:, u]):+.3f} | "
                     f"{np.mean(dP[:, u] < 0):.0%} | {np.median(mN[:, u]):+.2f} | {np.median(mP[:, u]):+.2f} |")
        k = args.k_heads if c[0] == "a" else args.k_neurons
        topk = np.argsort(np.median(delta[c][rank_mask], 0))[:k]
        L += ["", f"component total delta (median): N {tot[0]:+.3f}, P {tot[1]:+.3f}; top-{k} share: "
                  f"N {np.median(delta[c][side == 'N'][:, topk].sum(1)):+.3f}, P {np.median(delta[c][side == 'P'][:, topk].sum(1)):+.3f}", ""]
    L += ["## B/C. Patches at the answer position (causal)", "",
          "pull = 1 − slope(final margin on B); spread = SD(final)/SD(B); move = mean(P − B) or mean(B − N). "
          "B (no prefix): patient-mean patch. C (addback_*): patient's no-result deviation added back on top of the "
          "result-prompt mean — if the pull is lost in these units, pull falls and move stays about the same. "
          "rand = same number of random units of that component (control).", "",
          "| condition | N pull | N spread | N move | P pull | P spread | P move | N pull imaging | N pull lab |",
          "|---|---|---|---|---|---|---|---|---|"]

    def summarise(margs):
        rows = {"N": [], "P": []}
        for i, (t, w, s) in enumerate(cells):
            mb = A[f"mB|{t}"]
            ms = np.array(margs[i])
            mv = (ms - mb).mean() if s == "P" else (mb - ms).mean()
            rows[s].append((1 - slope(mb, ms), mv, t in IMG, ms.std() / mb.std()))
        f = lambda s, j: np.median([r[j] for r in rows[s]])
        fi = lambda s, img: np.median([r[0] for r in rows[s] if r[2] == img]) if any(r[2] == img for r in rows[s]) else float("nan")
        return f("N", 0), f("N", 3), f("N", 1), f("P", 0), f("P", 3), f("P", 1), fi("N", True), fi("N", False)

    allres = {"base": summarise([list(A["mS"][i]) for i in range(len(cells))])}
    for name in conds:
        pth = out / f"patch_{name}.json"
        if pth.exists():
            mg = json.loads(pth.read_text())["margins"]
            allres[name] = summarise([mg["|".join(cell)] for cell in cells])
    for name, r in allres.items():
        L.append("| " + name + " | " + " | ".join(f"{x:.2f}" for x in r) + " |")
    L += ["", "Top units used: " + "; ".join(f"{c}: {top[c]}" for c in pcomps), ""]
    tp = out / "track.json"
    if tp.exists():
        T = json.loads(tp.read_text())
        L += ["## D. Tracked units (raw value into the output projection)", "",
              "slope = unit value on B across patients; p5/p50/p95 over patients (median over cells).", "",
              "| unit | prompt | p5 | p50 | p95 | slope on B |", "|---|---|---|---|---|---|"]
        for key, d in T["tracks"].items():
            rowsB = [(np.percentile(v, [5, 50, 95]), slope(T["mB"][t], v)) for t, v in d["B"].items() if v]
            for lab, sd in (("no result", None), ("result N", "N"), ("result P", "P")):
                rr = rowsB if sd is None else \
                    [(np.percentile(v, [5, 50, 95]), slope(T["mB"][c.split("|")[0]], v))
                     for c, v in d["S"].items() if v and c.endswith("|" + sd)]
                q = np.median(np.array([r[0] for r in rr]), 0); sl = np.median([r[1] for r in rr])
                L.append(f"| {key} | {lab} | {q[0]:+.2f} | {q[1]:+.2f} | {q[2]:+.2f} | {sl:+.3f} |")
        L.append("")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps({"top": top, "summary": allres}, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
