"""Attractor 3.1 follow-ups (PLAN_attractor.md "下一步（3.1 后续）"), one pass per model.

Background (results/attractor/v2): adding the patient's no-result deviation back into the top erasing units lowers
the pull (Llama m24+m29: 0.51/0.63 -> 0.31/0.37; Qwen a27 heads: 0.78 -> 0.66), but only ~1/3 of what their
attribution (delta) promised.  Offline (results/attractor/v2/saturation_offline.txt): the key neurons sit far from 0
with a result, i.e. on the linear part of silu, so their own nonlinearity cannot be what flattens them.

(a) multi-layer add-back: groups of components patched together (addback_top_<G>, addback_whole_<G>,
    addback_rand_<G>), plus single components not tested before.  Every patched run also records the direct
    contribution of EVERY attention / MLP block ("monitor"), so we can see whether blocks AFTER the patched ones erase
    more when the difference is put back (re-erasure) or not (missing parallel route).
(b) held-out half: per test the patients are split in two halves (fixed seed). addback_fittop_<G> uses top units
    ranked on the fit half only; its pull is read on the other half (column "held" in the report).
(c) gate / up split for tracked neurons: value = silu(gate) * up.  Records gate pre-activation, up, and the RMS of
    the residual stream entering that MLP (its input norm).  Which of them loses its patient slope with a result?
      gate and up both keep their slope            -> flattening inside the neuron (would contradict the offline check)
      gate / up lose slope, residual RMS unchanged -> the patient signal is already gone (or cancelled) upstream
      residual RMS grows with the result            -> norm-driven shrink (passive)

  python -m harness.attractor_followup --model meta-llama/Llama-3.1-8B-Instruct --out <dir> \
      --selection <v2>/llama/selection.json --groups "L=m21,m24,m28,m29" --singles m21,m28 \
      --track m24:14331,m28:10558,m29:2039,m29:12580,m21:13111
Group spec: name=c1,c2[@N|@P|@both] (side used to rank its top units; default both).
Output: A.npz / A.json, cond_<name>.json, REPORT.md, report.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from harness.attractor_probe import IMG
from harness.attractor_units import Units, slope, vslope
from harness.cancel_bins import pick_pairs
from harness.diagnose import Runner, build_items


class Monitor:
    """Direct contribution of every attention / MLP block at the answer position (same linearisation as probe)."""

    def __init__(self, R, U):
        hf = R.hf
        self.U, self.cap, self.names = U, {}, []
        for L in range(R.cfg.num_hidden_layers):
            layer = hf.get_submodule(f"{U.prefix}.{L}")
            att = next(getattr(layer, s) for s in ("self_attn", "linear_attn", "attn", "mixer") if hasattr(layer, s))
            for nm, mod in ((f"a{L}", att), (f"m{L}", layer.mlp)):
                self.names.append(nm)
                mod.register_forward_hook(self._hook(nm))

    def _hook(self, nm):
        def hook(mod, args, out):
            x = out[0] if isinstance(out, tuple) else out
            self.cap[nm] = (x[0, -1].float() * self.U.wg).sum()
        return hook

    def vec(self):
        h = self.U.cap["h_final"][0]
        rms = torch.sqrt((h * h).mean() + self.U.eps)
        return (torch.stack([self.cap[n] for n in self.names]) / rms).cpu().numpy()


class GateUp:
    def __init__(self, R, U, track):
        self.cap, self.keys = {}, []
        for c, u in track:
            layer = R.hf.get_submodule(f"{U.prefix}.{int(c[1:])}")
            k = f"{c}:{u}"
            self.keys.append(k)
            layer.mlp.gate_proj.register_forward_hook(self._out(k + "|g", u))
            layer.mlp.up_proj.register_forward_hook(self._out(k + "|u", u))
            layer.post_attention_layernorm.register_forward_pre_hook(self._rms(k + "|r"))

    def _out(self, key, u):
        def hook(mod, args, out):
            self.cap[key] = out[0, -1, u].float()
        return hook

    def _rms(self, key):
        def hook(mod, args):
            x = args[0][0, -1].float()
            self.cap[key] = torch.sqrt((x * x).mean())
        return hook

    def read(self):
        return {k: float(v) for k, v in self.cap.items()}


def parse_groups(s):
    out = {}
    for part in [p for p in s.split(";") if p]:
        name, rest = part.split("=")
        comps, side = (rest.split("@") + ["both"])[:2]
        out[name] = (comps.split(","), side)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--groups", required=True, help='e.g. "L=m21,m24,m28,m29" or "N=a23,a27@N;P=m21,m28@P"')
    ap.add_argument("--singles", default="", help="components also added back alone (top units)")
    ap.add_argument("--track", default="", help="neurons for the gate/up split, e.g. m24:14331")
    ap.add_argument("--k-heads", type=int, default=4)
    ap.add_argument("--k-neurons", type=int, default=32)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from-saved", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="patients per test (smoke test only)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = parse_groups(args.groups)
    singles = [c for c in args.singles.split(",") if c]
    comps = sorted({c for cs, _ in groups.values() for c in cs} | set(singles), key=lambda c: (int(c[1:]), c[0]))
    track = [(x.split(":")[0], int(x.split(":")[1])) for x in args.track.split(",") if x]
    sel = json.loads(Path(args.selection).read_text())
    if args.limit:                                   # smoke test only
        sel = {t: v[:args.limit] for t, v in sel.items()}

    items = build_items(args.pool, 0, "orig")
    by_id = {it["hadm_id"]: it for it in items}
    tests = [t for t in items[0]["tests"] if sel.get(t)]
    pairs = {t: (pick_pairs(items[0]["tests"][t]["pairs"], args.pairs) if args.pairs else list(items[0]["tests"][t]["pairs"]))
             for t in tests}
    cells = [(t, w, s) for t in tests for w in pairs[t] for s in "PN"]
    ckey = lambda cell: "|".join(cell)
    rng = np.random.default_rng(args.seed)
    fit = {}
    for t in tests:
        perm = rng.permutation(len(sel[t]))
        m = np.zeros(len(sel[t]), bool)
        m[perm[: len(perm) // 2]] = True
        fit[t] = m                                   # True = fit half (ranking), False = held-out half

    apath = out / "A.npz"
    if not args.from_saved:
        R = Runner(args.model, args.device, 1)
        U = Units(R, comps)
        M = Monitor(R, U)
        G = GateUp(R, U, track)
        tok = lambda s: R.tok(R.chat(s), add_special_tokens=False).input_ids

    # ---------------- part A: unpatched pass
    if not apath.exists():
        t0 = time.time()
        mB, cB, xB, monB = ({t: [] for t in tests}, {t: {c: [] for c in comps} for t in tests},
                            {t: {c: [] for c in comps} for t in tests}, {t: [] for t in tests})
        rec = {cell: {"m": [], "c": {c: [] for c in comps}, "x": {c: 0.0 for c in comps}, "mon": []} for cell in cells}
        gu = {"B": {t: [] for t in tests}, "S": {ckey(cell): [] for cell in cells}}
        for t in tests:
            for hid in sel[t]:
                T = by_id[hid]["tests"][t]
                m, con, xin = U.run(tok(T["B"]))
                mB[t].append(m); monB[t].append(M.vec()); gu["B"][t].append(G.read())
                for c in comps:
                    cB[t][c].append(con[c].cpu().numpy().astype(np.float32))
                    xB[t][c].append(xin[c].cpu().numpy().astype(np.float32))
                for w in pairs[t]:
                    for s in "PN":
                        m, con, xin = U.run(tok(T["pairs"][w][s]))
                        r = rec[(t, w, s)]
                        r["m"].append(m); r["mon"].append(M.vec()); gu["S"][ckey((t, w, s))].append(G.read())
                        for c in comps:
                            r["c"][c].append(con[c].cpu().numpy().astype(np.float32))
                            r["x"][c] = r["x"][c] + xin[c].cpu().numpy().astype(np.float64)
            print(f"A {t}: {len(sel[t])} patients, {(time.time() - t0) / 60:.1f} min", flush=True)
        save = {}
        for t in tests:
            save[f"mB|{t}"] = np.array(mB[t]); save[f"monB|{t}"] = np.stack(monB[t])
            for c in comps:
                save[f"xB|{c}|{t}"] = np.stack(xB[t][c])
        for i, cell in enumerate(cells):
            t = cell[0]
            r = rec[cell]
            save[f"mS|{i}"] = np.array(r["m"]); save[f"monS|{i}"] = np.stack(r["mon"])
            for c in comps:
                save[f"xmean|{c}|{i}"] = (r["x"][c] / len(r["m"])).astype(np.float32)
                S = np.stack(r["c"][c]); Bc = np.stack(cB[t][c]); mb = np.array(mB[t])
                save[f"delta|{c}|{i}"] = vslope(mb, S) - vslope(mb, Bc)
                save[f"deltafit|{c}|{i}"] = vslope(mb[fit[t]], S[fit[t]]) - vslope(mb[fit[t]], Bc[fit[t]])
        np.savez(apath, **save)
        (out / "A.json").write_text(json.dumps({"model": args.model, "cells": cells, "comps": comps, "names": M.names,
                                                "fit": {t: fit[t].tolist() for t in tests}, "gateup": gu,
                                                "minutes": (time.time() - t0) / 60}))
    A = np.load(apath)
    meta = json.loads((out / "A.json").read_text())
    names = meta["names"]
    side = np.array([s for _, _, s in cells])

    def top_units(c, which, side_sel):
        mask = np.ones(len(cells), bool) if side_sel == "both" else side == side_sel
        D = np.stack([A[f"{which}|{c}|{i}"] for i in range(len(cells))])[mask]
        k = args.k_heads if c[0] == "a" else args.k_neurons
        return [int(u) for u in np.argsort(np.median(D, 0))[:k]]

    n_units = {c: A[f"delta|{c}|0"].shape[0] for c in comps}
    conds = {}
    for c in singles:
        conds[f"top_{c}"] = {c: top_units(c, "delta", "both")}
    for g, (cs, sd) in groups.items():
        conds[f"top_{g}"] = {c: top_units(c, "delta", sd) for c in cs}
        conds[f"whole_{g}"] = {c: "all" for c in cs}
        conds[f"rand_{g}"] = {c: [int(u) for u in rng.choice(n_units[c], args.k_heads if c[0] == "a" else args.k_neurons,
                                                          replace=False)] for c in cs}
        conds[f"fittop_{g}"] = {c: top_units(c, "deltafit", sd) for c in cs}

    # ---------------- part B: add-back runs (patient's no-result deviation on top of the result-prompt cell mean)
    for name, spec in conds.items():
        pth = out / f"cond_{name}.json"
        if pth.exists() or args.from_saved:
            continue
        t0 = time.time()
        res = {}
        for i, (t, w, s) in enumerate(cells):
            idx = {c: torch.arange(U.mods[c].in_features) if u == "all" else U.expand(c, u) for c, u in spec.items()}
            val = {}
            for c in spec:
                XB = torch.as_tensor(A[f"xB|{c}|{t}"]).float()
                val[c] = torch.as_tensor(A[f"xmean|{c}|{i}"]).float() + (XB - XB.mean(0))
            ms, mons = [], []
            for k, hid in enumerate(sel[t]):
                U.patch = {c: (idx[c].to(R.device), val[c][k][idx[c]].to(R.device)) for c in spec}
                ms.append(U.run(tok(by_id[hid]["tests"][t]["pairs"][w][s]))[0])
                mons.append(M.vec().tolist())
            U.patch = {}
            res[ckey((t, w, s))] = {"m": ms, "mon": mons}
        pth.write_text(json.dumps({"spec": spec, "results": res}))
        print(f"B {name}: {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---------------- report
    def pull_of(mb, ms):
        return 1 - slope(mb, ms)

    def summarise(get):
        """get(i) -> (margins, monitor matrix) for cell i. Returns per-side medians."""
        rows = {"N": [], "P": []}
        for i, (t, w, s) in enumerate(cells):
            mb = A[f"mB|{t}"]; ms, mon = get(i)
            ms = np.asarray(ms); ho = ~np.array(meta["fit"][t])
            mv = (ms - mb).mean() if s == "P" else (mb - ms).mean()
            dmon = vslope(mb, mon) - vslope(mb, A[f"monB|{t}"])
            rows[s].append(dict(pull=pull_of(mb, ms), held=pull_of(mb[ho], ms[ho]), spread=ms.std() / mb.std(),
                                move=mv, dmon=dmon))
        return rows

    base = summarise(lambda i: (A[f"mS|{i}"], A[f"monS|{i}"]))
    allres = {"base": base}
    for name in conds:
        pth = out / f"cond_{name}.json"
        if pth.exists():
            R_ = json.loads(pth.read_text())["results"]
            allres[name] = summarise(lambda i, R_=R_: (R_[ckey(cells[i])]["m"], np.array(R_[ckey(cells[i])]["mon"])))
    med = lambda rows, k: float(np.median([r[k] for r in rows]))

    L = [f"# Attractor follow-up: {meta['model']}", "",
         f"{len(cells)} cells; patients from {args.selection}; groups {args.groups}; singles {args.singles or '-'}.", "",
         "## (a)(b) Add-back", "",
         "pull = 1 − slope on B (all patients); held = same on the held-out half only (for fittop_* the units were "
         "ranked on the other half); spread = SD(F)/SD(B); move = mean push. rand = random units, same number.", "",
         "| condition | N pull | N held | N spread | N move | P pull | P held | P spread | P move |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, rows in allres.items():
        L.append(f"| {name} | " + " | ".join(f"{med(rows[s], k):.2f}" for s in "NP" for k in ("pull", "held", "spread", "move")) + " |")

    L += ["", "## (a) Where does the erasure go when a difference is added back?", "",
          "Sum of block deltas (slope with result − without, on B), median over cells, split at the LAST patched block: "
          "up to and including it / after it. Base row = same split, unpatched. If 'after' becomes more negative "
          "under add-back, later blocks re-erase what was put back.", "",
          "| condition | side | up to patched: base → cond | after patched: base → cond |", "|---|---|---|---|"]
    for name, spec in conds.items():
        if name not in allres:
            continue
        last = max(names.index(c) for c in spec)
        for s in "NP":
            b = np.array([r["dmon"] for r in base[s]]); c_ = np.array([r["dmon"] for r in allres[name][s]])
            L.append(f"| {name} | {s} | {np.median(b[:, :last + 1].sum(1)):+.2f} → {np.median(c_[:, :last + 1].sum(1)):+.2f} | "
                     f"{np.median(b[:, last + 1:].sum(1)):+.2f} → {np.median(c_[:, last + 1:].sum(1)):+.2f} |")
    for name in [n for n in conds if n.startswith("top_") and n[4:] in groups] + ["base"]:
        if name not in allres:
            continue
        for s in "NP":
            D = np.median(np.array([r["dmon"] for r in allres[name][s]]), 0)
            order = np.argsort(D)[:8]
            L.append(f"\n{name} {s} biggest erasing blocks: " + ", ".join(f"{names[j]} {D[j]:+.2f}" for j in order))

    gu = meta["gateup"]
    if track:
        L += ["", "## (c) Gate / up split of tracked neurons", "",
              "value = silu(gate)·up. Slopes on B across patients (median over cells); 'gate<0' = share of patients "
              "with negative gate pre-activation; rms = residual RMS entering the MLP.", "",
              "| neuron | prompt | gate p50 | gate<0 | slope gate | up p50 | slope up | slope value | rms p50 | slope rms |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        silu = lambda g: g / (1 + np.exp(-g))
        for c, u in track:
            k = f"{c}:{u}"
            for lab, sd in (("no result", None), ("result N", "N"), ("result P", "P")):
                rows = []
                if sd is None:
                    src = [(A[f"mB|{t}"], gu["B"][t]) for t in tests]
                else:
                    src = [(A[f"mB|{cl[0]}"], gu["S"][ckey(cl)]) for cl in cells if cl[2] == sd]
                for mb, recs in src:
                    g = np.array([r[k + "|g"] for r in recs]); up = np.array([r[k + "|u"] for r in recs])
                    rr = np.array([r[k + "|r"] for r in recs]); v = silu(g) * up
                    rows.append([np.median(g), np.mean(g < 0), slope(mb, g), np.median(up), slope(mb, up), slope(mb, v),
                                 np.median(rr), slope(mb, rr)])
                q = np.median(np.array(rows), 0)
                L.append(f"| {k} | {lab} | {q[0]:+.2f} | {q[1]:.0%} | {q[2]:+.3f} | {q[3]:+.2f} | {q[4]:+.3f} | "
                         f"{q[5]:+.3f} | {q[6]:.2f} | {q[7]:+.3f} |")
    L += ["", "Units used: " + "; ".join(f"{n}: " + ", ".join(f"{c} {v if v == 'all' else len(v)}" for c, v in sp.items())
                                         for n, sp in conds.items())]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps({"conds": conds, "summary": {
        n: {s: {k: med(r[s], k) for k in ("pull", "held", "spread", "move")} for s in "NP"} for n, r in allres.items()}}))
    print("\n".join(L))


if __name__ == "__main__":
    main()
