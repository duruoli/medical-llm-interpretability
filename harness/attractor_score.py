"""Attractor scoring (SPEC_attractor_harness.md). numpy only; shared by server.py and final_eval_attractor.py.

Per cell = (test, wording, side), on the mid-prior cases of that test (fixed once on the ORIGINAL model):
    B = no-result margin, F = margin with the result, push d = F - B
    d = move - p * (B - mean B) + r          (least squares; r = what the line does not explain)
  move   mean push                    -> CONSTRAINT: must not change
  p      pull (= 1 - slope of F on B) -> TARGET: median |p| over target cells, lower is better
  r      patient-specific remainder   -> report only (SD of r)
Report-only extras: spread = SD(F)/SD(B), AUC of F for the test's disease, negative results that RAISE belief,
ln(up/down) by start tercile (does the push ratio still depend on the start?).
"""
import math
import re

import numpy as np

from harness.config import ATTRACTOR, ATTRACTOR_PROFILES, DOWN_FLOOR, GENERAL_YESNO, TEST_DISEASE

A = ATTRACTOR


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    for v in np.unique(allv):                     # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cell_terms(B, F):
    B, F = np.asarray(B, float), np.asarray(F, float)
    d = F - B
    k, c = np.polyfit(B, d, 1)
    r = d - (c + k * B)
    return {"n": len(B), "p": float(-k), "move": float(d.mean()), "spread": float(F.std() / B.std()),
            "r_sd": float(r.std())}


def all_cells(m, items, mid):
    """m = run_items output. Returns {test/wording/side: terms (+auc, neg_raises)}."""
    out = {}
    for test in items[0]["tests"]:
        idx = mid.get(test, [])
        if len(idx) < A["n_min"]:
            continue
        B = np.array([m[i][test]["B"] for i in idx])
        if B.std() < 1e-6:
            continue
        dz = TEST_DISEASE.get(test)
        lab = np.array([items[i]["true_disease"] == dz for i in idx]) if dz else None
        for pair in items[0]["tests"][test]["pairs"]:
            for side in "PN":
                F = np.array([m[i][test][pair][side] for i in idx])
                c = cell_terms(B, F)
                c["auc"] = auc(F, lab) if lab is not None else float("nan")
                if side == "N":
                    c["neg_raises"] = int((F > B).sum())
                out[f"{test}/{pair}/{side}"] = c
    return out


def tercile_ratio(m, items, mid, ref):
    """Per test: ln(mean up / mean down) in each tercile of the ORIGINAL start (median over wordings)."""
    res = {}
    for test in items[0]["tests"]:
        idx = mid.get(test, [])
        if len(idx) < 3 * A["n_min"] // 2:
            continue
        b0 = np.array([ref[i][test]["B"] for i in idx])
        cut = np.quantile(b0, [1 / 3, 2 / 3])
        grp = np.digitize(b0, cut)
        rows = []
        for pair in items[0]["tests"][test]["pairs"]:
            B = np.array([m[i][test]["B"] for i in idx])
            up = np.array([m[i][test][pair]["P"] for i in idx]) - B
            dn = B - np.array([m[i][test][pair]["N"] for i in idx])
            rows.append([math.log(max(up[grp == g].mean(), DOWN_FLOOR) / max(dn[grp == g].mean(), DOWN_FLOOR))
                         for g in range(3)])
        res[test] = list(np.median(np.array(rows), 0))
    return res


def excluded(profile):
    """Function cell-key -> True if the cell is reported but not targeted."""
    prof = ATTRACTOR_PROFILES[profile]
    rx = re.compile(prof["exclude_wordings"]) if prof.get("exclude_wordings") else None
    tests = set(prof.get("exclude_tests") or [])
    return lambda k: bool(rx) and k.split("/")[0] in tests and bool(rx.search(k.split("/")[1]))


def attractor_report(cur, ref, g_cur, g_ref, items, mid, profile):
    prof = ATTRACTOR_PROFILES[profile]
    is_excl = excluded(profile)
    now, org = all_cells(cur, items, mid), all_cells(ref, items, mid)
    keys = [k for k in org if k in now]
    side = lambda k: k.rsplit("/", 1)[1]
    tgt = [k for k in keys if side(k) in prof["target_sides"] and not is_excl(k)]
    other = [k for k in keys if side(k) not in prof["target_sides"] and not is_excl(k)]
    excl_note = sorted({k.rsplit("/", 1)[0] for k in keys if is_excl(k)})
    med = lambda ks, cs, f=lambda c: abs(c["p"]): float(np.median([f(cs[k]) for k in ks])) if ks else float("nan")

    # ---- target
    abs_pull, abs_pull_orig = med(tgt, now), med(tgt, org)
    by_test = {}
    for k in tgt:
        by_test.setdefault(k.split("/")[0], []).append(k)
    test_rows = {t: (med(ks, org), med(ks, now)) for t, ks in by_test.items()}
    # ---- constraints
    move_fail = []
    for k in keys:
        m0, m1 = org[k]["move"], now[k]["move"]
        if abs(m1 - m0) > A["move_tol"] * max(abs(m0), A["move_floor"]):
            move_fail.append(k)
    floor_fail = [k for k in tgt if now[k]["p"] < A["p_floor"]]
    worse = [t for t, (a, b) in test_rows.items() if b > a + A["test_worse"]]
    other_rows = {}
    for s in "PN":
        ks = [k for k in other if side(k) == s]
        if ks:
            other_rows[s] = (med(ks, org, lambda c: c["p"]), med(ks, now, lambda c: c["p"]))
    other_fail = [s for s, (a, b) in other_rows.items() if abs(b - a) >= A["other_side_dp"]]
    drift = float(np.mean([abs(cur[i][t]["B"] - ref[i][t]["B"]) for i in range(len(items)) for t in items[i]["tests"]]))
    mass = float(np.mean([cur[i][t]["mass_B"] for i in range(len(items)) for t in items[i]["tests"]]))
    mass0 = float(np.mean([ref[i][t]["mass_B"] for i in range(len(items)) for t in items[i]["tests"]]))
    gold = [a for _, a in GENERAL_YESNO]
    acc = lambda ms: float(np.mean([(x > 0) == (a == "Yes") for x, a in zip(ms, gold)]))
    nll, nll0 = float(np.nanmean(g_cur["nll"])), float(np.nanmean(g_ref["nll"]))
    guards = {
        "baseline_drift": (drift, drift < A["drift_max"]),
        "general_acc": ((acc(g_cur["general_m"]), acc(g_ref["general_m"])), acc(g_cur["general_m"]) >= acc(g_ref["general_m"])),
        "history_nll": ((nll, nll0), abs(nll - nll0) < A["nll_max"] if not math.isnan(nll) else True),
        "yesno_mass": ((mass, mass0), mass >= mass0 - A["mass_drop"]),
    }
    constraints_ok = (not move_fail and not floor_fail and not worse and not other_fail
                      and all(ok for _, ok in guards.values()))
    target_ok = abs_pull <= A["success_ratio"] * abs_pull_orig
    # ---- report-only
    rep_items = {}
    for s in "PN":
        ks = [k for k in keys if side(k) == s]
        if ks:
            rep_items[s] = {f: (med(ks, org, lambda c, f=f: c[f]), med(ks, now, lambda c, f=f: c[f]))
                            for f in ("p", "move", "spread", "r_sd")}
            ka = [k for k in ks if not math.isnan(org[k]["auc"])]
            rep_items[s]["auc"] = (med(ka, org, lambda c: c["auc"]), med(ka, now, lambda c: c["auc"])) if ka else None
    neg0 = sum(org[k].get("neg_raises", 0) for k in keys)
    neg1 = sum(now[k].get("neg_raises", 0) for k in keys)
    terc0, terc1 = tercile_ratio(ref, items, mid, ref), tercile_ratio(cur, items, mid, ref)

    head = {"abs_pull": abs_pull, "abs_pull_orig": abs_pull_orig,
            "abs_pull_mean": float(np.mean([abs(now[k]["p"]) for k in tgt])) if tgt else float("nan"),
            "abs_pull_mean_orig": float(np.mean([abs(org[k]["p"]) for k in tgt])) if tgt else float("nan"),
            "target_ok": bool(target_ok), "constraints_ok": bool(constraints_ok),
            "success": bool(target_ok and constraints_ok), "n_target_cells": len(tgt),
            "move_fail": move_fail, "floor_fail": floor_fail, "tests_worse": worse, "other_side_fail": other_fail,
            "guards": guards, "neg_raises": (neg0, neg1)}
    f = lambda x: "  –  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2f}"
    ok = lambda b: "ok" if b else "FAIL"
    L = ["TARGET (pull p; median |p| over target cells: sides "
         f"{prof['target_sides']}{'; not targeted, reported only: ' + ', '.join(excl_note) if excl_note else ''})",
         f"  abs_pull  {abs_pull_orig:.3f} -> {abs_pull:.3f}   (success line <= {A['success_ratio'] * abs_pull_orig:.3f}: "
         f"{ok(target_ok)}; mean {head['abs_pull_mean_orig']:.3f} -> {head['abs_pull_mean']:.3f}; {len(tgt)} cells)",
         "  by test:  " + "   ".join(f"{t} {a:.2f}->{b:.2f}" for t, (a, b) in test_rows.items()),
         "CONSTRAINTS (all must hold)",
         f"  move unchanged per cell (±{A['move_tol']:.0%}, floor {A['move_floor']}): "
         f"{ok(not move_fail)}  {len(move_fail)} of {len(keys)} cells fail" + (f": {move_fail[:6]}" if move_fail else ""),
         f"  no target test worse by > {A['test_worse']}: {ok(not worse)} {worse if worse else ''}",
         f"  no target cell p < {A['p_floor']}: {ok(not floor_fail)} {floor_fail[:6] if floor_fail else ''}"]
    for s, (a, b) in other_rows.items():
        L.append(f"  other side {s}: median p {a:+.2f} -> {b:+.2f} (|change| < {A['other_side_dp']}): "
                 f"{ok(s not in other_fail)}")
    gd = guards
    L += [f"  baseline_drift {gd['baseline_drift'][0]:.3f} (< {A['drift_max']}): {ok(gd['baseline_drift'][1])}",
          f"  general_acc    {gd['general_acc'][0][1]:.3f} -> {gd['general_acc'][0][0]:.3f} (not lower): {ok(gd['general_acc'][1])}",
          f"  history_nll    {gd['history_nll'][0][1]:.4f} -> {gd['history_nll'][0][0]:.4f} (|change| < {A['nll_max']}): "
          f"{ok(gd['history_nll'][1])}",
          f"  yesno_mass     {gd['yesno_mass'][0][1]:.3f} -> {gd['yesno_mass'][0][0]:.3f}: {ok(gd['yesno_mass'][1])}",
          f"SUCCESS: {ok(head['success'])}",
          "REPORT ONLY (median over cells, original -> now)"]
    for s, d in rep_items.items():
        L.append(f"  side {s}: p {f(d['p'][0])}->{f(d['p'][1])}  move {f(d['move'][0])}->{f(d['move'][1])}  "
                 f"spread SD(F)/SD(B) {d['spread'][0]:.2f}->{d['spread'][1]:.2f}  r_sd {d['r_sd'][0]:.2f}->{d['r_sd'][1]:.2f}"
                 + (f"  AUC {d['auc'][0]:.3f}->{d['auc'][1]:.3f}" if d["auc"] else ""))
    L.append(f"  negative results that RAISED belief: {neg0} -> {neg1}")
    L.append("  ln(up/down) by start tercile (low / mid / high), original -> now:")
    for t in terc0:
        L.append(f"    {t:16s} " + " ".join(f"{x:+.2f}" for x in terc0[t]) + "  ->  "
                 + " ".join(f"{x:+.2f}" for x in terc1.get(t, [float('nan')] * 3)))
    L += ["CELLS (n, p, move, spread, r_sd; original -> now; * = target)",
          f"  {'cell':38s} {'n':>3s} {'p':>13s} {'move':>13s} {'spread':>11s} {'r_sd':>11s}"]
    for k in keys:
        a, b = org[k], now[k]
        flag = "*" if k in tgt else ("x" if k in move_fail else " ")
        L.append(f" {flag}{k:38s} {a['n']:3d} {a['p']:+5.2f}->{b['p']:+5.2f} {a['move']:+5.2f}->{b['move']:+5.2f} "
                 f"{a['spread']:4.2f}->{b['spread']:4.2f} {a['r_sd']:4.2f}->{b['r_sd']:4.2f}")
    return {"headline": head, "test_rows": test_rows, "other_rows": other_rows, "report_only": rep_items,
            "tercile": {"orig": terc0, "now": terc1}, "cells": now, "cells_orig": org}, "\n".join(L)
