"""Held-out evaluation of finished edits. Run by the HUMAN, not the researcher agent.

Scores the original model once, then each edit (at one or more scales) and a random control
(same parameters, same support, same norm, random content) on the TEST split of a suite.

  v1:      python -m harness.final_eval --edit $HARNESS_OUT/edits/007_x.pt --scales 0.5 1
  repair:  python -m harness.final_eval --suite repair --out $HARNESS_OUT \
             --edit FINAL_A.pt FINAL_B.pt FINAL_C.pt --arm A_readout B_translate C_free
Output: <out>/final/<edit stem>.md (+ .json) per edit.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from harness.config import SUITES, arm_check
from harness.core import (Model, load_edit, load_items, mid_cases, random_control, report,
                          run_guards, run_items)

LAB = {"lipase", "amylase"}


def group(cell, suite):
    test, pair = cell.split("/")
    if test in suite["heldout_tests"]:
        return "new test"
    if pair in suite["heldout_wordings"].get(test, {}):
        return "new wording"
    return "seen test+wording, new patients"


def kind(cell):
    return "lab" if cell.split("/")[0] in LAB else "imaging"


def table(rep, key_fn):
    groups = {}
    for k, c in rep["cells"].items():
        groups.setdefault(key_fn(k), []).append((c["asym"], rep["cells_orig"][k]["asym"]))
    return "\n".join(
        f"  {g:34s} asym {np.mean([o for _, o in v]):+.3f} -> {np.mean([a for a, _ in v]):+.3f}   "
        f"|asym| {np.mean([abs(o) for _, o in v]):.3f} -> {np.mean([abs(a) for a, _ in v]):.3f}  ({len(v)} cells)"
        for g, v in groups.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--suite", default="v1", choices=list(SUITES))
    ap.add_argument("--test", default=None, help="default: the suite's test file")
    ap.add_argument("--edit", required=True, nargs="+")
    ap.add_argument("--arm", nargs="*", default=None, help="one arm per edit (checked before running)")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--out", default=os.environ.get("HARNESS_OUT", "out/harness"))
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    suite = SUITES[args.suite]
    if args.arm and len(args.arm) != len(args.edit):
        raise SystemExit("--arm needs exactly one arm per --edit")

    edits = [(Path(e), load_edit(e), args.arm[i] if args.arm else None) for i, e in enumerate(args.edit)]
    for path, d, arm in edits:          # fail fast, before the long runs
        for n in d:
            arm_check(n, arm)

    out = Path(args.out) / "final"
    out.mkdir(parents=True, exist_ok=True)
    model = Model(args.model, args.device)
    items = load_items(args.test or suite["test"], args.limit_cases)

    t0 = time.time()
    ref, g_ref = run_items(model, items), run_guards(model, items)
    mid = mid_cases(ref, items)
    print(f"original scored ({time.time() - t0:.0f}s)", flush=True)

    for path, deltas, arm in edits:
        arms = [(f"edit x{s}", deltas, s) for s in args.scales]
        if not args.no_control:
            arms += [(f"random control x{s}", random_control(deltas), s) for s in args.scales]
        lines = [f"# Final test evaluation: {path.name}  (suite {args.suite}, arm {arm})\n",
                 f"model {args.model}; {len(items)} test patients; mid-prior cases per test "
                 f"{ {k: len(v) for k, v in mid.items()} }\n",
                 "asym: 0 = literature up/down ratio, >0 = positive bias; |asym| counts over- and "
                 "under-correction. Groups average over cells.\n"]
        results = {}
        for name, d, s in arms:
            stats = model.apply(d, s)
            cur, g = run_items(model, items), run_guards(model, items)
            model.reset()
            rep, text = report(cur, ref, g, g_ref, items, mid)
            lines += [f"## {name}  (nnz {stats['nnz']:,})\n", "```",
                      "BY GROUP", table(rep, lambda k: group(k, suite)),
                      "BY KIND", table(rep, kind), text, "```", ""]
            results[name] = {"stats": {k: v for k, v in stats.items() if k != "relative_norm"}, **rep}
            print(f"{path.stem} {name} done ({time.time() - t0:.0f}s)", flush=True)
        (out / f"{path.stem}.md").write_text("\n".join(lines))
        (out / f"{path.stem}.json").write_text(json.dumps(results, default=str))
        print(f"-> {out / path.stem}.md", flush=True)


if __name__ == "__main__":
    main()
