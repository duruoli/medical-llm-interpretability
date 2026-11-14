"""Held-out evaluation of a finished edit. Run by the HUMAN, not the researcher agent.

Scores the original model, the edit (at one or more scales) and a random control
(same parameters, same support, same norm, random content) on the TEST split:
new patients, new wordings, and two tests never seen in practice.

  python -m harness.final_eval --edit /scratch/dlf8982/harness/edits/007_x.pt --scales 0.5 1
Output: <out>/final/<edit stem>.md  (+ .json)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from harness.config import HELDOUT_TESTS, HELDOUT_WORDINGS
from harness.core import (Model, load_edit, load_items, mid_cases, random_control, report,
                          run_guards, run_items)


def group(cell):
    test, pair = cell.split("/")
    if test in HELDOUT_TESTS:
        return "new test"
    if pair in HELDOUT_WORDINGS.get(test, {}):
        return "new wording"
    return "seen test+wording, new patients"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--test", default="data/harness_test/test.jsonl")
    ap.add_argument("--edit", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--out", default="/scratch/dlf8982/harness")
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out = Path(args.out) / "final"
    out.mkdir(parents=True, exist_ok=True)
    model = Model(args.model, args.device)
    items = load_items(args.test, args.limit_cases)
    deltas = load_edit(args.edit)

    t0 = time.time()
    ref, g_ref = run_items(model, items), run_guards(model, items)
    mid = mid_cases(ref, items)
    arms = [(f"edit x{s}", deltas, s) for s in args.scales]
    if not args.no_control:
        arms += [(f"random control x{s}", random_control(deltas), s) for s in args.scales]

    lines = [f"# Final test evaluation: {Path(args.edit).name}\n",
             f"model {args.model}; {len(items)} test patients; mid-prior cases per test "
             f"{ {k: len(v) for k, v in mid.items()} }\n",
             "asym: 0 = literature up/down ratio, >0 = positive bias. Groups average asym over cells.\n"]
    results = {}
    for name, d, s in arms:
        stats = model.apply(d, s)
        cur, g = run_items(model, items), run_guards(model, items)
        model.reset()
        rep, text = report(cur, ref, g, g_ref, items, mid)
        groups = {}
        for k, c in rep["cells"].items():
            groups.setdefault(group(k), []).append((c["asym"], rep["cells_orig"][k]["asym"]))
        gtab = "\n".join(f"  {gname:34s} asym {np.mean([o for _, o in v]):+.3f} -> "
                         f"{np.mean([a for a, _ in v]):+.3f}  ({len(v)} cells)"
                         for gname, v in groups.items())
        lines += [f"## {name}  (nnz {stats['nnz']:,})\n", "```", "BY GROUP", gtab, text, "```", ""]
        results[name] = {"stats": {k: v for k, v in stats.items() if k != "relative_norm"}, **rep}
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)

    stem = Path(args.edit).stem
    (out / f"{stem}.md").write_text("\n".join(lines))
    (out / f"{stem}.json").write_text(json.dumps(results, default=str))
    print(f"-> {out / stem}.md")


if __name__ == "__main__":
    main()
