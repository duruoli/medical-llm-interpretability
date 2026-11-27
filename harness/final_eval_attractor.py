"""Held-out evaluation of attractor edits (SPEC_attractor_harness.md §10). Run by the HUMAN, never the agent.

For the original model, each edit (at each scale) and a random control (same parameters, support and norm, random
content): pull / move / r per cell on
  - the suite's TEST split: new patients (seen tests + wordings), new wordings, new tests
  - a NEW DOMAIN pool (default data/echo_pool.jsonl, cardiac/chest cases; all 188 patients)
Groups are summarised separately; the full attractor report (constraints, guards, report-only items) is printed
per data set.

  python -m harness.final_eval_attractor --model meta-llama/Llama-3.1-8B-Instruct --suite repair_llama \
      --profile llama --out <OUT> --edit FINAL.pt FALLBACK1.pt --scales 1
Output: <out>/final_attractor/<edit stem>.md (+ .json).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from harness.attractor_score import attractor_report, excluded
from harness.config import ATTRACTOR, ATTRACTOR_PROFILES, SUITES
from harness.core import Model, load_edit, load_items, mid_cases, random_control, run_guards, run_items


def group(cell, suite):
    test, pair, _ = cell.split("/")
    if test in suite["heldout_tests"]:
        return "new test"
    if pair in suite["heldout_wordings"].get(test, {}):
        return "new wording"
    return "new patients (seen test + wording)"


def group_table(now, org, profile, key_fn):
    prof = ATTRACTOR_PROFILES[profile]
    is_excl = excluded(profile)
    groups = {}
    for k in org:
        if k not in now or is_excl(k):
            continue
        groups.setdefault((key_fn(k), k.rsplit("/", 1)[1]), []).append(k)
    L = [f"  {'group':38s} side  cells  median |p| orig -> now   moves failing   spread orig -> now"]
    for (g, s), ks in sorted(groups.items()):
        p0 = np.median([abs(org[k]["p"]) for k in ks]); p1 = np.median([abs(now[k]["p"]) for k in ks])
        mf = sum(abs(now[k]["move"] - org[k]["move"]) > ATTRACTOR["move_tol"] * max(abs(org[k]["move"]), ATTRACTOR["move_floor"])
                 for k in ks)
        sp0 = np.median([org[k]["spread"] for k in ks]); sp1 = np.median([now[k]["spread"] for k in ks])
        tag = "target" if s in prof["target_sides"] else "other"
        L.append(f"  {g:38s} {s} ({tag:6s}) {len(ks):3d}   {p0:.3f} -> {p1:.3f}            {mf:3d}/{len(ks):<3d}"
                 f"        {sp0:.2f} -> {sp1:.2f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--profile", required=True, choices=list(ATTRACTOR_PROFILES))
    ap.add_argument("--domain-pool", default="data/echo_pool.jsonl", help="new-domain pool ('' = skip)")
    ap.add_argument("--edit", required=True, nargs="+")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    suite = SUITES[args.suite]
    edits = [(Path(e), load_edit(e)) for e in args.edit]
    out = Path(args.out) / "final_attractor"
    out.mkdir(parents=True, exist_ok=True)
    model = Model(args.model, args.device)

    sets = {"test split": load_items(suite["test"], args.limit_cases)}
    if args.domain_pool:
        sets["new domain"] = load_items(args.domain_pool, args.limit_cases)
    t0 = time.time()
    base = {}
    for name, items in sets.items():
        ref, g_ref = run_items(model, items), run_guards(model, items)
        base[name] = (items, ref, g_ref, mid_cases(ref, items))
        print(f"original on {name} ({len(items)} patients) {time.time() - t0:.0f}s", flush=True)

    for path, deltas in edits:
        arms = [(f"edit x{s}", deltas, s) for s in args.scales]
        if not args.no_control:
            arms += [(f"random control x{s}", random_control(deltas), s) for s in args.scales]
        lines = [f"# Held-out attractor evaluation: {path.name}", "",
                 f"model {args.model}; suite {args.suite}; profile {args.profile}; sets "
                 + ", ".join(f"{n}: {len(b[0])} patients" for n, b in base.items()), "",
                 "p = pull (1 - slope of final margin on no-result margin); target = median |p| over target cells; "
                 "move = mean push, must not change; spread = SD(final)/SD(start).", ""]
        results = {}
        for name, d, s in arms:
            stats = model.apply(d, s)
            results[name] = {"stats": {k: v for k, v in stats.items() if k != "relative_norm"}}
            lines += [f"## {name}  (nnz {stats['nnz']:,})", ""]
            for sname, (items, ref, g_ref, mid) in base.items():
                cur, g = run_items(model, items), run_guards(model, items)
                rep, text = attractor_report(cur, ref, g, g_ref, items, mid, args.profile)
                key_fn = (lambda k: group(k, suite)) if sname == "test split" else (lambda k: "new domain")
                lines += [f"### {sname}", "", "```", "BY GROUP",
                          group_table(rep["cells"], rep["cells_orig"], args.profile, key_fn), "", text, "```", ""]
                results[name][sname] = rep
                print(f"{path.stem} {name} {sname} done ({time.time() - t0:.0f}s)", flush=True)
            model.reset()
        (out / f"{path.stem}.md").write_text("\n".join(lines))
        (out / f"{path.stem}.json").write_text(json.dumps(results, default=str))
        print(f"-> {out / path.stem}.md", flush=True)


if __name__ == "__main__":
    main()
