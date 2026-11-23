"""Split data/sigmoid_pool.jsonl by PATIENT into practice / test for the weight-edit harness.

Suites (harness/config.py SUITES):
  v1      (rounds 1-5)
    practice: half the patients x {ct_sigmoid, ct_appendix} x PRACTICE_PAIRS
    test:     the other half x same tests/wordings + HELDOUT_WORDINGS + HELDOUT_TESTS
  repair  (round 6+)
    practice: as v1, plus lipase grafted onto the practice histories (a test that is NOT
              under-weighted: the "do not break it" control)
    test:     same test patients x seen tests/wordings (incl. lipase) + REPAIR_HELDOUT_WORDINGS
              + REPAIR_HELDOUT_TESTS

Same seed -> same patient split in both suites. Only needs the pool file (histories are
read back out of the prompts), so it can run on Quest directly:
  python -m harness.make_splits --suite repair
"""
import argparse
import json
import random
import re
from pathlib import Path

from harness.config import PRACTICE_PAIRS, SUITES, YESNO_PROMPT

HIST = re.compile(r"Patient history:\n(.*)\n\nNew test result:", re.S)


def graft(hist, spec):
    fill = lambda r: YESNO_PROMPT.format(history=hist, result=r, target_name=spec["target_name"])
    return {"target_name": spec["target_name"], "B": fill(spec["baseline"]),
            "pairs": {p: {"P": fill(rp), "N": fill(rn)} for p, (rp, rn) in spec["pairs"].items()}}


def item_from_pool(c, suite, with_heldout):
    hist = HIST.search(c["tests"]["ct_sigmoid"]["yesno_B"]).group(1)
    tests = {}
    for test, t in c["tests"].items():
        target_name = re.search(r"Does this patient have (.*)\?", t["yesno_B"]).group(1)
        pairs = {p: {"P": t["pairs"][p]["yesno_P"], "N": t["pairs"][p]["yesno_N"]}
                 for p in PRACTICE_PAIRS if p in t["pairs"]}
        tests[test] = {"target_name": target_name, "B": t["yesno_B"], "pairs": pairs}
    for test, spec in suite["practice_extra"].items():
        tests[test] = graft(hist, spec)
    if with_heldout:
        for test, words in suite["heldout_wordings"].items():
            tn = tests[test]["target_name"]
            for p, (rp, rn) in words.items():
                tests[test]["pairs"][p] = {k: YESNO_PROMPT.format(history=hist, result=r, target_name=tn)
                                           for k, r in (("P", rp), ("N", rn))}
        for test, spec in suite["heldout_tests"].items():
            tests[test] = graft(hist, spec)
    return {"hadm_id": c["hadm_id"], "true_disease": c["true_disease"], "history": hist,
            "tests": tests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="v1", choices=list(SUITES))
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--practice-out", default=None)
    ap.add_argument("--test-out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    suite = SUITES[args.suite]

    pool = [json.loads(l) for l in open(args.pool)]
    rng = random.Random(args.seed)
    practice, test = [], []
    for dz in sorted({c["true_disease"] for c in pool}):        # stratify by true disease
        cs = [c for c in pool if c["true_disease"] == dz]
        rng.shuffle(cs)
        half = len(cs) // 2
        practice += [item_from_pool(c, suite, False) for c in cs[:half]]
        test += [item_from_pool(c, suite, True) for c in cs[half:]]

    for path, items in ((args.practice_out or suite["practice"], practice),
                        (args.test_out or suite["test"], test)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        n_prompts = sum(1 + 2 * len(t["pairs"]) for it in items for t in it["tests"].values())
        print(f"{p}: {len(items)} patients, {n_prompts} prompts")


if __name__ == "__main__":
    main()
