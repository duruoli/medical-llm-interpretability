"""Split data/sigmoid_pool.jsonl by PATIENT into practice / test for the weight-edit harness.

practice (agent sees): half the patients x {ct_sigmoid, ct_appendix} x PRACTICE_PAIRS
test (agent never sees): the other half x
    - the same tests and wordings           (new patients)
    - HELDOUT_WORDINGS for the same tests   (new phrasing)
    - HELDOUT_TESTS us_gallbladder, lipase  (new organ / disease)

Only needs the pool file (histories are read back out of the prompts), so it can run
on Quest directly:
  python -m harness.make_splits
Output: data/harness/practice.jsonl, data/harness_test/test.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

from harness.config import HELDOUT_TESTS, HELDOUT_WORDINGS, PRACTICE_PAIRS, YESNO_PROMPT

HIST = re.compile(r"Patient history:\n(.*)\n\nNew test result:", re.S)


def item_from_pool(c, with_heldout):
    hist = HIST.search(c["tests"]["ct_sigmoid"]["yesno_B"]).group(1)
    tests = {}
    for test, t in c["tests"].items():
        target_name = re.search(r"Does this patient have (.*)\?", t["yesno_B"]).group(1)
        pairs = {p: {"P": t["pairs"][p]["yesno_P"], "N": t["pairs"][p]["yesno_N"]}
                 for p in PRACTICE_PAIRS if p in t["pairs"]}
        if with_heldout:
            for p, (rp, rn) in HELDOUT_WORDINGS.get(test, {}).items():
                pairs[p] = {k: YESNO_PROMPT.format(history=hist, result=r, target_name=target_name)
                            for k, r in (("P", rp), ("N", rn))}
        tests[test] = {"target_name": target_name, "B": t["yesno_B"], "pairs": pairs}
    if with_heldout:
        for test, spec in HELDOUT_TESTS.items():
            fill = lambda r: YESNO_PROMPT.format(history=hist, result=r,
                                                 target_name=spec["target_name"])
            tests[test] = {"target_name": spec["target_name"], "B": fill(spec["baseline"]),
                           "pairs": {p: {"P": fill(rp), "N": fill(rn)}
                                     for p, (rp, rn) in spec["pairs"].items()}}
    return {"hadm_id": c["hadm_id"], "true_disease": c["true_disease"], "history": hist,
            "tests": tests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--practice-out", default="data/harness/practice.jsonl")
    ap.add_argument("--test-out", default="data/harness_test/test.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool)]
    rng = random.Random(args.seed)
    practice, test = [], []
    for dz in sorted({c["true_disease"] for c in pool}):        # stratify by true disease
        cs = [c for c in pool if c["true_disease"] == dz]
        rng.shuffle(cs)
        half = len(cs) // 2
        practice += [item_from_pool(c, False) for c in cs[:half]]
        test += [item_from_pool(c, True) for c in cs[half:]]

    for path, items in ((args.practice_out, practice), (args.test_out, test)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        n_prompts = sum(1 + 2 * len(t["pairs"]) for it in items for t in it["tests"].values())
        print(f"{p}: {len(items)} patients, {n_prompts} prompts")


if __name__ == "__main__":
    main()
