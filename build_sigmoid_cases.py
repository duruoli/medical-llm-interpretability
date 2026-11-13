"""Build the sigmoid-CT likelihood-ratio check (behavior only, no patching).

Question: does a negative CT for diverticulitis move the model less than it should,
relative to a positive one, when the yardstick is the literature likelihood ratio?

Each case = one real MIMIC `Patient History` (same leak filter as build_posneg_cases.py)
+ one CT result line. P and N lines need NOT be token-aligned here, so they use wording
that real MIMIC CT reports use (checked against congraph/data/raw_data reports):
  positive: "wall thickening and fat stranding ... diverticula ... consistent with acute ___"
  negative: "diverticulosis without evidence of acute diverticulitis", "no evidence of ...",
            "sigmoid colon and rectum are normal in appearance"
The numeric form ("sigmoid wall 8 mm") is dropped: real reports almost never measure it.

Two tests on the SAME histories (same scan, different organ):
  ct_sigmoid  -> target diverticulitis (the suspected flaw)
  ct_appendix -> target appendicitis   (control: earlier run had a ~1:1 ratio, as literature)

Two question formats per prompt:
  mc4    which of 4 diagnoses (as before)       m = logit(target) - logsumexp(other 3)
  yesno  "does this patient have <target>?"     m = logit(Yes) - logit(No)
yesno is the closer match to a likelihood ratio (target vs not-target).

Run (repo root, needs the MIMIC pickles, e.g. from congraph):
  python build_sigmoid_cases.py --raw-dir ../congraph/data/raw_data
Output: data/sigmoid_pool.jsonl   (scp to Quest, then sigmoid_lr.slurm)
"""
import argparse
import json
import random
from pathlib import Path

from build_posneg_cases import CHOICES, DISEASES, LEAK, PROMPT, clean, load_histories

YESNO_PROMPT = (
    "An emergency department patient presents with abdominal pain.\n\n"
    "Patient history:\n{history}\n\n"
    "New test result:\n{result}\n\n"
    "Does this patient have {target_name}?\n"
    "\nAnswer Yes or No only."
)

BASELINE = "CT abdomen/pelvis: result pending."

# pair -> (P line, N line). Same pair keys in both tests so they can be compared.
#   lexical          the earlier template (continuity with the first run)
#   impression       short radiologist conclusion
#   full_impression  the commonest real wording; for sigmoid the N side mentions
#                    "diverticulosis", a near-homonym of the target
#   findings         organ description without the disease name
TESTS = {
    "ct_sigmoid": {
        "target": "C", "disease": "diverticulitis", "target_name": "acute diverticulitis",
        "pairs": {
            "lexical": ("CT abdomen/pelvis: sigmoid colon wall thick.",
                        "CT abdomen/pelvis: sigmoid colon wall normal."),
            "impression": ("CT abdomen/pelvis impression: Acute sigmoid diverticulitis.",
                           "CT abdomen/pelvis impression: No evidence of diverticulitis."),
            "full_impression": (
                "CT abdomen/pelvis impression: Sigmoid diverticulosis with wall thickening and "
                "pericolonic fat stranding, consistent with acute diverticulitis.",
                "CT abdomen/pelvis impression: Sigmoid diverticulosis without evidence of acute "
                "diverticulitis."),
            "findings": (
                "CT abdomen/pelvis: Focal wall thickening of the sigmoid colon with surrounding fat "
                "stranding and multiple sigmoid diverticula.",
                "CT abdomen/pelvis: The sigmoid colon and rectum are normal in appearance."),
        },
    },
    "ct_appendix": {
        "target": "A", "disease": "appendicitis", "target_name": "acute appendicitis",
        "pairs": {
            "lexical": ("CT abdomen/pelvis: appendix enlarged.",
                        "CT abdomen/pelvis: appendix normal."),
            "impression": ("CT abdomen/pelvis impression: Acute appendicitis.",
                           "CT abdomen/pelvis impression: No evidence of appendicitis."),
            "full_impression": (
                "CT abdomen/pelvis impression: Dilated appendix with wall thickening and "
                "periappendiceal fat stranding, consistent with acute appendicitis.",
                "CT abdomen/pelvis impression: Normal appendix, without evidence of acute "
                "appendicitis."),
            "findings": (
                "CT abdomen/pelvis: The appendix is dilated to 12 mm with wall thickening and "
                "surrounding fat stranding.",
                "CT abdomen/pelvis: The appendix is normal in caliber and appearance."),
            # first run: "9 mm" sometimes LOWERED appendicitis. Real reports measure the
            # appendix often (654/957), so keep it here; sigmoid has no real numeric form.
            "numeric": ("CT abdomen/pelvis: appendix diameter 9 mm (normal under 6 mm).",
                        "CT abdomen/pelvis: appendix diameter 4 mm (normal under 6 mm)."),
        },
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--per-disease", type=int, default=50, help="histories per true disease (cap)")
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hist = [dict(r, history=clean(r["history"])) for r in load_histories(Path(args.raw_dir))]
    ok = [r for r in hist if args.min_chars <= len(r["history"]) <= args.max_chars
          and not LEAK.search(r["history"])]
    cases = []
    for dz in DISEASES:
        rs = [r for r in ok if r["true_disease"] == dz]
        rng.shuffle(rs)
        for r in rs[:args.per_disease]:
            h = r["history"]
            c = {"hadm_id": r["hadm_id"], "true_disease": dz, "choices": CHOICES,
                 "mc4_B": PROMPT.format(history=h, result=BASELINE), "tests": {}}
            for test_id, spec in TESTS.items():
                yn = lambda res: YESNO_PROMPT.format(history=h, result=res,
                                                     target_name=spec["target_name"])
                c["tests"][test_id] = {
                    "target_letter": spec["target"], "target_disease": spec["disease"],
                    "yesno_B": yn(BASELINE),
                    "pairs": {k: {"P_result": p, "N_result": n,
                                  "mc4_P": PROMPT.format(history=h, result=p),
                                  "mc4_N": PROMPT.format(history=h, result=n),
                                  "yesno_P": yn(p), "yesno_N": yn(n)}
                              for k, (p, n) in spec["pairs"].items()},
                }
            cases.append(c)
        print(f"{dz}: {min(len(rs), args.per_disease)} of {len(rs)} filtered histories")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
