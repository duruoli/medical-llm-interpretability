"""Case pool for the neuron screen (neuron_posneg.py): 8 tests x 3 forms x 2 wordings x patients.

Each record = one patient history + one test + one wording, with P and N per form (same fields as
build_posneg_cases.py, plus `wording`). Two tests per target diagnosis, so a whole test can be held out.
P and N of one form have the same token count and differ in one contiguous span (checked here).

Run (repo root; needs the MIMIC hadm_info pickles):
    python build_neuron_pool.py --raw-dir ../congraph/data/raw_data
Output: data/posneg_pool_v2.jsonl    (git-ignored: contains MIMIC text)
"""
import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from build_posneg_cases import CHOICES, DISEASES, LEAK, PROMPT, check_aligned, clean, load_histories

CT = "CT abdomen/pelvis"
RUQ = "Right upper quadrant ultrasound"
RLQ = "Right lower quadrant ultrasound"
PN = ("positive", "negative")
LEX = ("abnormal", "normal")


def forms(numeric, lexical, negation):
    return {"numeric": numeric, "lexical": lexical, "negation": negation}


# test_id -> target letter, disease, baseline, and per form [original wording, second wording],
# each wording = (template, P value, N value)
TESTS = {
    "lipase": dict(target="D", disease="pancreatitis", baseline="Serum lipase: result pending.", forms=forms(
        [("Serum lipase: {v} U/L (reference interval 0-160 U/L).", "980", "120"),
         ("Serum lipase measures {v} U/L (upper limit of normal 160 U/L).", "980", "120")],
        [("Serum lipase: {v}.", "elevated", "normal"), ("Result: {v} serum lipase.", *LEX)],
        [("Serum lipase: {v} elevated.", "markedly", "not"), ("Serum lipase: {v} for acute pancreatitis.", *PN)])),
    "amylase": dict(target="D", disease="pancreatitis", baseline="Serum amylase: result pending.", forms=forms(
        [("Serum amylase: {v} U/L (reference interval 30-110 U/L).", "760", "105"),
         ("Serum amylase measures {v} U/L (upper limit of normal 110 U/L).", "760", "105")],
        [("Serum amylase: {v}.", "elevated", "normal"), ("Result: {v} serum amylase.", *LEX)],
        [("Serum amylase: {v} elevated.", "markedly", "not"), ("Serum amylase: {v} for acute pancreatitis.", *PN)])),
    "ct_appendix": dict(target="A", disease="appendicitis", baseline=f"{CT}: result pending.", forms=forms(
        [(f"{CT}: appendix diameter {{v}} mm (normal under 6 mm).", "9", "4"),
         (f"{CT}: appendix measures {{v}} mm in diameter (upper limit of normal 6 mm).", "9", "4")],
        [(f"{CT}: appendix {{v}}.", "enlarged", "normal"), (f"{CT}: {{v}} appendix.", *LEX)],
        [(f"{CT} impression: {{v}} appendicitis.", "acute", "no"), (f"{CT} impression: {{v}} for acute appendicitis.", *PN)])),
    "us_appendix": dict(target="A", disease="appendicitis", baseline=f"{RLQ}: result pending.", forms=forms(
        [(f"{RLQ}: appendix diameter {{v}} mm (normal under 6 mm).", "9", "4"),
         (f"{RLQ}: appendix measures {{v}} mm in diameter (upper limit of normal 6 mm).", "9", "4")],
        [(f"{RLQ}: appendix {{v}}.", "enlarged", "normal"), (f"{RLQ}: {{v}} appendix.", *LEX)],
        [(f"{RLQ} impression: {{v}} appendicitis.", "acute", "no"), (f"{RLQ} impression: {{v}} for acute appendicitis.", *PN)])),
    "us_gallbladder": dict(target="B", disease="cholecystitis", baseline=f"{RUQ}: result pending.", forms=forms(
        [(f"{RUQ}: gallbladder wall thickness {{v}} mm (normal up to 3 mm).", "6", "2"),
         (f"{RUQ}: gallbladder wall measures {{v}} mm thick (upper limit of normal 3 mm).", "6", "2")],
        [(f"{RUQ}: gallbladder wall {{v}}.", "thick", "normal"), (f"{RUQ}: {{v}} gallbladder wall.", *LEX)],
        [(f"{RUQ} impression: {{v}} cholecystitis.", "acute", "no"), (f"{RUQ} impression: {{v}} for acute cholecystitis.", *PN)])),
    "hida": dict(target="B", disease="cholecystitis", baseline="HIDA scan: result pending.", forms=forms(
        [("HIDA scan: gallbladder filling time {v} minutes (normal under 60 minutes).", "90", "30"),
         ("HIDA scan: gallbladder fills in {v} minutes (upper limit of normal 60 minutes).", "90", "30")],
        [("HIDA scan: gallbladder filling {v}.", "absent", "normal"), ("HIDA scan: {v} gallbladder filling.", *LEX)],
        [("HIDA scan impression: {v} cholecystitis.", "acute", "no"), ("HIDA scan impression: {v} for acute cholecystitis.", *PN)])),
    "ct_sigmoid": dict(target="C", disease="diverticulitis", baseline=f"{CT}: result pending.", forms=forms(
        [(f"{CT}: sigmoid colon wall thickness {{v}} mm (normal up to 3 mm).", "8", "2"),
         (f"{CT}: sigmoid colon wall measures {{v}} mm thick (upper limit of normal 3 mm).", "8", "2")],
        [(f"{CT}: sigmoid colon wall {{v}}.", "thick", "normal"), (f"{CT}: {{v}} sigmoid colon wall.", *LEX)],
        [(f"{CT} impression: {{v}} diverticulitis.", "acute", "no"), (f"{CT} impression: {{v}} for acute diverticulitis.", *PN)])),
    "ct_pericolic": dict(target="C", disease="diverticulitis", baseline=f"{CT}: result pending.", forms=forms(
        [(f"{CT}: pericolic fat density {{v}} HU (normal under -90 HU).", "-40", "-95"),
         (f"{CT}: pericolic fat measures {{v}} HU (upper limit of normal -90 HU).", "-40", "-95")],
        [(f"{CT}: pericolic fat {{v}}.", "stranded", "normal"), (f"{CT}: {{v}} pericolic fat.", *LEX)],
        [(f"{CT} impression: {{v}} colonic diverticulitis.", "acute", "no"), (f"{CT} impression: {{v}} for acute diverticulitis.", *PN)])),
}
N_WORDINGS = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default="data/posneg_pool_v2.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-14B")
    ap.add_argument("--patients-per-test", type=int, default=10)
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)
    ok = [dict(r, history=clean(r["history"])) for r in load_histories(Path(args.raw_dir))]
    ok = [r for r in ok if args.min_chars <= len(r["history"]) <= args.max_chars and not LEAK.search(r["history"])]
    by_dz = {dz: [r for r in ok if r["true_disease"] == dz] for dz in DISEASES}
    for v in by_dz.values():
        rng.shuffle(v)

    used, cases = set(), []
    for test_id, spec in TESTS.items():
        # half from the target disease, half spread over the other three (as in build_posneg_cases.py)
        others = [dz for dz in DISEASES if dz != spec["disease"]]
        plan = [spec["disease"]] * (args.patients_per_test // 2)
        plan += [others[i % 3] for i in range(args.patients_per_test - len(plan))]
        for dz in plan:
            r = next((r for r in by_dz[dz] if r["hadm_id"] not in used), None)
            if r is None:
                raise SystemExit(f"not enough {dz} histories")
            used.add(r["hadm_id"])
            for w in range(N_WORDINGS):
                row = {}
                for form, wordings in spec["forms"].items():
                    tmpl, pv, nv = wordings[w]
                    p_result, n_result = tmpl.format(v=pv), tmpl.format(v=nv)
                    p, n = (PROMPT.format(history=r["history"], result=x) for x in (p_result, n_result))
                    if not check_aligned(tok, p, n):
                        raise SystemExit(f"P/N not token-aligned: {test_id}/{form}/wording {w}")
                    row[form] = {"P": p, "N": n, "P_result": p_result, "N_result": n_result}
                cases.append(dict(
                    case_id=f"{test_id}__{r['hadm_id']}__w{w}", test_id=test_id, hadm_id=r["hadm_id"], wording=w,
                    true_disease=dz, target_letter=spec["target"], target_disease=spec["disease"], choices=CHOICES,
                    B=PROMPT.format(history=r["history"], result=spec["baseline"]), B_result=spec["baseline"], forms=row))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"{len(cases)} records, {len(used)} patients, {len(TESTS)} tests -> {out}")


if __name__ == "__main__":
    main()
