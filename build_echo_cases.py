"""Build the MIMIC-IV-Echo case pool: same pipeline as the abdominal set, new data.

Cohort (extract_echo_mimic.sh): ED admissions with >= 1 TTE during the stay. Here we keep admissions
whose PRIMARY ICD diagnosis (seq_num 1) is one of four diseases of the chest-pain / shortness-of-breath
differential, the analogue of the four abdominal diseases:
    heart_failure, acute_mi, pneumonia, pulmonary_embolism

Patient history = discharge summary from "History of Present Illness" up to "Physical Exam"
(HPI + past medical / social / family history), as in the abdominal pickles. Differences from the
abdominal filter, agreed 2026-11-24, because cardiopulmonary HPIs nearly always narrate the ED work-up:
  1. the HPI is CUT at the start of the ED course ("In the ED ...", "ED course"), so the work-up
     (troponin, CXR, CTA ...) is removed instead of the whole case being dropped;
  2. the HPI must not mention a test, a candidate diagnosis, or "diagnosed / transferred / OSH ...";
  3. the past medical history MAY name old diagnoses (e.g. "CHF", "prior MI": real pre-test information)
     but must not contain a test or its result (e.g. "EF 25%", "cath 2019").
Abbreviations are matched case-sensitively ("AS", "PE", "MI", "EF"), words case-insensitively.

Output is already in harness item format (no grafting, no practice/test split: nothing is fitted on
this data; the abdominal edit is applied as is):
  {hadm_id, subject_id, true_disease, secondary_dzs, history, tests: {test: {target_name, B, pairs: {w: {P, N}}}}}
Tests, wordings and literature sens/spec: harness/config.py ECHO_TESTS / ECHO_LIT.

Run (repo root; needs data/raw/echo_mimiciv_3_1 from extract_echo_mimic.sh):
  python build_echo_cases.py            # -> data/echo_pool.jsonl  (scp to Quest)
  python -m harness.diagnose --domain echo --model Qwen/Qwen3.5-9B
"""
import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from harness.config import ECHO_PROMPT, ECHO_TESTS

DISEASES = ["heart_failure", "acute_mi", "pneumonia", "pulmonary_embolism"]
ALL_DZ = DISEASES + ["endocarditis", "pericardial", "aortic_stenosis", "copd_exacerbation"]

# start of the ED course inside the HPI: everything from here on is work-up, not history
ED_CUT = re.compile(r"(?:In|At|Upon arrival (?:to|in)|On arrival (?:to|in)|Once in|While in|On presentation to)"
                    r" (?:the |our )?(?:ED|ER|emergency)|\bED course\b|\bIn ED\b|\bIn the ED\b", re.I)

_TEST_WORDS = (r"\bCT\b|\bCTA\b|x-?ray|\bCXR\b|imaging|\bscan|ultrasound|\bU/?S\b|echo|\bTTE\b|\bTEE\b|"
               r"ejection|\bBNP|troponin|\btrop\b|\btrops\b|d-?dimer|\blabs?\b|lab work|\bWBC\b|white count|"
               r"\bECG\b|\bEKG\b|\bcath\b|catheteri|angiogra|stress test|nuclear|infiltrate|consolidation|"
               r"opacit|V/?Q\b|procalcitonin")
TEST = re.compile(_TEST_WORDS, re.I)
TEST_CS = re.compile(r"\bL?V?EF\b|\bST\b|\bSTE\b|\bLVH\b")             # case-sensitive abbreviations
DX = re.compile(r"heart failure|cardiomyopathy|pneumonia|embol|endocarditis|pericard|tamponade|effusion|"
                r"aortic stenosis|STEMI|infarct|acute coronary|COPD|emphysema|myocardial|ischemi", re.I)
DX_CS = re.compile(r"\bCHF\b|\bHF\b|\bHF[rp]?EF\b|\bPNA\b|\bCAP\b|\bHCAP\b|\bPE\b|\bDVT\b|\bAS\b|\bT?AVR\b|\bMI\b|"
                   r"\bNSTEMI\b|\bACS\b|\bCAD\b")
MISC = re.compile(r"diagnos|found to have|transferr|\bOSH\b|outside hospital|concern for|\bdx\b|"
                  r"referred|sent in|sent to the ED|was told", re.I)


def has(rx_ci, rx_cs, s):
    return bool(rx_ci.search(s) or rx_cs.search(s))


def history_parts(text):
    """(hpi_cut, pmh_rest, ed_cut_found) from a discharge summary; ('', '', False) if no HPI."""
    m = re.search(r"History of Present Illness:(.*?)(?:\n\s*Physical Exam|\n\s*Pertinent Results)", text, re.S)
    if not m:
        return "", "", False
    h = m.group(1)
    k = re.search(r"Past Medical History:", h)
    hpi, rest = (h[:k.start()], h[k.start():]) if k else (h, "")
    c = ED_CUT.search(hpi)
    return (hpi[:c.start()] if c else hpi), rest, bool(c)


def flat(s):
    return re.sub(r"\s+", " ", s).strip()


def build_tests(history):
    fill = lambda res, tn: ECHO_PROMPT.format(history=history, result=res, target_name=tn)
    return {t: {"target_name": s["target_name"], "B": fill(s["baseline"], s["target_name"]),
                "pairs": {w: {"P": fill(p, s["target_name"]), "N": fill(n, s["target_name"])}
                          for w, (p, n) in s["pairs"].items()}}
            for t, s in ECHO_TESTS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/echo_mimiciv_3_1")
    ap.add_argument("--out", default="data/echo_pool.jsonl")
    ap.add_argument("--per-disease", type=int, default=47, help="47 x 4 = 188, the size of the abdominal pool")
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    c = pd.read_parquet(raw / "cohort")
    n = pd.read_parquet(raw / "discharge_notes").sort_values("note_seq").drop_duplicates("hadm_id")
    d = c.merge(n[["hadm_id", "text"]], on="hadm_id")
    d["true_disease"] = None
    for dz in DISEASES:
        d.loc[d[f"seq_{dz}"] == 1, "true_disease"] = dz
    d = d[d.true_disease.notna()].copy()

    parts = d.text.map(history_parts)
    d["hpi"], d["pmh"], d["ed_cut"] = parts.str[0], parts.str[1], parts.str[2]
    d["history"] = (d.hpi.map(flat) + " " + d.pmh.map(flat)).str.strip()
    d["n_chars"] = d.history.str.len()

    steps = {"primary dx + note": d.true_disease.value_counts()}
    d = d[d.hpi.str.strip().str.len() > 0]
    d = d[d.n_chars.between(args.min_chars, args.max_chars)]
    steps["length ok"] = d.true_disease.value_counts()
    d = d[~d.hpi.map(lambda s: has(TEST, TEST_CS, s) or has(DX, DX_CS, s) or bool(MISC.search(s)))]
    steps["HPI clean"] = d.true_disease.value_counts()
    d = d[~d.pmh.map(lambda s: has(TEST, TEST_CS, s))]
    steps["PMH no test"] = d.true_disease.value_counts()
    # one admission per patient (the first that survives), so patients never repeat in the pool
    d = d.sort_values(["subject_id", "admittime"]).drop_duplicates("subject_id")
    steps["1 per patient"] = d.true_disease.value_counts()
    print(pd.DataFrame(steps).reindex(DISEASES).fillna(0).astype(int).to_string(), "\n")

    rng = random.Random(args.seed)
    cases = []
    for dz in DISEASES:
        rows = d[d.true_disease == dz].to_dict("records")
        rows.sort(key=lambda r: r["hadm_id"])
        rng.shuffle(rows)
        if len(rows) < args.per_disease:
            print(f"WARNING {dz}: only {len(rows)} histories (< {args.per_disease})")
        for r in rows[:args.per_disease]:
            sec = [z for z in ALL_DZ if z != dz and pd.notna(r[f"seq_{z}"])]
            cases.append({"hadm_id": int(r["hadm_id"]), "subject_id": int(r["subject_id"]),
                          "true_disease": dz, "secondary_dzs": sec, "ed_cut": bool(r["ed_cut"]),
                          "history": r["history"], "tests": build_tests(r["history"])})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for x in cases:
            f.write(json.dumps(x) + "\n")
    n_prompts = sum(1 + 2 * len(t["pairs"]) for x in cases[:1] for t in x["tests"].values()) * len(cases)
    print(f"wrote {len(cases)} cases ({n_prompts} prompts) -> {out}")


if __name__ == "__main__":
    main()
