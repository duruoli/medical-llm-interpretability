"""Build the positive / negative evidence patching dataset (MIMIC abdominal).

Each case = one real MIMIC `Patient History` + one templated test result, written in
three surface forms x three conditions:

    form      numeric  : a value plus reference range, no negation word ("980" vs "120")
              lexical  : a property word, no negation word ("enlarged" vs "normal")
              negation : the negative side uses a negation word ("acute" vs "no")
    condition P (positive) / N (negative) / B (baseline: "result pending")

P and N of the same form differ only in a few tokens and have the SAME token count
(checked here with the Qwen3.5 tokenizer), so activations can be patched position by
position. B is only a behavioral anchor and is never patched, so it need not align.

The output is a POOL (default 10 histories per test). `patch_posneg.py` scores the pool
behaviorally and keeps the first k eligible cases per test (default 5 -> 20 cases).

Histories that already mention imaging, labs, a candidate diagnosis, or a relevant
operation are dropped: they would leak the answer or bypass the test.

Run (repo root; needs the MIMIC hadm_info pickles, e.g. from congraph):
  python build_posneg_cases.py --raw-dir ../congraph/data/raw_data
Output: data/posneg_pool.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
CHOICES = {"A": "Acute appendicitis", "B": "Acute cholecystitis",
           "C": "Acute diverticulitis", "D": "Acute pancreatitis"}

# test_id -> target letter, true-disease name, and the result line per form.
# "{v}" is filled with the P or N value; B uses `baseline`.
TESTS = {
    "lipase": {
        "target": "D", "disease": "pancreatitis",
        "baseline": "Serum lipase: result pending.",
        "forms": {
            "numeric":  ("Serum lipase: {v} U/L (reference interval 0-160 U/L).", "980", "120"),
            "lexical":  ("Serum lipase: {v}.", "elevated", "normal"),
            "negation": ("Serum lipase: {v} elevated.", "markedly", "not"),
        },
    },
    "ct_appendix": {
        "target": "A", "disease": "appendicitis",
        "baseline": "CT abdomen/pelvis: result pending.",
        "forms": {
            "numeric":  ("CT abdomen/pelvis: appendix diameter {v} mm (normal under 6 mm).", "9", "4"),
            "lexical":  ("CT abdomen/pelvis: appendix {v}.", "enlarged", "normal"),
            "negation": ("CT abdomen/pelvis impression: {v} appendicitis.", "acute", "no"),
        },
    },
    "us_gallbladder": {
        "target": "B", "disease": "cholecystitis",
        "baseline": "Right upper quadrant ultrasound: result pending.",
        "forms": {
            "numeric":  ("Right upper quadrant ultrasound: gallbladder wall thickness {v} mm (normal up to 3 mm).", "6", "2"),
            "lexical":  ("Right upper quadrant ultrasound: gallbladder wall {v}.", "thick", "normal"),
            "negation": ("Right upper quadrant ultrasound impression: {v} cholecystitis.", "acute", "no"),
        },
    },
    "ct_sigmoid": {
        "target": "C", "disease": "diverticulitis",
        "baseline": "CT abdomen/pelvis: result pending.",
        "forms": {
            "numeric":  ("CT abdomen/pelvis: sigmoid colon wall thickness {v} mm (normal up to 3 mm).", "8", "2"),
            "lexical":  ("CT abdomen/pelvis: sigmoid colon wall {v}.", "thick", "normal"),
            "negation": ("CT abdomen/pelvis impression: {v} diverticulitis.", "acute", "no"),
        },
    },
}

PROMPT = (
    "An emergency department patient presents with abdominal pain.\n\n"
    "Patient history:\n{history}\n\n"
    "New test result:\n{result}\n\n"
    "Which diagnosis is most likely?\n"
    + "".join(f"{k}. {v}\n" for k, v in CHOICES.items())
    + "\nAnswer with one letter only."
)

# Mentions that leak a test result, name a candidate diagnosis / organ, or imply a
# relevant prior operation. Deliberately broad: the pool is large.
LEAK = re.compile(
    r"\bCT\b|CTA|cat scan|ultrasound|\bU/?S\b|sonogra|imaging|\bMRI\b|MRCP|HIDA|ERCP|x-?ray|\bKUB\b|"
    r"radiolog|scan\b|lipase|amylase|\bLFTs?\b|\bWBC\b|white count|leukocyt|\blabs?\b|lab work|bilirubin|"
    r"append|chole|gallbladder|gall ?stone|biliary|diverticul|pancrea|sigmoid|colitis|"
    r"ectomy|found to have|diagnos|transferr|\bOSH\b|outside hospital|\bdx\b|\bimp\b|concern for",
    re.I,
)


def load_histories(raw_dir: Path):
    rows = []
    for dz in DISEASES:
        d = pd.read_pickle(raw_dir / f"{dz}_hadm_info_first_diag.pkl")
        for hadm, rec in d.items():
            h = (rec.get("Patient History") or "").strip()
            rows.append({"hadm_id": int(hadm), "true_disease": dz, "history": h})
    return rows


def clean(h: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", "\n", h)).strip()


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def check_aligned(tok, p_text, n_text):
    p = tok(chat(tok, p_text), add_special_tokens=False).input_ids
    n = tok(chat(tok, n_text), add_special_tokens=False).input_ids
    if len(p) != len(n):
        return False
    diff = [i for i, (a, b) in enumerate(zip(p, n)) if a != b]
    return bool(diff) and diff == list(range(diff[0], diff[-1] + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir with {disease}_hadm_info_first_diag.pkl (congraph/data/raw_data)")
    ap.add_argument("--out", default="data/posneg_pool.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--per-test", type=int, default=10, help="pool size per test")
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)

    hist = [dict(r, history=clean(r["history"])) for r in load_histories(Path(args.raw_dir))]
    ok = [r for r in hist if args.min_chars <= len(r["history"]) <= args.max_chars
          and not LEAK.search(r["history"])]
    by_dz = {dz: [r for r in ok if r["true_disease"] == dz] for dz in DISEASES}
    print("histories after leak/length filter:", {dz: len(v) for dz, v in by_dz.items()},
          f"of {len(hist)}")
    for v in by_dz.values():
        rng.shuffle(v)

    used, cases = set(), []
    for test_id, spec in TESTS.items():
        # half from the target disease, half spread over the other three
        others = [dz for dz in DISEASES if dz != spec["disease"]]
        plan = [spec["disease"]] * (args.per_test // 2)
        plan += [others[i % 3] for i in range(args.per_test - len(plan))]
        for dz in plan:
            r = next(r for r in by_dz[dz] if r["hadm_id"] not in used)
            used.add(r["hadm_id"])
            forms = {}
            for form, (tmpl, pv, nv) in spec["forms"].items():
                p = PROMPT.format(history=r["history"], result=tmpl.format(v=pv))
                n = PROMPT.format(history=r["history"], result=tmpl.format(v=nv))
                if not check_aligned(tok, p, n):
                    raise SystemExit(f"P/N not token-aligned: {test_id}/{form}")
                forms[form] = {"P": p, "N": n,
                               "P_result": tmpl.format(v=pv), "N_result": tmpl.format(v=nv)}
            cases.append({
                "case_id": f"{test_id}__{r['hadm_id']}",
                "test_id": test_id, "hadm_id": r["hadm_id"], "true_disease": dz,
                "target_letter": spec["target"], "target_disease": spec["disease"],
                "choices": CHOICES,
                "B": PROMPT.format(history=r["history"], result=spec["baseline"]),
                "B_result": spec["baseline"],
                "forms": forms,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cases)} pool cases -> {out}")


if __name__ == "__main__":
    main()
