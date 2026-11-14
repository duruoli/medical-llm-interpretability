"""Fixed settings for the weight-edit harness (positive-evidence bias in Qwen3.5-9B).

Everything the researcher agent may NOT change lives here: literature yardsticks,
held-out wordings/tests, forbidden parameters, budgets.
"""
import math
import re

# ---- literature likelihood ratios (sens, spec) ---------------------------------
# ideal Bayesian move in log-odds: up = ln(sens/(1-spec)), down = ln(spec/(1-sens))
LIT = {
    "ct_sigmoid":     (0.94, 0.99),  # CT for diverticulitis, Lameris 2008 Eur Radiol
    "ct_appendix":    (0.95, 0.94),  # contrast CT for appendicitis, pooled meta-analyses
    "us_gallbladder": (0.71, 0.85),  # US for acute cholecystitis, Huang 2023 WJES
    "lipase":         (0.77, 0.92),  # lipase >3x ULN for pancreatitis (ranges .75-.79 / .89-.95)
}


def lit_moves(test):
    sens, spec = LIT[test]
    return math.log(sens / (1 - spec)), math.log(spec / (1 - sens))


PB_RANGE = (0.02, 0.95)   # "mid prior": original model's baseline P(Yes) in this range
DOWN_FLOOR = 0.1          # mean move below this is floored when taking log ratios

# ---- prompts --------------------------------------------------------------------
YESNO_PROMPT = (
    "An emergency department patient presents with abdominal pain.\n\n"
    "Patient history:\n{history}\n\n"
    "New test result:\n{result}\n\n"
    "Does this patient have {target_name}?\n"
    "\nAnswer Yes or No only."
)

# Tests seen in practice come from data/sigmoid_pool.jsonl (build_sigmoid_cases.py).
# "numeric" (appendix 9 mm) is dropped: it is a missing ability, not an edit target.
PRACTICE_PAIRS = ["lexical", "impression", "full_impression", "findings"]

# Extra wordings that appear ONLY in the test split (new phrasing, same tests).
HELDOUT_WORDINGS = {
    "ct_sigmoid": {
        "new_impression": (
            "CT abdomen/pelvis impression: Findings compatible with uncomplicated sigmoid diverticulitis.",
            "CT abdomen/pelvis impression: No acute intra-abdominal process. The colon is unremarkable."),
    },
    "ct_appendix": {
        "new_impression": (
            "CT abdomen/pelvis impression: Findings consistent with acute uncomplicated appendicitis.",
            "CT abdomen/pelvis impression: The appendix is visualized and is normal. No acute "
            "intra-abdominal process."),
    },
}

# Whole tests that appear ONLY in the test split (new organ, new disease).
HELDOUT_TESTS = {
    "us_gallbladder": {
        "target_name": "acute cholecystitis",
        "baseline": "Abdominal ultrasound: result pending.",
        "pairs": {
            "impression": ("Abdominal ultrasound impression: Acute cholecystitis.",
                           "Abdominal ultrasound impression: No sonographic evidence of acute "
                           "cholecystitis."),
            "full_impression": (
                "Abdominal ultrasound impression: Cholelithiasis with gallbladder wall thickening, "
                "pericholecystic fluid and a positive sonographic Murphy sign, consistent with acute "
                "cholecystitis.",
                "Abdominal ultrasound impression: Cholelithiasis without sonographic evidence of acute "
                "cholecystitis."),
            "findings": (
                "Abdominal ultrasound: Distended gallbladder with wall thickening, pericholecystic fluid "
                "and a positive sonographic Murphy sign.",
                "Abdominal ultrasound: The gallbladder is normal without stones, wall thickening or "
                "pericholecystic fluid."),
        },
    },
    "lipase": {
        "target_name": "acute pancreatitis",
        "baseline": "Serum lipase: result pending.",
        "pairs": {
            "numeric": ("Serum lipase: 1240 U/L (reference 13-60).",
                        "Serum lipase: 32 U/L (reference 13-60)."),
            "lexical": ("Serum lipase markedly elevated, more than 3 times the upper limit of normal.",
                        "Serum lipase within normal limits."),
        },
    },
}

# ---- guards -----------------------------------------------------------------------
# General yes/no questions: an edit that just pushes "No" (or "Yes") everywhere shows here.
GENERAL_YESNO = [
    ("Is water made of hydrogen and oxygen?", "Yes"),
    ("Is the Pacific Ocean the largest ocean on Earth?", "Yes"),
    ("Is Paris the capital of France?", "Yes"),
    ("Do penguins live naturally at the North Pole?", "No"),
    ("Is 17 a prime number?", "Yes"),
    ("Is 21 a prime number?", "No"),
    ("Does the sun rise in the west?", "No"),
    ("Is a tomato botanically a fruit?", "Yes"),
    ("Is the Great Wall located in Brazil?", "No"),
    ("Do humans have three lungs?", "No"),
    ("Is iron a metal?", "Yes"),
    ("Is the boiling point of water at sea level 50 degrees Celsius?", "No"),
    ("Was the Eiffel Tower built in the 20th century?", "No"),
    ("Is Mount Everest the highest mountain above sea level?", "Yes"),
    ("Can a triangle have four sides?", "No"),
    ("Is the heart a muscle?", "Yes"),
    ("Do plants use sunlight to make food?", "Yes"),
    ("Is 100 divisible by 7?", "No"),
    ("Is English written from right to left?", "No"),
    ("Is the moon larger than the Earth?", "No"),
    ("Does a normal chest X-ray rule out every possible lung disease?", "No"),
    ("Is aspirin a medication?", "Yes"),
    ("Is it true that whales are not mammals?", "No"),
    ("Is it true that snow is not hot?", "Yes"),
    ("Is the number 0 an even number?", "Yes"),
    ("Is Tokyo in China?", "No"),
    ("Do fish breathe with gills?", "Yes"),
    ("Is the human body temperature normally around 37 degrees Celsius?", "Yes"),
    ("If a blood test for a disease is normal, does that make the disease less likely?", "Yes"),
    ("If a test result is negative, does that usually make the disease more likely?", "No"),
    ("Is gold heavier than the same volume of water?", "Yes"),
    ("Do all birds fly?", "No"),
    ("Is a week seven days long?", "Yes"),
    ("Is Antarctica a warm desert?", "No"),
    ("Does a square have equal sides?", "Yes"),
    ("Is silver a gas at room temperature?", "No"),
    ("Are there 24 hours in a day?", "Yes"),
    ("Is it true that a normal CT scan makes appendicitis less likely?", "Yes"),
    ("Is it true that an elevated white blood cell count rules out infection?", "No"),
    ("Is Shakespeare known for writing plays?", "Yes"),
]
GENERAL_SUFFIX = "\n\nAnswer Yes or No only."
NLL_TEXTS = 24       # practice histories used for the language-modelling guard
NLL_MAX_TOKENS = 256

# ---- edit rules ---------------------------------------------------------------------
# No embeddings, no output head, no final norm, no vision tower: these are "global"
# knobs (the blog's rule: no embedding / lm_head edits, no global scaling).
FORBIDDEN = re.compile(r"embed|lm_head|visual|vision|(^|\.)norm\.weight$")
MAX_NNZ = 5_000_000       # max changed weights per edit (~0.06% of 9B)

# ---- budget -------------------------------------------------------------------------
BUDGET = {
    "score_calls": 60,      # full practice evaluations (blog: 50 forward-pass calls)
    "probe_prompts": 4000,  # prompts run through /logits and /acts, summed
}
MAX_PROBE_PER_CALL = 256
