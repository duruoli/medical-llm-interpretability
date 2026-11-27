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
    # added for the repair suite (round 6+); held out from practice
    "us_appendix":    (0.82, 0.86),  # US for appendicitis in adults, pooled 0.821/0.859 (2022 SR/MA, PMID 36130469)
    "ct_kidney_stone": (0.96, 0.94), # standard-dose non-contrast CT for urolithiasis, Dahm et al. 2022 JACEP Open
    "amylase":        (0.72, 0.93),  # amylase >3x ULN for pancreatitis, Cochrane CD012010 (Rompianesi 2017)
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

# ---- repair suite (round 6+) ------------------------------------------------------------
# Lessons from rounds 1-5: (1) the round-1 edit pushed every negative toward No and overshot the
# lab test, which practice never contained; (2) the original test split has been read by the humans.
# So: lipase moves INTO practice as a "do not break" control, and fresh wordings / tests are held out.
# Same patient split as v1 (seed 0); the agent still never sees the test patients.
REPAIR_PRACTICE_EXTRA = {"lipase": HELDOUT_TESTS["lipase"]}   # grafted onto practice histories

REPAIR_HELDOUT_WORDINGS = {
    "ct_sigmoid": {
        "new_impression": HELDOUT_WORDINGS["ct_sigmoid"]["new_impression"],
        "implicit": (   # negative without a negator or the disease word
            "CT abdomen/pelvis: Sigmoid wall thickening with adjacent inflamed diverticulum and pericolic "
            "fat stranding.",
            "CT abdomen/pelvis: The sigmoid colon has a normal wall thickness and the surrounding fat is "
            "clear."),
    },
    "ct_appendix": {
        "new_impression": HELDOUT_WORDINGS["ct_appendix"]["new_impression"],
        "implicit": (
            "CT abdomen/pelvis: The appendix is dilated to 12 mm with wall enhancement and periappendiceal "
            "fat stranding.",
            "CT abdomen/pelvis: The appendix measures 5 mm, is filled with air, and the surrounding fat is "
            "clear."),
    },
    "lipase": {
        "new_numeric": ("Serum lipase 980 U/L (normal 13-60).",
                        "Serum lipase 41 U/L (normal 13-60)."),
    },
}

REPAIR_HELDOUT_TESTS = {
    "us_gallbladder": HELDOUT_TESTS["us_gallbladder"],
    "us_appendix": {
        "target_name": "acute appendicitis",
        "baseline": "Right lower quadrant ultrasound: result pending.",
        "pairs": {
            "impression": (
                "Right lower quadrant ultrasound impression: Findings consistent with acute appendicitis.",
                "Right lower quadrant ultrasound impression: No sonographic evidence of acute appendicitis."),
            "findings": (
                "Right lower quadrant ultrasound: Noncompressible, dilated appendix measuring 10 mm with "
                "surrounding hyperemia.",
                "Right lower quadrant ultrasound: The appendix is visualized, compressible and measures 4 mm."),
        },
    },
    "ct_kidney_stone": {
        "target_name": "a kidney or ureteral stone",
        "baseline": "CT abdomen/pelvis without contrast (stone protocol): result pending.",
        "pairs": {
            "impression": (
                "CT stone protocol impression: 5 mm obstructing stone in the left distal ureter with mild "
                "hydronephrosis.",
                "CT stone protocol impression: No renal or ureteral calculi. No hydronephrosis."),
            "findings": (
                "CT stone protocol: A 5 mm calculus is seen in the left distal ureter with upstream "
                "hydroureteronephrosis.",
                "CT stone protocol: The kidneys and ureters are normal in appearance and the urinary tract "
                "is clear."),
        },
    },
    "amylase": {
        "target_name": "acute pancreatitis",
        "baseline": "Serum amylase: result pending.",
        "pairs": {
            "numeric": ("Serum amylase: 1150 U/L (reference 30-110).",
                        "Serum amylase: 64 U/L (reference 30-110)."),
            "lexical": ("Serum amylase markedly elevated, more than 3 times the upper limit of normal.",
                        "Serum amylase within normal limits."),
        },
    },
}

# ---- echo domain (MIMIC-IV-Echo cohort; build_echo_cases.py, diagnose --domain echo) --------------
# Same pipeline, new data: ED patients with chest pain / shortness of breath whose stay included a TTE.
# True disease = primary ICD diagnosis (heart_failure / acute_mi / pneumonia / pulmonary_embolism).
# Nothing is fitted here: the abdominal-domain edit is applied as is.
ECHO_PROMPT = (
    "An emergency department patient presents with chest pain or shortness of breath.\n\n"
    "Patient history:\n{history}\n\n"
    "New test result:\n{result}\n\n"
    "Does this patient have {target_name}?\n"
    "\nAnswer Yes or No only."
)

ECHO_LIT = {
    # imaging
    "tte_rwma_mi":   (0.79, 0.87),   # cardiac US for acute myocardial ischemia in the ED, 29 studies (Zarama 2024, SJTREM, PMID 38468316)
    "tte_rv_pe":     (0.53, 0.83),   # TTE "right heart strain" for PE (Fields 2017, J Am Soc Echocardiogr, PMID 28495379)
    "ctpa_pe":       (0.83, 0.96),   # CT pulmonary angiography, PIOPED II (Stein 2006, NEJM)
    "cxr_hf":        (0.73, 0.90),   # chest radiograph for acute decompensated HF (Maw 2019, JAMA Netw Open, PMID 30874784)
    "lus_hf":        (0.88, 0.90),   # lung ultrasound B-lines for acute decompensated HF (same meta-analysis)
    "cxr_pneumonia": (0.77, 0.91),   # chest radiograph vs discharge diagnosis, 5-study LUS-vs-CXR meta-analysis (PLoS ONE 2015)
    # labs (controls)
    "troponin_mi":   (0.895, 0.771), # single baseline hs-cTnT (Elecsys) in the ED, pooled (Heart 2015); TODO confirm cut-off = 14 ng/L
    "ntprobnp_hf":   (0.79, 0.84),   # NT-proBNP 900 pg/mL rule-in, age 50-75 (ICON-RELOADED, Januzzi 2018)
    "ddimer_pe":     (0.95, 0.40),   # PROVISIONAL: Cochrane CD010864 gives only ranges (sens .80-1.00, spec .23-.63); TODO pooled source
}
LIT.update(ECHO_LIT)

ECHO_TESTS = {
    "tte_rwma_mi": {
        "target_name": "an acute myocardial infarction",
        "baseline": "Transthoracic echocardiogram: result pending.",
        "pairs": {
            "lexical": ("Transthoracic echocardiogram: left ventricular wall motion abnormal.",
                        "Transthoracic echocardiogram: left ventricular wall motion normal."),
            "impression": ("Transthoracic echocardiogram impression: New regional wall motion abnormality.",
                           "Transthoracic echocardiogram impression: No regional wall motion abnormality."),
            "full_impression": (
                "Transthoracic echocardiogram impression: Hypokinesis of the inferior and inferolateral walls, "
                "suggestive of ischemia in the right coronary artery territory.",
                "Transthoracic echocardiogram impression: Normal regional and global left ventricular systolic "
                "function. No wall motion abnormality to suggest ischemia."),
            "findings": (
                "Transthoracic echocardiogram: The inferior and inferolateral walls are hypokinetic. LVEF 50%.",
                "Transthoracic echocardiogram: All left ventricular segments contract normally. LVEF 60%."),
        },
    },
    "tte_rv_pe": {
        "target_name": "a pulmonary embolism",
        "baseline": "Transthoracic echocardiogram: result pending.",
        "pairs": {
            "lexical": ("Transthoracic echocardiogram: right ventricle enlarged.",
                        "Transthoracic echocardiogram: right ventricle normal."),
            "impression": (
                "Transthoracic echocardiogram impression: Right ventricular dilation and dysfunction, consistent "
                "with acute right heart strain.",
                "Transthoracic echocardiogram impression: Normal right ventricular size and function. No "
                "echocardiographic evidence of right heart strain."),
            "findings": (
                "Transthoracic echocardiogram: The right ventricle is dilated (RV/LV ratio 1.2) with free wall "
                "hypokinesis and apical sparing.",
                "Transthoracic echocardiogram: The right ventricle is normal in size and systolic function "
                "(RV/LV ratio 0.7)."),
        },
    },
    "ctpa_pe": {
        "target_name": "a pulmonary embolism",
        "baseline": "CT pulmonary angiogram: result pending.",
        "pairs": {
            "impression": ("CT pulmonary angiogram impression: Acute pulmonary embolism.",
                           "CT pulmonary angiogram impression: No evidence of pulmonary embolism."),
            "full_impression": (
                "CT pulmonary angiogram impression: Acute pulmonary emboli in the right lower lobe segmental "
                "and subsegmental arteries.",
                "CT pulmonary angiogram impression: No pulmonary embolism to the subsegmental level. No acute "
                "intrathoracic process."),
            "findings": (
                "CT pulmonary angiogram: Filling defects are seen in the right lower lobe segmental pulmonary "
                "arteries.",
                "CT pulmonary angiogram: The pulmonary arteries opacify well without filling defects."),
        },
    },
    "cxr_hf": {
        "target_name": "acute decompensated heart failure",
        "baseline": "Chest radiograph: result pending.",
        "pairs": {
            "impression": ("Chest radiograph impression: Pulmonary edema.",
                           "Chest radiograph impression: No pulmonary edema."),
            "full_impression": (
                "Chest radiograph impression: Pulmonary vascular congestion and interstitial edema, consistent "
                "with congestive heart failure.",
                "Chest radiograph impression: No pulmonary vascular congestion or edema. No acute "
                "cardiopulmonary process."),
            "findings": (
                "Chest radiograph: Cephalization of the pulmonary vessels, interstitial edema with Kerley B "
                "lines and small bilateral pleural effusions.",
                "Chest radiograph: The pulmonary vasculature is normal. The lungs are clear without edema or "
                "effusion."),
        },
    },
    "lus_hf": {
        "target_name": "acute decompensated heart failure",
        "baseline": "Lung ultrasound: result pending.",
        "pairs": {
            "impression": (
                "Lung ultrasound impression: Diffuse bilateral B-lines, consistent with pulmonary edema.",
                "Lung ultrasound impression: Bilateral A-line pattern, no evidence of pulmonary edema."),
            "findings": (
                "Lung ultrasound: Three or more B-lines in multiple zones of both lungs.",
                "Lung ultrasound: A-lines with lung sliding in all zones; no B-line pattern."),
        },
    },
    "cxr_pneumonia": {
        "target_name": "pneumonia",
        "baseline": "Chest radiograph: result pending.",
        "pairs": {
            "impression": ("Chest radiograph impression: Right lower lobe pneumonia.",
                           "Chest radiograph impression: No evidence of pneumonia."),
            "full_impression": (
                "Chest radiograph impression: Focal consolidation in the right lower lobe, consistent with "
                "pneumonia.",
                "Chest radiograph impression: No focal consolidation to suggest pneumonia. No acute "
                "cardiopulmonary process."),
            "findings": (
                "Chest radiograph: There is a focal opacity in the right lower lobe with air bronchograms.",
                "Chest radiograph: The lungs are clear. No focal opacity, effusion or pneumothorax."),
        },
    },
    "troponin_mi": {
        "target_name": "an acute myocardial infarction",
        "baseline": "High-sensitivity troponin T: result pending.",
        "pairs": {
            "numeric": ("High-sensitivity troponin T: 86 ng/L (99th percentile upper reference limit 14 ng/L).",
                        "High-sensitivity troponin T: 6 ng/L (99th percentile upper reference limit 14 ng/L)."),
            "lexical": ("High-sensitivity troponin T elevated above the 99th percentile upper reference limit.",
                        "High-sensitivity troponin T within normal limits."),
        },
    },
    "ntprobnp_hf": {
        "target_name": "acute decompensated heart failure",
        "baseline": "NT-proBNP: result pending.",
        "pairs": {
            "numeric": ("NT-proBNP: 4850 pg/mL (reference below 125 pg/mL).",
                        "NT-proBNP: 85 pg/mL (reference below 125 pg/mL)."),
            "lexical": ("NT-proBNP markedly elevated.", "NT-proBNP within normal limits."),
        },
    },
    "ddimer_pe": {
        "target_name": "a pulmonary embolism",
        "baseline": "D-dimer: result pending.",
        "pairs": {
            "numeric": ("D-dimer: 2450 ng/mL FEU (reference below 500 ng/mL FEU).",
                        "D-dimer: 230 ng/mL FEU (reference below 500 ng/mL FEU)."),
            "lexical": ("D-dimer elevated.", "D-dimer within normal limits."),
        },
    },
}

SUITES = {
    "v1": {"practice": "data/harness/practice.jsonl", "test": "data/harness_test/test.jsonl",
           "practice_extra": {}, "heldout_wordings": HELDOUT_WORDINGS, "heldout_tests": HELDOUT_TESTS},
    # test file sits under data/harness_test/ so the existing deny rule keeps the agent out of it
    "repair": {"practice": "data/harness/repair_practice.jsonl", "test": "data/harness_test/repair_test.jsonl",
               "practice_extra": REPAIR_PRACTICE_EXTRA, "heldout_wordings": REPAIR_HELDOUT_WORDINGS,
               "heldout_tests": REPAIR_HELDOUT_TESTS},
    # Per-model split (PIPELINE.md "Per-model split", fixed 2026-11-26). Same principle as `repair`, re-applied
    # to Llama's own stage-1 numbers (diagnose q_orig, mid prior): practice = targets + one balanced control,
    # held out = at least one imaging target, one lab target and one balanced control.
    #   practice: ct_sigmoid +0.74, ct_appendix +1.14, lipase +2.27 (targets); ct_kidney_stone +0.08 (control)
    #   held out: us_appendix +2.22, amylase +1.32 (targets); us_gallbladder +0.10 (control); new wordings
    "repair_llama": {"practice": "data/harness/repair_llama_practice.jsonl",
                     "test": "data/harness_test/repair_llama_test.jsonl",
                     "practice_extra": {"lipase": HELDOUT_TESTS["lipase"],
                                        "ct_kidney_stone": REPAIR_HELDOUT_TESTS["ct_kidney_stone"]},
                     "heldout_wordings": REPAIR_HELDOUT_WORDINGS,
                     "heldout_tests": {k: v for k, v in REPAIR_HELDOUT_TESTS.items() if k != "ct_kidney_stone"}},
}

# ---- edit arms (repair suite) ------------------------------------------------------------
# Where an edit may live. Round 5: the P/N difference in the on-axis "translation" is concentrated in
# L27 attention (heads 27.11/27.13 write Yes for positives, ~nothing for negatives) and the L28 DeltaNet
# mixer; the round-1 edit was a readout at L31. Comparing arms that generalise tells us where the fault is.
ARMS = {
    "A_readout":   {"layers": range(30, 32), "modules": ("self_attn", "linear_attn", "mlp")},
    "B_translate": {"layers": range(23, 30), "modules": ("self_attn", "linear_attn")},
    "C_free":      {"layers": range(0, 64),  "modules": ("self_attn", "linear_attn", "mlp")},
}
ARM_PARAM = re.compile(r"layers\.(\d+)\.(self_attn|linear_attn|mlp)\.")


def arm_check(name, arm):
    """Raise if parameter `name` is outside edit arm `arm` (None = no arm restriction)."""
    if arm is None:
        return
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {list(ARMS)}")
    m = ARM_PARAM.search(name)
    a = ARMS[arm]
    if not m or int(m.group(1)) not in a["layers"] or m.group(2) not in a["modules"]:
        raise ValueError(f"parameter {name!r} is outside arm {arm} "
                         f"(layers {a['layers'].start}-{a['layers'].stop - 1}, modules {a['modules']})")


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

# ---- attractor target (SPEC_attractor_harness.md, fixed 2026-11-27) ------------------------------
# Per cell (test, wording, side), mid-prior cases, push d = F - B = move - p*(B - mean B) + r.
#   target:      median |p| over target cells (pull)  -> success if <= success_ratio * original
#   constraints: every cell's move within move_tol (relative; floor move_floor for small moves),
#                no target test's median |p| worse by > test_worse, no target cell p < p_floor,
#                non-target side: |median p change| < other_side_dp, guards (drift, general, nll, mass)
ATTRACTOR = {"success_ratio": 0.5, "test_worse": 0.1, "p_floor": -0.2, "move_tol": 0.15, "move_floor": 0.5,
             "other_side_dp": 0.1, "drift_max": 0.15, "nll_max": 0.01, "mass_drop": 0.02, "n_min": 12}
# target sides per model; cells of `exclude_tests` whose wording matches `exclude_wordings` are reported but
# not targeted (Llama: numeric lab wordings are a separate, missing ability - PIPELINE.md change log (e))
ATTRACTOR_PROFILES = {
    "llama": {"target_sides": "PN", "exclude_tests": ["lipase", "amylase", "troponin_mi", "ntprobnp_hf", "ddimer_pe"],
              "exclude_wordings": r"^(new_)?numeric$"},
    "qwen": {"target_sides": "N", "exclude_tests": [], "exclude_wordings": None},
}
# which disease each test asks about (AUC report item); None = not one of the pool's diseases
TEST_DISEASE = {"ct_sigmoid": "diverticulitis", "ct_appendix": "appendicitis", "us_appendix": "appendicitis",
                "us_gallbladder": "cholecystitis", "lipase": "pancreatitis", "amylase": "pancreatitis",
                "ct_kidney_stone": None,
                "tte_rwma_mi": "acute_mi", "troponin_mi": "acute_mi", "tte_rv_pe": "pulmonary_embolism",
                "ctpa_pe": "pulmonary_embolism", "ddimer_pe": "pulmonary_embolism", "cxr_hf": "heart_failure",
                "lus_hf": "heart_failure", "ntprobnp_hf": "heart_failure", "cxr_pneumonia": "pneumonia"}
