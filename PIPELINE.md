# Pipeline: diagnose → repair → evaluate (any model)

Goal: take a new model X and answer, with the same yardstick every time,
(1) does X weight negative results less than positive ones, relative to the literature (asym > 0)?
(2) can a small weight edit repair it without breaking anything? (3) does the repair hold on
patients, wordings and tests the edit never saw?

**Repair target (fixed 2026-11-26): the RELATIVE weight of negative vs positive evidence only**, i.e. bring
each test's asym toward 0 so that negatives stop counting less than positives. The model's overall
over-reaction (both up and down larger than the literature; the scale k in `results/diagnose`) is a
separate problem and is not repaired; an edit only must not make it worse (`move_err` guard). An argument
of the form "the down move is already N× the literature" is not a reason against an edit; the relative
version of that worry is handled by the control test (below).

The framework below is fixed. Details (thresholds, budgets, arms) may change, but every change is
written into the "Change log" at the bottom with a reason, so results across models stay comparable.

## Fixed across models

| what | where |
|---|---|
| 188 MIMIC histories, patient split (seed 0, 94/94 stratified by true disease) | `data/sigmoid_pool.jsonl`, `harness/make_splits.py` |
| prompts (Yes/No question), 7 tests, all wordings | `harness/config.py` (suite `repair`) |
| literature sens/spec → ideal up/down moves | `config.LIT`, `config.lit_moves` |
| metric: `asym = ln(up/down) − ln(ideal up/ideal down)`, mid-prior cases only (P(Yes) 0.02–0.95 on the original model) | `harness/core.py: cell_stats` |
| repair score: mean \|asym\| over (test, wording) cells | `core.report` |
| guards: no-result drift, general yes/no 40/40, history NLL | `core.report` |

## Per-model split (fixed 2026-11-26)

What is fixed across models is the **principle** of the practice / held-out split, not the list of tests.
Each model gets its own suite in `config.SUITES`, chosen from ITS stage-1 numbers (diagnose, original
question, mid prior), before any repair:

- **Practice = repair targets + one control.** Targets: tests with asym clearly > 0 on this model.
  Control: one test that is already balanced (asym ≈ 0) on this model, the "do not break it" check.
- **Held out = at least one imaging target, one lab target and one balanced control**, plus the new wordings.
  The held-out targets test transfer; the held-out control tests selectivity.
- The patient split (seed 0, 94/94) never changes.
- Record the suite and the stage-1 asym of every test in a `config.py` comment and in the change log.

| model | suite | practice targets | practice control | held-out targets | held-out control |
|---|---|---|---|---|---|
| Qwen3.5-9B | `repair` | CT sigmoid, CT appendix | lipase (−0.41) | US appendix, CT kidney stone, US gallbladder | amylase (−0.22) |
| Llama-3.1-8B | `repair_llama` | CT sigmoid +0.74, CT appendix +1.14, lipase +2.27 | CT kidney stone +0.08 | US appendix +2.22, amylase +1.32 | US gallbladder +0.10 |

The stage-1 gate's "DISCUSS: labs biased too" therefore no longer blocks a repair: it only means the
lab test is a target, not the control, on that model.

## Stage 1 — diagnose (behaviour only, no edits)

```
python -m harness.diagnose --model <hf id> --out /projects/p53430/medllm/diagnose/<model>   # → REPORT.md; scp back to results/diagnose/<model>/
```

All 188 patients × all tests and wordings on the original model. The report starts with a GATE:

1. **Validity checks**: Yes/No are single tokens; P(Yes)+P(No) ≥ 0.90; general yes/no ≥ 0.85;
   the model reads the history (median AUC of the no-result belief for the true disease ≥ 0.65);
   ≥ 30 mid-prior cases per test. Any failure → **STOP**: the asym numbers are not meaningful
   (the 0.8B lesson).
2. **Bias pattern**:
   - **PROCEED**: most imaging tests have a 95% CI above 0, and labs are not biased (mean asym ≤ 0.3).
   - **DISCUSS**: labs are biased too, or the bias is reversed. The repair recipe assumes that lab
     tests are an unbiased control, so it does not apply as is.
   - **NO BIAS**: report this as a negative result. No repair.

## Stage 2 — repair (harness server + researcher agent)

Same server, scoring and guards as round 6 (`harness/server.py`); the agent sees only the practice half of
the model's own suite (see "Per-model split") and has a fixed budget. **Order fixed 2026-11-26: blind first,
the Qwen pattern last**, so that a match with Qwen is evidence of shared structure, not of a guided search.

- **2B — blind free exploration (first).** The agent gets the harness, the rules, the metric and the
  model's practice split, and NOTHING learned on earlier models. It works in a clean copy built by
  `blind_setup.py` (code with neutral docstrings, a rebuilt `config.py` without comments or held-out
  names, the practice file, a blind brief such as `HANDOFF_blind_llama.md`, deny rules for the repo and
  earlier round directories; the script refuses to finish if a leak word is found). One free arm (any
  layer, attention or MLP). Budget 100 score calls, 8000 probes. Deliverables: NOTEBOOK, REPORT (causal
  story, edit anatomy), FINAL edit + up to two pre-registered fallbacks.
- **Pre-registered comparison with Qwen (after 2B, before 2A).** A fixed script, no agent, measures the
  criteria written in `PLAN_crossmodel.md` §7 on the new model and on Qwen with identical code; the blind
  agent's mechanism is then compared with them.
- **2A — recipe transfer (last, capped ≤ 15 score calls).** Only now the Qwen recipe is given as a hint:
  a selective read in late layers → rank-1 write toward "No" (C3 = attention/mixer write + last-layer
  MLP read), placed by relative depth. Logged separately; reported as "recipe transferred: yes/no".

Per-model input = a model profile: number of layers and module names (from `model_info.json`); arms by
*relative* depth. The repo's `config.ARMS` stays Qwen-specific (`linear_attn`, L23–31); a blind copy
replaces it with one free arm.

## Stage 3 — evaluate (human runs it; the agent never sees the test half)

```
python -m harness.final_eval --suite repair --model <hf id> --edit FINAL_*.pt --arm ... --scales ...
```

On the 94 test patients: the seen tests on new patients, plus new wordings and the suite's held-out
tests (Qwen `repair`: US gallbladder, US appendix, CT kidney stone, amylase; Llama `repair_llama`:
US gallbladder, US appendix, amylase). Pass the model's own `--suite`. Each edit is compared with a random control
of the same size and position. Fallback doses are pre-registered in 2A/2B before any held-out scoring,
and every version is reported.

**Success** means all of these hold:
- held-out |asym| drops clearly, judged against the original model and the random control;
- the held-out control test stays where it was (|asym| no worse by more than 0.1);
- negatives that raised belief mostly disappear;
- drift ≤ 0.15, 40/40 general questions correct, ΔNLL ≤ 0.01.

## Per-model checklist

1. Download weights once (gated models: accept the license on HF first).
2. Stage 1 → read the GATE.
3. If PROCEED or DISCUSS-labs-biased: pick the model's suite by the per-model split rule (no need to
   re-ask), `make_splits --suite <suite>`, write a blind brief from `HANDOFF_blind_llama.md`,
   `blind_setup.py`, start a new server out dir (fresh budget) from the clean copy.
4. Stage 3 → write-up: the same figures as the Qwen page, plus "recipe transferred: yes/no, N calls".

## Change log

- 2026-11-21: framework fixed; stage 1 implemented (`harness/diagnose.py`). It matches `core.Model.yesno`
  exactly on Qwen3.5-0.8B; left-padded batching changes margins by ≤ 0.0006 (gemma-3-1b, CPU).
- 2026-11-21: stage 1 additions after MedGemma-1.5-4B (median no-result P(Yes) 0.1–0.5%, only 4–17
  mid-prior cases per test):
  (a) §1b prior-band check on all patients, to test whether a low prior compresses the moves. On MedGemma it does:
      the down move rises with the prior, rank correlation +0.81 to +0.94, while the up move does not. So the 0.02 lower bound stays,
      and low-prior cases may NOT be pooled.
  (b) a `source` column (neg under-used / pos over-used) plus a GATE note, because the repair direction depends on it;
  (c) `--from-margins` re-analysis without a GPU and `--pb-range` (non-default ranges are written to separate files).
- 2026-11-21: `--question considering` adds "The emergency physician is considering {d}." before the
  unchanged question, in B, P and N alike, so it cancels in up/down. Why: MedGemma 4B and 27B put the
  no-result P(Yes) at about 0.1%, which is the floor, so too few mid-prior cases remain and they are skewed
  to true cases. Validity check: rerun Qwen3.5-9B with orig and with considering. asym must be about unchanged.
- 2026-11-21 (KEY): §1c asym by FIXED prior bins (<1, 1-3, 3-10, 10-30, 30-70, >70%), all patients, imaging
  families vs the lab row. Runs: Qwen3.5-9B orig/could, MedGemma-4B orig/could, MedGemma-27B orig.
  Findings:
  (1) In every model, wording and test, asym falls as the prior rises: positive at low priors, about 0 near
      10-50%, negative above. Updates shrink on the side the belief is already near. Bayes says
      asym = 0 in every bin. The same bin under a different wording gives a similar value (Qwen orig vs could
      differ by about 0.3), so the prior, not the wording, sets most of asym.
  (2) Within one bin, imaging sits above lab by +1.71 (Qwen orig), +1.34 (Qwen could) and +0.99 (MG4 could).
      This test-type gap is what survives prior matching. The <1% bins are floor noise; the MG27 orig gap of
      −0.44 comes from its <1% lab cell.
  Consequence: the mid-prior mean asym (Qwen page, harness score) mixes the prior distribution with the effect,
  and Qwen's "negatives under-used" holds at its typical priors (2-10%), not at 30-70%. Stage-1 headline =
  §1c (slope over bins + imaging−lab gap in matched bins). Repair target to be decided.
  Note: with a non-orig `--question`, `--out` gets `q_<question>` appended, so pass the model directory.
- 2026-11-22: `--edit FILE [--scale s] [--control]` in stage 1: scores an edited model and compares it with the unedited
  run of the same model and wording (must exist in `<out>[/q_<question>]`). Output `<…>/edit_<stem>_x<s>[_control]/`, new §1d.
  Mid-prior cases and prior bins are chosen by the ORIGINAL model's belief, so selection does not move with the edit.
  First use: Qwen round-6 C3 under `--question could`. Question: is C3 a constant extra push on imaging negatives
  (high-prior bins overshoot), and does it stay imaging-selective when the prior distribution moves?
- 2026-11-22: `harness/prior_ladder.py` (controlled prior): each patient is scored at stated pre-test probabilities
  none/1/5/20/50/80% (one sentence after the history, same in B/P/N; 60 stratified patients by default). Report:
  A adoption (does the model take the stated level; AUC inside a level), B per level, C pooled curve binned by the
  model's actual prior, D shape (slope, balance point, floor, ceiling). Plan and reasons in `PLAN_crossmodel.md`.
- 2026-11-22: `--batch` default 8 → 1. Llama-3.1-8B (bf16, H100) batch 8 vs 1 changed the no-result margin by up to 0.25
  (gemma on CPU fp32 had been ≤ 0.0006). Batch 1 is exact and costs ~15 min per wording for an 8-9B model.
- 2026-11-24: new DATA, same pipeline: `--domain echo` (MIMIC-IV-Echo cohort). `extract_echo_mimic.sh` (BigQuery →
  `data/raw/echo_mimiciv_3_1/`) → `build_echo_cases.py` → `data/echo_pool.jsonl` (188 = 47 × heart failure / acute MI /
  pneumonia / PE by primary ICD; ED admissions with a TTE). 9 tests: imaging TTE wall motion (MI), TTE RV strain (PE),
  CTPA (PE), CXR (HF, pneumonia), lung US (HF); labs troponin T, NT-proBNP, D-dimer. Prompts/LR in `config.ECHO_*`.
  Filter differs from abd (agreed): HPI cut at the ED course; HPI clean of tests/diagnoses; PMH may name old
  diagnoses but no tests. No practice/test split (nothing is fitted here; the abd edit is applied as is).
  D-dimer LR is provisional (Cochrane gives ranges only); troponin cut-off to confirm.
- 2026-11-26 (agreed with the user; applies to every later model without re-asking):
  (a) repair target = relative pos/neg weight only (asym → 0 per test); overall over-reaction is a separate,
      set-aside problem (see Goal);
  (b) per-model split principle (practice = targets + one balanced control; held out = imaging target, lab
      target, balanced control), first instance `repair_llama` (lipase becomes a target, CT kidney stone the
      control). Why: the round-6 split encoded Qwen's pattern (lab = control); keeping the principle, not the
      test list, is what keeps the logic the same across models;
  (c) stage-2 order: blind 2B first (clean copy via `blind_setup.py`, brief `HANDOFF_blind_llama.md`,
      100 score / 8000 probes), then the pre-registered Qwen comparison (`PLAN_crossmodel.md` §7), then the
      hinted 2A recipe transfer. Why: a match found without the hint is evidence of shared structure.
  Context: `results/diagnose/CROSSMODEL_batch1.md` §12 (the original-question asym is mostly the low-prior
  side of the compression curve; mid-prior 2–95% band composition is nearly identical for Qwen and Llama,
  so the pooled harness metric is comparable between them).
