# Pipeline: diagnose → repair → evaluate (any model)

Goal: take a new model X and answer, with the same yardstick every time,
(1) does X under-use negative imaging results relative to the literature?
(2) can a small weight edit repair it without breaking anything? (3) does the repair hold on
patients, wordings and tests the edit never saw?

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

## Stage 1 — diagnose (behaviour only, no edits)

```
python -m harness.diagnose --model <hf id> --out /scratch/dlf8982/diagnose/<model>   # → REPORT.md; scp back to results/diagnose/<model>/
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

Same server and rules as round 6 (`harness/server.py`, `HANDOFF_round6.md`): the agent sees only the
practice half (CT diverticulitis, CT appendicitis, lipase as the do-not-break control) and has a fixed budget.
Two phases, logged separately so the result says whether the *recipe* transferred:

- **2A — recipe transfer (capped, e.g. ≤ 15 score calls).** Only the Qwen-validated recipe: a ridge-fitted
  *selective read* in late layers (fires on CT negatives, not on lab negatives) → rank-1 write toward "No"
  (C3 = mixer/attention write + last-layer MLP read). The only free choices are the layer and the dose.
- **2B — free exploration.** Only if 2A fails a guard or leaves |asym| > ~0.4, or if stage 1 showed a new
  pattern.

Per-model input = a **model profile**: number of layers, module names, and edit arms placed by
*relative* depth instead of Qwen layer numbers. This is not written yet. Current `config.ARMS` is
Qwen-specific: `linear_attn` and L23-31.

## Stage 3 — evaluate (human runs it; the agent never sees the test half)

```
python -m harness.final_eval --suite repair --model <hf id> --edit FINAL_*.pt --arm ... --scales ...
```

On the 94 test patients: the seen tests on new patients, plus new wordings and new tests
(US gallbladder, US appendix, CT kidney stone, amylase). Each edit is compared with a random control
of the same size and position. Fallback doses are pre-registered in 2A/2B before any held-out scoring,
and every version is reported.

**Success** means all of these hold:
- held-out |asym| drops clearly, judged against the original model and the random control;
- the labs stay where they were;
- negatives that raised belief mostly disappear;
- drift ≤ 0.15, 40/40 general questions correct, ΔNLL ≤ 0.01.

## Per-model checklist

1. Download weights once (gated models: accept the license on HF first).
2. Stage 1 → read the GATE.
3. If PROCEED: write the model profile, start a new server out dir (fresh budget), write a round brief
   from the 2A/2B template.
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
