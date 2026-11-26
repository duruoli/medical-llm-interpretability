# Handoff: find and repair under-weighted negative evidence in Llama-3.1-8B-Instruct

You are the researcher. Your job is to find, inside the model, why negative test results move its belief
less than positive ones (relative to medical evidence), and to repair that with a small, targeted weight edit.
There is no training and no gradient descent: you analyse weights and activations, write a delta, and test it.
The human defined the task, built this harness, and grades the result on a held-out set you never see.

This is an independent investigation of this model. Form your own hypotheses from what you measure.

Method reference: "weight-space programming" (https://rikiyat.github.io/blog/weight-space-programming/).
There, an LLM fixed a small model's repetition loops by finding a direction written by a few MLP layers and
editing a few output columns. Two lessons apply:
- Diagnose before editing. The edit should follow from a mechanism; then a dose-response and a random
  control confirm it.
- An edit can remove an obstacle or connect information the model already has; it cannot add a missing
  ability. Prefer hypotheses like "the model represents the negative result but under-uses it" over "the
  model lacks medical knowledge".

## 1. The flaw

Prompt: a real emergency-department history (MIMIC), one test result, then "Does this patient have
<disease>? Answer Yes or No only." Belief = log-odds m = logit(Yes) − logit(No) at the answer position.
For each patient and test there are three prompts: B (result pending), P (positive result), N (negative result).
- up = m_P − m_B: how far a positive result raises belief.
- down = m_B − m_N: how far a negative result lowers it.

An ideal Bayesian moves up by ln(LR+) and down by −ln(LR−), whatever the prior. The literature sensitivity and
specificity of each practice test are in `harness/config.py` (`LIT`).

The flaw: asym = ln(up/down) − ln(lit_up/lit_down), on mid-prior cases (original baseline P(Yes) in
0.02–0.95), is above 0 on most practice tests, i.e. negatives count too little relative to positives.
**The goal is to bring asym toward 0 on each test.**

Practice tests (`data/harness/practice.jsonl`, 94 patients):

| test | question | role |
|---|---|---|
| `ct_sigmoid` (4 wordings) | acute diverticulitis | repair target |
| `ct_appendix` (4 wordings) | acute appendicitis | repair target |
| `lipase` (numeric, lexical) | acute pancreatitis | repair target |
| `ct_kidney_stone` (impression, findings) | kidney or ureteral stone | **control: already balanced, do not break it** |

The original model's numbers are in `<OUT>/scores/original.txt`. Read them first.

## 2. Score

`score` evaluates the current weights on practice. Report:
- **`abs_asym` (primary):** mean over cells of |asym|. Over- and under-correction both count. Also given per test.
- `asym_mean` (signed), `move_err` (|up − lit_up| + |down − lit_down|), `neg_raises` (cases where a negative
  RAISED belief).
- **Guards:** `baseline_drift` (change in the no-result belief), `yesno_mass`, `general_acc` and `general_shift`
  (40 general yes/no questions; catches "just say No more"), `history_nll` (language modelling on clinical text).
- **Cells** per test and wording, and the worst individual negative moves (`case` = index into practice.jsonl).

**Success on practice:**
- `abs_asym` clearly below the original;
- `ct_kidney_stone` |asym| no worse than the original by more than 0.1;
- `neg_raises` falls;
- `baseline_drift` < 0.15, `general_acc` unchanged, history NLL change < 0.01, `move_err` not worse by more than 0.1;
- the fix should come from negatives counting more, not from positives counting less: watch the `up` column.

**Not the target.** Do not try to fix the model's overall overconfidence (e.g. positives moving 5 log-odds where
the literature says 2). Just don't make it worse.

The human then runs a held-out evaluation: new patients, new wordings, and new tests (imaging and lab, including
tests that are already balanced). It includes a random control with the same parameters, support and norm as
your edit. An edit that only fits the practice wordings or tests will show there.

## 3. Rules

1. **Work only inside this copy and `<OUT>`.** Other directories on this cluster hold unrelated earlier work;
   reading them invalidates this round. Do not search the filesystem outside these two directories.
2. Do not modify `harness/`. Do not restart the server with budget overrides. Do not touch `<OUT>/budget.json`
   or `<OUT>/log.jsonl`.
3. Edits may not touch embeddings, `lm_head` or the final norm (the server refuses). Avoid global scaling such as
   multiplying a whole layer. At most 5,000,000 changed weights; smaller is better.
4. **Budget:** 100 `score` calls and 8000 probe prompts in total across `logits` and `acts` (at most 256 per call).
   Reading weights and CPU-side analysis are free. Spend probes on diagnosis, score calls on candidate edits.
5. All forward passes go through the server. Do not load the model yourself.
6. Privacy: prompts contain MIMIC patient text (PhysioNet DUA). Never send them to an external service, never
   commit them, keep work under `<OUT>`.

## 4. Tools

`<OUT>` = the server's output directory (the human tells you; pass it as `--out <OUT>` to every client call).
Parameter and module names, and the model config, are in `<OUT>/model_info.json`.

```
python -m harness.client --out <OUT> status                    # budget left, active edit
python -m harness.client --out <OUT> score                     # practice evaluation        [1 score call]
python -m harness.client --out <OUT> apply EDIT.pt [--scale s] [--note "..."]   # W = W_orig + s*delta (free)
python -m harness.client --out <OUT> reset                     # original weights (free)
python -m harness.client --out <OUT> weights NAME [NAME ...]   # ORIGINAL weights -> <OUT>/weights/*.pt (free)
python -m harness.client --out <OUT> logits prompts.json [--top-k 5]   # margins + top tokens   [probe prompts]
python -m harness.client --out <OUT> acts spec.json            # activations -> <OUT>/acts/*.pt  [probe prompts]
python -m harness.client --out <OUT> intervene spec.json       # diagnostic hooks, no weight change (free)
python -m harness.client --out <OUT> clear                     # remove all hooks (free)
```
From Python: `from harness.client import call; call("logits", out="<OUT>", prompts=[...])`.

- **Prompts:** `{"case": i, "test": "ct_sigmoid", "variant": "B"}`,
  `{"case": i, "test": "lipase", "pair": "numeric", "variant": "N"}`, or `{"text": "any user message"}`
  (chat template applied; add `"raw": true` to skip it).
- **Activations:** `spec.json` = `{"prompts": [...], "modules": ["model.layers.12.mlp", ...],
  "positions": "last" | "all" | [ints], "kind": "output" | "input"}`. The saved file holds `acts[module]`
  (positions × hidden per prompt), `tokens` and `yes_minus_no`. The answer position is the last token.
- **Edit file:** `torch.save({"deltas": {param_name: delta_tensor}}, path)`, same shape as the weight, float32 on CPU.
  The server keeps a hashed copy of every applied edit and logs it. Deltas far below bf16 resolution are partly lost.
- **Interventions** (`intervene`; stay active for every later score / logits / acts call until `clear`):

  | type | fields | effect |
  |---|---|---|
  | `add` | `layer` (or `module`), `vector` (list, or .pt path + optional `key`), `alpha`, `positions` | `h[pos] += alpha * v` |
  | `project` | `layer`/`module`, `vector`, `strength` (default 1), `center` (default 0), `positions` | `h[pos] -= strength * ((h·v̂) − center) v̂` |
  | `patch` | `layer`/`module`, `positions` (`"last"` or negative ints) | `h[pos]` = same module and position (from the end) in the prompt's `patch_from` source; every probe prompt then needs `"patch_from"`; not usable with `score` |
  | `ablate_neurons` | `layer`, `neurons`, `scale` | MLP neurons scaled by (1 − scale) at the input of `down_proj` |
  | `ablate_heads` | `layer`, `heads`, `scale` | heads scaled by (1 − scale) at the input of `o_proj` |
  | `freeze` (top-level list) | module names | these modules output their value on the ORIGINAL model for the same prompt (tests compensation by later layers); each prompt runs twice |

  `layer: L` hooks the output of decoder layer L (0-based), i.e. the residual stream after layer L.

## 5. Suggested workflow

1. Read `original.txt`. Use `logits` on a few cases to see B, P and N for one patient across tests and wordings.
2. Diagnose: where does a negative result fail to reach the answer, compared with a positive one and with the
   control test? Measure before you hypothesise; then form ONE mechanistic hypothesis and test it causally
   (interventions) before editing.
3. Edit: the smallest change that follows from the hypothesis.
4. Dose-response at 2–3 scales (a real mechanism gives a monotone, graded response).
5. Your own control: a random edit with the same shape and norm should do nothing comparable.
6. Iterate. Keep only edits that lower `abs_asym` and keep the control, the `up` column and the guards intact.

## 6. Deliverables (in `<OUT>/work/`)

- `NOTEBOOK.md`, kept as you go: every hypothesis, the evidence for and against, every edit (file, scale, score
  number), what you concluded.
- `FINAL.pt`: the best edit, plus up to two fallback doses or variants, named and listed **before** the held-out
  evaluation.
- `REPORT.md`:
  - **plain-language summary first:** what is wrong inside the model, where, and what the edit does;
  - the causal story, with the evidence for each link, keeping causally tested claims apart from observations;
  - the edit's anatomy: which parameters, what it reads, what it writes, and whether later layers compensate it;
  - the practice report of the final edit and its dose-response and random control;
  - what is NOT verified.

Stop when the budget is spent, or earlier if you have a clean result.
