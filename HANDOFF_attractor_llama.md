# Handoff: repair the attractor (pull) in Llama-3.1-8B-Instruct with a targeted weight edit

You are the researcher. Your job is to make the model keep each patient's own starting belief when a test result
arrives, with a small, targeted weight edit, and to explain from that edit WHY the model loses it.
There is no training and no gradient descent: you analyse weights and activations, write a delta, and test it.
The human defined the task, built this harness, and grades the result on held-out data you never see.

Unlike an open-ended search, the location of the problem is already largely known from earlier manual work
(section 3). What is not known is the cause, and therefore how to fix it inside the model. You are given those
manual results as information; treat the "established" part as reliable and the "open" part as possibly wrong.

Method reference: "weight-space programming" (https://rikiyat.github.io/blog/weight-space-programming/).
- Diagnose before editing; an edit should follow from a mechanism, and a dose-response and a random control
  confirm it.
- An edit can remove an obstacle or reconnect information the model already has; it cannot add a missing ability.

## 1. The flaw: the attractor

Prompt: a real emergency-department history (MIMIC), one test result, then "Does this patient have <disease>?
Answer Yes or No only." Belief = log-odds m = logit(Yes) − logit(No) at the answer position.
For each patient and test there are three prompts: B (result pending), P (positive result), N (negative result).

Fix one **cell** = test × wording × side (P or N). Over the cell's mid-prior patients (original no-result P(Yes)
in 0.02–0.95), with start B_i, final F_i and push d_i = F_i − B_i, a least-squares line splits the push into:

```
d_i = move + (−p) × (B_i − mean B) + r_i
```

| term | name | meaning | role here |
|---|---|---|---|
| `move` | mean push | how far the result moves belief on average (evidence strength) | **must not change** |
| `p` | **pull** | how much the push depends on the start; p = 1 − slope of F on B | **target: lower \|p\|** |
| `r_i` | remainder | patient-specific push not explained by the start | report only |

Bayes: the same result gives every patient the same push, so p = 0 and two patients who started 2 apart end
2 apart. With p > 0 the model partly forgets the start: on average, landings are pulled toward a fixed point
(an "attractor"), and patients whose start is beyond it can even move the wrong way (a negative result
raising belief). In this model p is about 0.55–0.6 on both sides.

Notes:
- p large does not mean patients land in one spot. That also needs r small. The spread SD(F)/SD(B) is reported
  but not targeted.
- p = 0 is an idealisation: it assumes a test is equally informative for every patient. This is why the target
  is "halve it", not "zero it".
- This definition is fixed. Do not redefine the target. If, after an edit, the straight-line description clearly
  stops fitting (e.g. the push becomes curved in B), report it.

**Why a single-pass fix must be possible:** the prompt order is history → result → question. The model reads left
to right, so at every history token its internal state is IDENTICAL to the no-result prompt. The patient
information is always present in the prompt's history positions. It is lost only in how the positions after the
result assemble the answer.

## 2. Score

`score` evaluates the current weights (and any active interventions) on the practice split
(`data/harness/practice.jsonl`, 94 patients; tests `ct_sigmoid`, `ct_appendix` (4 wordings each), `lipase`
(numeric, lexical), `ct_kidney_stone` (impression, findings)). The original model's report is in
`<OUT>/scores/original.txt`. Read it first.

**Target:** `abs_pull` = median over target cells of |p|. Target cells = both sides of every cell except the
`lipase/numeric` cells (a separate, known failure of this model: it misreads numeric lab values; reported only).
**Success line on practice:** `abs_pull` at most half the original.

**Constraints (all must hold; the report prints ok/FAIL for each):**
1. every cell's `move` within ±15% of the original (floor 0.5 for small moves). This blocks the trivial "fix" of
   making the model ignore the result (which also gives p = 0);
2. no target test's median |p| worse than the original by more than 0.1;
3. no target cell with p < −0.2 (overshoot: the start amplified);
4. `baseline_drift` < 0.15 (no-result beliefs unchanged; the edit should act when a result is present);
5. `general_acc` (40 general yes/no questions) not lower; `history_nll` change < 0.01; `yesno_mass` not lower.

**Report only (do not optimise):** spread SD(F)/SD(B), SD of r, AUC (does the final belief separate patients with
the disease from the rest), number of negative results that RAISE belief, ln(up/down) by start tercile.

The human then evaluates on held-out data: new patients, new wordings, new tests, and a new clinical domain,
together with a random control edit (same parameters, support and norm). An edit that only fits the practice
cells will show there.

## 3. What earlier manual work found (all 188 patients, 7 tests, all wordings; your practice numbers will differ)

### Established (causal tests, held-out half of patients)
- Behaviour: pull N 0.53 / P 0.66; spread 0.87 / 0.84. Not B measurement noise (instrument: another test's B),
  not disease mixture (same pull within one disease).
- NOT "the model stops reading the history": attention from the answer position to the history is unchanged when
  a result is present. NOT the final norm: pull before the final norm equals the final pull; RMS ratio ≈ 1.
- Direct attribution (slope with result − slope without, per block, sums to −pull): erasure builds up over layers
  16–28. Biggest erasers, the same on both sides: `m24` (−0.37 N / −0.41 P; almost all neuron 14331), `m29`
  (neurons 2039, 12580), `m21` (13111), `m28` (10558); also m18, m19, m26, a24. Late blocks (m30, m31) restore part.
- **Add-back test (causal).** On the result prompt, set chosen units (MLP neuron = input of down_proj) at the answer
  position to "cell mean with result + this patient's no-result deviation from its mean". Mean push unchanged in
  every condition; random units of the same number do nothing.
  - m24 + m29, top 32 units each: pull 0.51 → 0.31 (N), 0.63 → 0.37 (P).
  - m18, m19, m21, a24, m24, m26, m28, m29, top units (228 in total): **0.51 → 0.04 (N), 0.63 → 0.12 (P)**;
    on the held-out half of patients (units ranked on the other half) 0.07 / 0.20; spread back to ≈ 1.0.
  - Replacing the same units by the patient MEAN raises the pull (0.51 → 0.61): these units carry the start without
    a result and lose it with one.
- Not saturation: for 14331, 10558, 13111 the gate pre-activation stays positive (0% of patients below 0) and the
  residual RMS entering the MLP is unchanged, but BOTH gate and up lose their dependence on B when a result is
  present (14331: slope of value on B 0.61 without a result, ≈ 0.2 with one; the value is pushed to −8.0 (N) /
  −1.3 (P)). The patient signal is gone, or cancelled, in the residual stream these neurons read.
  Exception: m29 12580 on the P side (gate below 0 for 28% of patients), and m29 2039 is nearly off under N.
- Compensation: when the difference is put back early, later blocks erase more (adding back m21 alone: blocks after
  it go from −0.27 to −0.47 on P). With the wide add-back, m31 becomes the largest eraser on P (−0.21). The landing
  is maintained by several blocks; one block alone recovers only about 1/3 of its attribution.

### Open (possibly wrong; part of your deliverables, section 6)
- Is the start information at the answer position CANCELLED (no longer linearly decodable) or MOVED to a direction
  these units do not read?
- What signal from the result, through which upstream components, removes the patient dependence from these
  neurons' inputs?
- Are the compensating late blocks actively pulling back toward the target landing, or reacting to the unnatural
  state the add-back creates?
- Why does whole-component add-back do worse on P (0.33) than top-units-only (0.12)?

### Method lessons
- Use add-back, not mean-patching, to test whether a unit erases.
- Attribution is not causation; judge every intervention by the pull at the OUTPUT.
- Select units on some patients, confirm on others.
- Later layers compensate: an edit at one place may be undone downstream.

### Two kinds of fix (decide by experiment)
- **Fix the read:** the start is still at the answer position but in a direction the eraser units no longer read →
  change those units' input weights.
- **Fix the transport:** the answer position no longer gathers it → change the attention that collects from the
  history, or block the result-driven signal that cancels it.
`patch_units` with `"offsets": "auto"` (section 5) reproduces the add-back inside `score`: it is the upper
reference a weight edit at those units can aim for.

## 4. Rules

1. **Work only inside this copy and `<OUT>`.** Other directories on this machine hold unrelated earlier work;
   reading them invalidates this round. Do not search the filesystem outside these two directories.
2. Do not modify `harness/`. Do not restart the server with budget overrides. Do not touch `<OUT>/budget.json`
   or `<OUT>/log.jsonl`.
3. Edits may not touch embeddings, `lm_head` or the final norm (the server refuses). Avoid global scaling such as
   multiplying a whole layer. At most 5,000,000 changed weights; smaller is better.
4. **Budget:** 100 `score` calls and 20,000 probe prompts in total across `logits`, `acts`, `attrib` and automatic
   offsets (at most 256 prompts per call). Reading weights and CPU-side analysis are free. Spend probes on
   diagnosis, score calls on candidate edits.
5. All forward passes go through the server. Do not load the model yourself.
6. Privacy: prompts contain MIMIC patient text (PhysioNet DUA). Never send them to an external service, never
   commit them, keep work under `<OUT>`.

## 5. Tools

`<OUT>` = the server's output directory (the human tells you; pass `--out <OUT>` to every client call).
Parameter and module names and the model config are in `<OUT>/model_info.json`.

```
python -m harness.client --out <OUT> status                    # budget left, active edit
python -m harness.client --out <OUT> score                     # practice evaluation        [1 score call]
python -m harness.client --out <OUT> apply EDIT.pt [--scale s] [--note "..."]   # W = W_orig + s*delta (free)
python -m harness.client --out <OUT> reset                     # original weights (free)
python -m harness.client --out <OUT> weights NAME [NAME ...]   # ORIGINAL weights -> <OUT>/weights/*.pt (free)
python -m harness.client --out <OUT> logits prompts.json [--top-k 5]   # margins + top tokens   [probe prompts]
python -m harness.client --out <OUT> acts spec.json            # activations -> <OUT>/acts/*.pt  [probe prompts]
python -m harness.client --out <OUT> attrib spec.json          # per-block contributions -> .pt  [probe prompts]
python -m harness.client --out <OUT> intervene spec.json       # diagnostic hooks, no weight change
python -m harness.client --out <OUT> clear                     # remove all hooks (free)
```
From Python: `from harness.client import call; call("logits", out="<OUT>", prompts=[...])`.

- **Prompts:** `{"case": i, "test": "ct_sigmoid", "variant": "B"}`,
  `{"case": i, "test": "lipase", "pair": "lexical", "variant": "N"}`, or `{"text": "any user message"}`
  (chat template applied; add `"raw": true` to skip it). The answer position is the last token.
- **Activations:** `spec.json` = `{"prompts": [...], "modules": ["<prefix>.layers.24.mlp.down_proj", ...] (exact names in model_info.json),
  "positions": "last" | "all" | [ints], "kind": "output" | "input"}`. `kind: input` on `mlp.down_proj` gives the
  neuron values; on `mlp.gate_proj` / `mlp.up_proj` it gives the residual stream entering the MLP (after its norm).
- **Attribution:** `spec.json` = `{"prompts": [...], "units": ["m24", "a24"]}`. Saves `contrib` (prompts × blocks:
  embed, a0, m0, a1, ...; sums to the margin, `decomp_err` checks it, final norm linearised per prompt) and per-unit
  contributions of the listed blocks (MLP: per neuron; attention: per head). Active edits and interventions apply.
- **Edit file:** `torch.save({"deltas": {param_name: delta_tensor}}, path)`, same shape as the weight, float32 on CPU.
  The server keeps a hashed copy of every applied edit. Deltas far below bf16 resolution are partly lost.
- **Interventions** (`intervene`; stay active for every later score / logits / acts / attrib call until `clear`):

  | type | fields | effect |
  |---|---|---|
  | `add` | `layer` (or `module`), `vector` (list, or .pt path + optional `key`), `alpha`, `positions` | `h[pos] += alpha * v` |
  | `project` | `layer`/`module`, `vector`, `strength` (default 1), `center` (default 0), `positions` | `h[pos] -= strength * ((h·v̂) − center) v̂` |
  | `patch` | `layer`/`module`, `positions` (`"last"` or negative ints) | `h[pos]` = same module and position (from the end) in the prompt's `patch_from` source; every probe prompt then needs `"patch_from"`; not usable with `score` |
  | `patch_units` | `layer`, `kind` (`"mlp"` neurons or `"heads"`), `units` (list or `"all"`), `offsets` (`"auto"`, or `{"test\|pair\|side": [values per feature]}`, or a .pt path) | at the answer position: units = value on the SAME patient's B prompt + offset[cell]. `"auto"` = mean over practice mid cases of value(result) − value(B), computed by the server (charged as probe prompts, about 1,300 for this split): the add-back of section 3. Works inside `score`. Prompts with no B source (B prompts, general questions, `text` prompts) are untouched |
  | `ablate_neurons` | `layer`, `neurons`, `scale` | MLP neurons scaled by (1 − scale) at the input of `down_proj` |
  | `ablate_heads` | `layer`, `heads`, `scale` | heads scaled by (1 − scale) at the input of `o_proj` |
  | `freeze` (top-level list) | module names | these modules output their value on the ORIGINAL model for the same prompt (tests compensation by later layers); each prompt runs twice |

  `layer: L` for add / project / patch hooks the output of decoder layer L (0-based), i.e. the residual stream after it.

## 6. Deliverables (in `<OUT>/work/`)

**A. The edit (scored by the harness).**
- `FINAL.pt`, plus up to two fallbacks (other doses or variants), named and listed in REPORT **before** the
  held-out evaluation.

**B. The explanation (judged by the human).** Answer as many as you can, in this priority order; E0 and E1 are
required. Every answer must rest on a causal test; label each claim "verified" (causal) or "observed"
(correlational). "Could not decide" is acceptable if you show what you tried and why it does not discriminate.
Do not change the target or the definition to answer these.

| # | question | what an answer needs |
|---|---|---|
| E0 | Is your fix a **read** fix or a **transport** fix, i.e. is the cause in how the eraser units read, or upstream in what reaches the answer position? | placement comparison (similar budget at both kinds of location) and the layer-by-layer decodability of B with your edit |
| E1 | At the answer position with a result, is the start **cancelled** or **moved**? | layer-by-layer linear readout of B (practice patients, cross-validated), result vs no result; relation of the readout direction to the eraser units' input weights |
| E3 | Are the compensating late blocks actively pulling back, or reacting to an unnatural state? | without any intervention: do they erase more for patients with a large natural deviation r? and do they compensate your edit? |
| E2 | Which result-driven signal, through which upstream components, removes the patient dependence from the eraser units' inputs? | interventions that remove or replace a candidate signal and restore the units' slope on B |
| E4 | Why whole-component add-back is worse on P than top units only | the units that act against, confirmed causally |

**C. Files.**
- `NOTEBOOK.md`, kept as you go: every hypothesis, the evidence for and against, every edit (file, scale, score
  number), what you concluded. Record failed and contrary results too.
- `REPORT.md`:
  - plain-language summary first: what is wrong inside the model, where, and what the edit does;
  - answers to E0–E4, each labelled verified / observed;
  - the edit's anatomy: which parameters, what it reads, what it writes, whether later layers compensate it;
  - the practice report of the final edit, its dose-response at 2–3 scales and your own random control
    (same shape and norm, should do nothing comparable);
  - what is NOT verified.

Stop when the budget is spent, or earlier if you have a clean result.
