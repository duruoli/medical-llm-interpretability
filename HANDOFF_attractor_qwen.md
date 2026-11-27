# Handoff: repair the attractor (pull) after NEGATIVE results in Qwen3.5-9B with a targeted weight edit

You are the researcher. Your job is to make the model keep each patient's own starting belief when a NEGATIVE test
result arrives, with a small, targeted weight edit, and to explain from that edit WHY the model loses it.
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
raising belief). In this model p is about 0.8 after negative results and about 0.2 after positive ones: the
attractor sits on the negative side, which is also where negative evidence ends up counting less than positive.

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
(numeric, lexical)). The original model's report is in `<OUT>/scores/original.txt`. Read it first.

**Target:** `abs_pull` = median over target cells of |p|. Target cells = the **N side** (negative results) of every
cell. **Success line on practice:** `abs_pull` at most half the original.

**Constraints (all must hold; the report prints ok/FAIL for each):**
1. every cell's `move` within ±15% of the original (floor 0.5 for small moves), on BOTH sides. This blocks the
   trivial "fix" of making the model ignore the result (which also gives p = 0);
2. **the P side is left alone:** median p of the P cells changes by less than 0.1 (and its moves stay within ±15%,
   item 1). The two sides run through different components in this model (section 3); fix the negative one only;
3. no target test's median |p| worse than the original by more than 0.1;
4. no target cell with p < −0.2 (overshoot: the start amplified);
5. `baseline_drift` < 0.15 (no-result beliefs unchanged; the edit should act when a result is present);
6. `general_acc` (40 general yes/no questions) not lower; `history_nll` change < 0.01; `yesno_mass` not lower.

**Report only (do not optimise):** spread SD(F)/SD(B), SD of r, AUC (does the final belief separate patients with
the disease from the rest), number of negative results that RAISE belief, ln(up/down) by start tercile.

The human then evaluates on held-out data: new patients, new wordings, new tests, and a new clinical domain,
together with a random control edit (same parameters, support and norm). An edit that only fits the practice
cells will show there.

## 3. What earlier manual work found (all 188 patients, 7 tests, all wordings; your practice numbers will differ)

Architecture note: most layers of this model use a linear-attention mixer (`linear_attn`); a regular subset are
full softmax-attention layers (`self_attn`, e.g. layer 27; see `layer_types` in `<OUT>/model_info.json`). Below,
`aL` = the attention / mixer block of layer L, `mL` = its MLP; "head" in a linear-attention layer = a value head.

### Established (causal tests, held-out half of patients)
- Behaviour: pull N 0.79 / P 0.23; spread SD(F)/SD(B) 0.57 / 1.17. After negative results the patients are truly
  compressed; after positive results the pull is small and the spread is mostly re-ordering (large r).
  Not B measurement noise (instrument: another test's B), not disease mixture (same pull within one disease).
- NOT "the model stops reading the history": attention from the answer position to the history is almost unchanged
  when a result is present (only layer 19 drops, 0.12 → 0.07). NOT the final norm: pull before the final norm equals
  the final pull; RMS ratio ≈ 1.
- Direct attribution (slope with result − slope without, per block, sums to −pull). N side: erasure in layers 22–28,
  mainly `a27` heads 11, 12, 13, 15, `m27` (neurons 2411, 5179), `a23`, `m26` (4632), `m24`.
  **The P side uses a different set:** `m28`, `m24`, `m21`.
- **Add-back test (causal).** On the result prompt, set chosen units (MLP neuron = input of down_proj; head = its
  slice of the input of the output projection) at the answer position to "cell mean with result + this patient's
  no-result deviation from its mean". Mean push unchanged in every condition.
  - a27 top 4 heads: N pull 0.78 → 0.66; a27 + m27: 0.64; m27 alone: 0.76 (the heads matter, not the MLP).
  - a23, a27, m24, m26, m27 (top units): 0.78 → 0.54 (held-out half 0.61).
  - a22, a23, a27, a29, a30, m21, m24, m26, m27: top units 0.78 → 0.34 (held-out 0.37); **whole components
    0.78 → 0.20 (held-out 0.22)**; spread 0.60 → 0.84; P pull unchanged (0.24 → 0.23).
  - Random units, same number: 0.71–0.73 (held-out 0.78–0.81). Note the attention control is not tiny: 4 random of
    16 heads is a quarter of the layer.
  - **The two sides are separate:** adding back the P set (m21, m24, m28, ranked on P) lowers P pull 0.24 → 0.15
    (held-out 0.16) and leaves N unchanged (0.78 → 0.75).
- Not saturation: m26 4632, m27 5179 / 2411 have positive gate pre-activations and unchanged input RMS, but under
  a NEGATIVE result both gate and up lose their dependence on B (slope ≈ 0); under a POSITIVE result both keep it
  (value slopes 0.75 / 0.68 / −1.08). The patient signal is gone, or cancelled, in the residual stream they read,
  and only when the result is negative.
- Compensation: when the N set is added back, later attention starts erasing: a28 goes from ≈ 0 to −0.32 (−0.36
  with the wide set), a30 from −0.07 to −0.26; the late MLP that used to restore part (m31) restores less. a28 was
  never in an add-back set, which is one reason the pull does not reach 0. One component alone recovers only about
  1/3 of its attribution (a27: attribution −0.37, add-back lowers the pull by 0.11).

### Open (possibly wrong; part of your deliverables, section 6)
- Is the start information at the answer position CANCELLED (no longer linearly decodable) or MOVED to a direction
  the eraser units do not read?
- What signal from the negative result, through which upstream components, removes the patient dependence from
  these units' inputs?
- Are a28 / a30 actively pulling back toward the target landing, or reacting to the unnatural state the add-back
  creates? What do they read: the result tokens, or the current belief at the answer position?
- Why does the negative side have its own, stronger erasing set? What do a27 heads 11–15 read under a negative
  result, and what reaches them under a positive one?

### Method lessons
- Use add-back, not mean-patching, to test whether a unit erases (mean-patching raised the pull in earlier work).
- Attribution is not causation; judge every intervention by the pull at the OUTPUT.
- Select units on some patients, confirm on others.
- Later layers compensate: an edit at one place may be undone downstream.

### Two kinds of fix (decide by experiment)
- **Fix the read:** the start is still at the answer position but in a direction the eraser units no longer read →
  change those units' input weights.
- **Fix the transport:** the answer position no longer gathers it → change the attention / mixer that collects from
  the history, or block the negative-result signal that cancels it.
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
  `{"case": i, "test": "lipase", "pair": "numeric", "variant": "N"}`, or `{"text": "any user message"}`
  (chat template applied; add `"raw": true` to skip it). The answer position is the last token.
- **Activations:** `spec.json` = `{"prompts": [...], "modules": ["<prefix>.layers.24.mlp.down_proj", ...] (exact names in model_info.json),
  "positions": "last" | "all" | [ints], "kind": "output" | "input"}`. `kind: input` on `mlp.down_proj` gives the
  neuron values; on `mlp.gate_proj` / `mlp.up_proj` it gives the residual stream entering the MLP (after its norm).
- **Attribution:** `spec.json` = `{"prompts": [...], "units": ["m27", "a27"]}`. Saves `contrib` (prompts × blocks:
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
  | `patch_units` | `layer`, `kind` (`"mlp"` neurons or `"heads"`), `units` (list or `"all"`), `offsets` (`"auto"`, or `{"test\|pair\|side": [values per feature]}`, or a .pt path) | at the answer position: units = value on the SAME patient's B prompt + offset[cell]. `"auto"` = mean over practice mid cases of value(result) − value(B), computed by the server (charged as probe prompts, about 1,450 for this split): the add-back of section 3. Works inside `score`. Prompts with no B source (B prompts, general questions, `text` prompts) are untouched |
  | `ablate_neurons` | `layer`, `neurons`, `scale` | MLP neurons scaled by (1 − scale) at the input of `down_proj` |
  | `ablate_heads` | `layer`, `heads`, `scale` | heads scaled by (1 − scale) at the input of `o_proj` (full-attention layers only; use `patch_units` or `acts` for linear-attention value heads) |
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
| E4 | Why the negative side has its own, stronger erasing set (what a27 heads 11–15 read under N vs P) | interventions on what those heads read, with the P side as the comparison |

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
