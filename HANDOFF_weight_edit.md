# Handoff: weight edit for the positive-evidence bias in Qwen3.5-9B

You are the researcher. Your job is to find, in Qwen3.5-9B's weights, the cause of a
specific flaw and remove it with a small, targeted weight edit. There is no training
and no gradient descent: you analyse weights and activations, write a delta, and test it.
The human defined the task, built this harness, and grades the result on a held-out
test set that you never see.

Method reference: "weight-space programming" (https://rikiyat.github.io/blog/weight-space-programming/).
There, an LLM fixed Qwen3.5-0.8B's repetition loops by finding a direction written by a
few MLP layers and subtracting it from 64 output columns of two layers (131k weights).
Two lessons from that work apply here:
- Diagnose before editing. The edit came from a mechanism, then dose-response and a
  random-direction control confirmed it.
- An edit can remove an obstacle but cannot add a missing ability. Prefer hypotheses of
  the form "something suppresses / under-weights negative evidence" over "the model
  lacks medical knowledge".

## 1. The flaw

Prompt: a real emergency-department history (MIMIC), one test result, then
"Does this patient have <disease>? Answer Yes or No only." Belief = log-odds
m = logit(Yes) - logit(No) at the answer position.
For each patient and test there are three prompts: B (result pending), P (positive
result) and N (negative result).
- up = m_P - m_B: how far a positive result raises belief.
- down = m_B - m_N: how far a negative result lowers it.

An ideal Bayesian moves by ln(LR+) up and -ln(LR-) down, whatever the prior.
Literature values (sens, spec):
- CT for diverticulitis: 0.94, 0.99, which gives up 4.54 and down 2.80.
- CT for appendicitis: 0.95, 0.94, which gives up 2.76 and down 2.93.

The 9B model's moves on mid-prior cases (baseline P(Yes) between 0.02 and 0.95), measured
2026-11-12 on 188 patients:

| test / wording | up | down | up/down | literature up/down |
|---|---|---|---|---|
| sigmoid CT, "Acute sigmoid diverticulitis" vs "No evidence of diverticulitis" | 5.06 | 2.15 | 2.36 | 1.13-1.62 |
| sigmoid CT, full impression ("Sigmoid diverticulosis without evidence of acute diverticulitis") | 5.54 | 1.16 | 4.78 | |
| sigmoid CT, findings ("The sigmoid colon and rectum are normal in appearance") | 4.97 | 0.25 | 19.5 | |
| appendix CT, all wordings | | | 1.6-2.8 | 0.94 |

Bootstrap 95% confidence intervals exclude the literature ratio. So the model
over-weights positive evidence about 2x relative to negative evidence. The effect is
worst when the negative sentence names the organ or a near-homonym of the disease
("sigmoid", "diverticulosis"): in 48/103 cases the "findings" negative RAISES belief.

Earlier mechanistic results on this model and task (activation patching, different prompts):
- Information about the result leaves the result tokens at layers about 5-11.
- Negation is carried onto the following word (the disease name) at about layers 6-10.
- Attention heads 19.13, 19.14 and 27.10 prefer "no" (negation); head 23.12 is lexical.
- "no" and "normal" receive about half the attention that positive words get.

These are leads, not facts about the yes/no prompts. Check them before relying on them.

Architecture: 32 layers, hybrid. Every 4th layer (3, 7, ..., 31) is full attention;
the rest are Gated DeltaNet (linear attention). Parameter and module names are listed
in `<OUT>/model_info.json`, and the config's `layer_types` gives the layer type.

## 2. Goal and how you are scored

The `score` command evaluates the current weights on the practice set. Its report has
three parts.

**Headline**
- `asym_mean`: the mean over cells of ln(up/down) - ln(lit_up/lit_down). The original
  model's value is in `<OUT>/scores/original.txt` (positive = positive bias). **The goal is
  to bring this toward 0.**
- `move_err`: the mean |up-lit_up| + |down-lit_down|. This is secondary, but an edit
  that fixes the ratio by killing positive moves is NOT a fix. We want negatives to
  count more, not positives to count less. Watch the `up` column.
- `neg_raises`: the number of cases where a negative result raised belief.

**Guards** (an edit that breaks these is rejected)
- `baseline_drift`: the change in the no-result belief.
- `yesno_mass`: the model still answers Yes/No.
- `general_acc` and `general_shift`: 40 general yes/no questions. These catch
  "just say No more".
- `history_nll`: plain language modelling on clinical text.

**Cells**: per test and wording, before and after the edit, plus the worst individual
negative moves (`case` = index into practice.jsonl) so you can read the failures.

The human then runs `harness/final_eval.py` on the test split. It has new patients,
two wordings you never saw, and two tests you never saw: gallbladder ultrasound and
serum lipase. It also runs a random-content control with the same support and norm as
your edit. An edit that only fits the practice wordings will show up there.

## 3. Rules

1. Do not read `data/harness_test/` or anything that `harness/final_eval.py` writes.
   Do not run final_eval.
2. Do not modify `harness/*.py` or `harness/config.py`. Do not restart the server with
   budget overrides. Do not touch `<OUT>/budget.json` or `<OUT>/log.jsonl`.
3. Edits may not touch embeddings, `lm_head`, the final norm or the vision tower (the
   server refuses). Avoid global scaling such as multiplying a whole layer.
   At most 5,000,000 changed weights. Smaller is better; the blog used 131k.
4. Budget:
   - 60 `score` calls (about 2 min each on 9B).
   - 4000 probe prompts in total across `logits` and `acts`, at most 256 per call.
   - Reading weights is free.

   Spend probes on diagnosis and score calls on candidate edits.
5. All model forward passes go through the server. CPU-side analysis of saved weights
   and activations in your own Python is free and unlimited. Do not load the model
   yourself.

## 4. Tools

The human starts the server once (tmux window 1, repo root):

```
conda activate /projects/p31777/envs/interp
python -m harness.server --model Qwen/Qwen3.5-9B --out /scratch/dlf8982/harness
```

It scores the original model (free; saved in `<OUT>/scores/original.txt`), writes
`model_info.json`, then prints `ready`. `<OUT>` = `/scratch/dlf8982/harness`.

```
python -m harness.client status                       # budget left, active edit
python -m harness.client score                        # practice evaluation   [1 score call]
python -m harness.client apply EDIT.pt [--scale s] [--note "..."]   # W = W_orig + s*delta (free)
python -m harness.client reset                        # original weights (free)
python -m harness.client weights NAME [NAME ...]      # ORIGINAL weights -> <OUT>/weights/*.pt (free)
python -m harness.client logits prompts.json [--top-k 5]   # margins + top tokens   [probe prompts]
python -m harness.client acts spec.json               # activations -> <OUT>/acts/*.pt  [probe prompts]
```

From Python: `from harness.client import call; call("logits", prompts=[...])`.

- **Prompt references**:
  - `{"case": i, "test": "ct_sigmoid", "variant": "B"}`
  - `{"case": i, "test": "ct_appendix", "pair": "findings", "variant": "N"}`
  - `{"text": "any user message"}` (chat template applied; add `"raw": true` to skip it)

  `case` indexes `data/harness/practice.jsonl` (94 patients). Pairs are `lexical`,
  `impression`, `full_impression` and `findings`. You may read practice.jsonl freely.
- **Activations**: `spec.json` = `{"prompts": [...], "modules": ["model.layers.12.mlp", ...],
  "positions": "last" | "all" | [ints], "kind": "output" | "input"}`. The saved file holds
  `acts[module]` = list of tensors (positions x hidden), `tokens` and `yes_minus_no`.
  The answer position is the last token. For attention patterns, capture the inputs of
  `self_attn` or the q/k projections and recompute.
- **Edit file**: `torch.save({"deltas": {param_name: delta_tensor}}, path)`, with the
  same shape as the weight and float32 on CPU. The server keeps a hashed copy of every
  applied edit in `<OUT>/edits/` and logs it. Deltas much smaller than bf16 resolution
  are partly lost; `apply` warns you.
- **Working files**: put your scripts, edits and notes under `/scratch/dlf8982/harness/work/`.

## 5. Suggested workflow

1. Read `<OUT>/scores/original.txt`. Use `logits` on a few cases to see B, P and N for
   one patient across wordings.
2. Diagnose. Where does a negative result fail to reach the answer?
   - Answer-position activations for B/P/N.
   - The difference direction P-B versus B-N per layer.
   - Which layers or components write along or against the "No" direction.
   - Whether the negation information dies at the result tokens, during handoff, or
     is overridden late.

   Compare a failing wording (findings, full_impression) with one that works
   (impression). Form ONE mechanistic hypothesis and test it with probes before
   editing.
3. Edit. Make the smallest change that follows from the hypothesis. Typical shapes:
   - a rank-1 change to an MLP `down_proj` or an attention `o_proj`;
   - removing or adding a direction in a few output columns.
4. Run a dose-response: score at 2-3 scales. A real mechanism gives a monotone,
   graded response.
5. Run your own control: a random direction with the same shape and norm should do
   nothing comparable.
6. Iterate. Keep only edits that improve `asym_mean` and keep `up` and the guards
   intact.

Keep a lab notebook at `/scratch/dlf8982/harness/work/NOTEBOOK.md`. Record every
hypothesis, the evidence for or against it, every edit (file, scale, score number) and
what you concluded. Finish with `/scratch/dlf8982/harness/work/FINAL.md` containing:
- the chosen edit file and scale;
- the causal story;
- the evidence for each link in the story;
- what is NOT verified;
- the practice report of the final edit.

Stop when the budget is spent, or earlier if you have a clean result.

Privacy: prompts contain MIMIC patient text (PhysioNet DUA). Never send them to any
external service, never commit `data/` or outputs to git, and keep work under
/scratch/dlf8982 (chmod 700).
