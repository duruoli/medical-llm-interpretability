# Weight-edit harness: positive-evidence bias in Qwen3.5-9B

A harness for **weight-space editing without training**. A researcher (in practice an LLM coding
agent, e.g. Claude Code) diagnoses a behavioural flaw from weights and activations, writes a small
weight delta by hand, and tests it. The harness owns the model, enforces the rules and the budget,
and keeps a held-out test split that the researcher never sees.

Method reference: [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/).

## The flaw being fixed

Prompt: a real emergency-department history (MIMIC-IV), one test result, then
*"Does this patient have <disease>? Answer Yes or No only."*
Belief is the log-odds `m = logit(Yes) - logit(No)` at the answer position.

For every patient and test there are three prompts:

| variant | result line |
|---|---|
| `B` | "result pending" (baseline) |
| `P` | positive result |
| `N` | negative result |

- `up   = m_P - m_B` — how far a positive result raises belief
- `down = m_B - m_N` — how far a negative result lowers it

An ideal Bayesian moves by `ln LR+` up and `-ln LR-` down, whatever the prior. The literature
sensitivity/specificity of each test (`harness/config.py: LIT`) gives the ideal ratio `lit_up / lit_down`.

Qwen3.5-9B over-weights positive evidence: on CT for diverticulitis its `up/down` ratio is about
2-20x depending on wording, against a literature ratio of about 1.1-1.6. In some wordings a
*negative* CT even *raises* belief. The goal is an edit that makes negative results count as much
as they should, without weakening positive results and without side effects.

### Score

Per cell (test x wording), on "mid-prior" cases only (original baseline `P(Yes)` in `[0.02, 0.95]`):

```
asym = ln(up / down) - ln(lit_up / lit_down)        # 0 = literature ratio, > 0 = positive bias
```

| metric | meaning |
|---|---|
| `asym_mean` | mean `asym` over cells (suite `v1` target: bring toward 0) |
| `abs_asym` | mean `|asym|`; over- and under-correction both count (suite `repair` target) |
| `move_err` | mean `|up - lit_up| + |down - lit_down|`; catches "fixed the ratio by killing positives" |
| `neg_raises` | cases where a negative result raised belief |

Guards (an edit that breaks these is rejected): `baseline_drift` (change in the no-result belief),
`yesno_mass` (still answers Yes/No), `general_acc` / `general_shift` (40 general yes/no questions,
catches "just say No more"), `history_nll` (plain language modelling on clinical text).

## Pipeline

```
 MIMIC pickles ──(1) build_sigmoid_cases.py──▶ data/sigmoid_pool.jsonl
                                                  │
                         (2) harness.make_splits  ▼  split by PATIENT, stratified by true disease
                 data/harness/practice.jsonl          data/harness_test/test.jsonl
                          │                                   │  (never shown to the researcher)
       (3) harness.server ▼  GPU, holds the model             │
                          ▲                                   │
       (4) researcher ────┘  harness.client:                  │
           (agent + briefs/)   score / apply / logits / acts / │
                               weights / intervene             │
                          │ final edit .pt                    │
                          ▼                                   ▼
                     (5) harness.final_eval  (run by the human) ──▶ $HARNESS_OUT/final/<edit>.md
```

### 0. Setup

```bash
git clone -b harness-qwen3.5-9b git@github.com:duruoli/medical-llm-interpretability.git
cd medical-llm-interpretability
conda create -n wedit python=3.10 -y && conda activate wedit
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick the CUDA build your driver supports
pip install -r requirements.txt
hf download Qwen/Qwen3.5-9B
export HARNESS_OUT=$SCRATCH/harness     # where the server writes everything (default: out/harness)
mkdir -p $HARNESS_OUT && chmod 700 $HARNESS_OUT
```

Hardware: one GPU with >= 40 GB (A100 / H100). The model is ~19 GB in bf16. One `score` call takes
about 2 min (suite `v1`) or 3.5 min (suite `repair`) on an A100.

On a cluster, put the conda env, pip cache and `HF_HOME` on a project allocation rather than
`$HOME`: torch plus the model weights easily exceed a home quota.

### 1. Build the case pool (needs MIMIC access)

The input is the per-disease `{disease}_hadm_info_first_diag.pkl` files of the MIMIC-IV clinical
decision-making dataset (appendicitis, cholecystitis, diverticulitis, pancreatitis), which requires
PhysioNet credentialed access. Only the `Patient History` field is used.

```bash
python build_sigmoid_cases.py --raw-dir /path/to/raw_data     # -> data/sigmoid_pool.jsonl
```

Histories that mention imaging, labs, a candidate diagnosis or a relevant operation are dropped
(they would leak the answer). Each case gets two tests on the same history (`ct_sigmoid` ->
diverticulitis, `ct_appendix` -> appendicitis), each in several wordings
(`lexical`, `impression`, `full_impression`, `findings`, `numeric`).
`build_posneg_cases.py` is imported for the shared filter and history loader.

If the pool is built on a laptop, copy it to the GPU machine with `scp`; never commit it.

### 2. Make the practice / test split

```bash
python -m harness.make_splits --suite v1       # rounds 1-5
python -m harness.make_splits --suite repair   # round 6+
```

Patients are split in half (seed 0, same split in both suites). The test half additionally gets
**held-out wordings** and **held-out tests** that never appear in practice:

| suite | practice | test (held out) |
|---|---|---|
| `v1` | CT sigmoid + CT appendix, 4 wordings | + 2 new wordings, + gallbladder ultrasound, + serum lipase |
| `repair` | as `v1` + serum lipase (a "do not break" control) | + new / implicit wordings, + appendix ultrasound, + kidney-stone CT, + amylase |

All held-out definitions live in `harness/config.py` (`HELDOUT_*`, `REPAIR_HELDOUT_*`).

### 3. Start the server (once per GPU session)

```bash
nohup python -m harness.server --model Qwen/Qwen3.5-9B \
    --practice data/harness/practice.jsonl --out $HARNESS_OUT \
    > $HARNESS_OUT/server.log 2>&1 &
```

On start it scores the original model on the practice split (free; saved to
`$HARNESS_OUT/scores/original.txt`), writes `model_info.json` (all parameter and module names),
then listens on `127.0.0.1:8765`. Use `--limit-cases 4` for a quick smoke test.
For suite `repair`, pass `--practice data/harness/repair_practice.jsonl`.

### 4. The researcher edits the model

The researcher talks to the server only through the client:

```bash
python -m harness.client status                            # budget left, active edit
python -m harness.client score                             # full practice evaluation   [1 score call]
python -m harness.client apply EDIT.pt [--scale s] [--arm A_readout] [--note "..."]   # W = W_orig + s*delta (free)
python -m harness.client reset                             # back to original weights (free)
python -m harness.client weights NAME [NAME ...]           # ORIGINAL weights -> $HARNESS_OUT/weights/*.pt (free)
python -m harness.client logits prompts.json [--top-k 5]   # yes/no margin + top tokens [probe prompts]
python -m harness.client acts spec.json                    # activations -> $HARNESS_OUT/acts/*.pt [probe prompts]
python -m harness.client intervene spec.json               # diagnostic hooks, no weight change (free)
python -m harness.client clear                             # remove hooks (free)
```

From Python: `from harness.client import call; call("logits", prompts=[...])`.

- **Prompt reference**: `{"case": i, "test": "ct_sigmoid", "variant": "B"}`,
  `{"case": i, "test": "ct_appendix", "pair": "findings", "variant": "N"}`, or
  `{"text": "any user message"}` (chat template applied; `"raw": true` skips it).
  `case` indexes the practice file.
- **Activation spec**: `{"prompts": [...], "modules": ["model.layers.12.mlp", ...],
  "positions": "last" | "all" | [ints], "kind": "output" | "input"}`.
- **Edit file**: `torch.save({"deltas": {param_name: delta_tensor}}, path)`; same shape as the
  weight, float32 on CPU. Every applied edit is copied (with a hash) to `$HARNESS_OUT/edits/`.
- **Diagnostic interventions** (`intervene`): `add` a vector, `project` out / clamp a direction,
  `patch` from another prompt, `ablate_neurons`, `ablate_heads`, and `freeze` modules to their
  original-model outputs (tests self-repair). They stay on for every later `score` / `logits` /
  `acts` call until `clear`. Spec format: docstring of `harness/core.py` and `briefs/round3.md`.

Rules enforced by the server (`harness/config.py`):

| rule | value |
|---|---|
| forbidden parameters | embeddings, `lm_head`, final norm, vision tower |
| max changed weights per edit | 5,000,000 (~0.06% of 9B) |
| budget | 60 `score` calls, 4000 probe prompts (max 256 per call); override with `--score-calls`, `--probe-prompts` |
| edit arms (`--arm`, suite `repair`) | `A_readout` (layers 30-31), `B_translate` (attention / DeltaNet mixers in 23-29), `C_free` (anything allowed) |

Rules the researcher must follow by convention: never read `data/harness_test/`, never edit
`harness/`, never load the model outside the server. CPU-side analysis of saved weights and
activations is free and unlimited. `.claude/settings.json` denies Claude Code access to the test
split and to the harness code.

#### Running it with an agent

`briefs/` holds the task briefs given to the agent, one per round. Each is self-contained:
the flaw, the score, the rules, the tools, a suggested workflow and the required write-up
(`work/NOTEBOOK.md` lab notebook, `work/FINAL.md` with the causal story and the evidence).

| brief | round focus |
|---|---|
| `round1_weight_edit.md` | **start here**: full task, rules and tools; first edit |
| `round3.md` | diagnostic interventions (add / project / patch / ablate / freeze) |
| `round4.md` | non-circular shared components |
| `round5.md` | the off-axis part of negative evidence |
| `round6.md` | repair suite: `abs_asym`, lipase control, edit arms |

Rounds 3-6 build on findings written by earlier rounds; they are kept as examples of how the task
was narrowed. Paths in the briefs use `$SCRATCH` for the output root. Typical start:

```bash
cd medical-llm-interpretability && claude
> Read briefs/round1_weight_edit.md and do the task. The server is running with --out $HARNESS_OUT.
```

### 5. Held-out evaluation (human only)

```bash
python -m harness.final_eval --edit $HARNESS_OUT/edits/007_x.pt --scales 0.5 1
python -m harness.final_eval --suite repair --edit A.pt B.pt C.pt --arm A_readout B_translate C_free
```

Scores the original model once, then each edit at each scale, plus a **random control** (same
parameters, same support, same norm per parameter, random content). The report groups cells into
*seen test + wording, new patients* / *new wording* / *new test*, and *imaging* vs *lab*, so an edit
that only fits the practice wordings shows up. Output: `$HARNESS_OUT/final/<edit>.md` and `.json`.

## Output directory

```
$HARNESS_OUT/
  server.json         port (read by the client)
  model_info.json     config, parameter shapes, module types
  budget.json         budget used
  log.jsonl           every command, with active edit and interventions
  scores/             original.txt, 001.txt/.json, ...
  edits/              hashed copy of every applied edit
  acts/  weights/     saved activations and original weights
  work/               the researcher's scripts, notebook, FINAL.md (by convention)
  final/              held-out reports (final_eval)
```

## Repository layout

| file | role |
|---|---|
| `build_posneg_cases.py` | shared MIMIC loader, leak filter, prompt (imported by the builder below) |
| `build_sigmoid_cases.py` | builds `data/sigmoid_pool.jsonl` |
| `harness/config.py` | everything the researcher may NOT change: literature LRs, prompts, held-out sets, suites, arms, guards, rules, budget |
| `harness/make_splits.py` | patient-level practice / test split per suite |
| `harness/core.py` | model wrapper: edit with rollback, interventions, scoring, guards, report |
| `harness/server.py` | HTTP service on the GPU; enforces rules and budget; logs everything |
| `harness/client.py` | command-line / Python client |
| `harness/final_eval.py` | held-out evaluation with random control |
| `briefs/` | agent task briefs, rounds 1-6 |

## Adapting to another model or flaw

- **Model**: `--model` on server and final_eval. The edit arms (`ARMS`) and the
  `FORBIDDEN` pattern assume Qwen3.5 parameter names (`layers.N.self_attn|linear_attn|mlp`); adjust
  them for other architectures.
- **Flaw**: change the prompt, the tests and their likelihood ratios in `config.py` and the
  builder; `core.cell_stats` / `core.report` define the score.

## Privacy

Prompts contain MIMIC patient text (PhysioNet DUA). `data/` and all outputs are git-ignored.
Never commit them, never send them to an external service, and keep the output directory
private (`chmod 700`). An LLM agent used as the researcher must run where the data use agreement
permits it.
