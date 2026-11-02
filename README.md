# medical-llm-interpretability

How does an LLM use **positive vs negative** test evidence in a diagnosis?
Real MIMIC-IV abdominal-pain histories plus one templated test result, scored and
activation-patched on Qwen3.5-9B.

## Experiment: positive / negative evidence patching (`posneg`)

Each case is one real `Patient History` (filtered so it mentions no tests or diagnoses)
plus one constructed test result:

| test | target diagnosis |
|---|---|
| serum lipase | acute pancreatitis |
| CT appendix | acute appendicitis |
| RUQ ultrasound, gallbladder | acute cholecystitis |
| CT sigmoid colon | acute diverticulitis |

Each result is written in 3 surface forms, under 3 conditions:

- forms: `numeric` (value + reference range), `lexical` ("enlarged" / "normal"),
  `negation` ("acute" / "no")
- conditions: `P` positive, `N` negative, `B` baseline ("result pending")

Within a form, P and N have the same token count, so activations can be patched position by
position. The model answers a 4-way multiple choice (A-D). The metric is
`m = logit(target) - logsumexp(other three letters)`.

Patching runs on the residual stream, every layer x 5 position segments, in both directions:

- `denoise`: run N, insert P
- `noise`: run P, insert N

Both report recovery as the fraction of the P-N gap moved.

## Files

| file | where it runs | what it does |
|---|---|---|
| `build_posneg_cases.py` | locally (needs the MIMIC pickles) | writes `data/posneg_pool.jsonl` (40 candidate cases) |
| `patch_posneg.py` | GPU | behavior on the pool, keeps 5 cases per test (20), patches them |
| `plot_posneg.py` | anywhere | `behavior_summary.md` + recovery heatmaps |
| `patch_posneg.slurm` | Quest | runs the two scripts above on Qwen3.5-9B |

## Data

`data/` and `results/` are git-ignored: they contain MIMIC-derived text, which must not be
pushed (PhysioNet DUA). Build the pool locally, then copy it to the cluster:

```
python build_posneg_cases.py --raw-dir ../congraph/data/raw_data
scp data/posneg_pool.jsonl <quest>:/home/dlf8982/AAA/medical-llm-interpretability/data/
```

## Running on Quest

Put the conda env, pip cache and model weights under the project allocation, **not** `$HOME`.
The home quota (~80 GB) fills up: torch + CUDA libs are several GB and Qwen3.5-9B is ~19 GB.
Symptom: `OSError: [Errno 122] Disk quota exceeded`.

One-time setup, on a login node:

```
PROJ=/projects/p31777
export PIP_CACHE_DIR=$PROJ/.cache/pip  HF_HOME=$PROJ/.cache/huggingface  CONDA_PKGS_DIRS=$PROJ/.cache/conda_pkgs
git clone git@github.com:duruoli/medical-llm-interpretability.git ~/AAA/medical-llm-interpretability
conda create -p $PROJ/envs/interp python=3.10 -y && conda activate $PROJ/envs/interp
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA 12 wheel; see note
pip install -r requirements.txt
hf download Qwen/Qwen3.5-9B   # huggingface_hub>=1.0 renamed huggingface-cli to hf
```

Note on torch: a plain `pip install torch` pulls the newest CUDA-13 wheel, which may be too new
for the cluster's GPU driver. The `cu126` index gives a CUDA-12 build. If an existing env already
works on the GPUs, match its CUDA version:
`python -c "import torch; print(torch.__version__, torch.version.cuda)"`.

Optional speed-up for the linear-attention layers (needs a CUDA build; skip it if it fails):

```
pip install flash-linear-attention causal-conv1d
```

Submit from the repo root. Needs 1 x A100 (40 GB is enough: logits are computed for the last position only), 64G memory, under 4 h:

```
mkdir -p data runs && sbatch patch_posneg.slurm
tail -f runs/patch_posneg_<jobid>.out
```

Results land in `results/posneg/Qwen3.5-9B/`.

## Follow-up: mechanism of the handoff (`mechanism_posneg.py`)

Five hand-picked cases (clearest noise-vs-denoise asymmetry in the first run) x 3 forms.
Three measurements in one run (~10 min on an A100; fine in an interactive session):

- component patching at layers 4-12: resid / mixer (DeltaNet or attention) / MLP, at the
  differing token and the words after it
- attention weights to the differing token at every full-attention layer, in clean and
  patched runs, to tell "the host run decides" from "the content decides"
- distance between P and N residual streams per layer x segment ("is it there" vs "is it used")

```
sbatch mechanism_posneg.slurm        # or, in an interactive session: python -u mechanism_posneg.py --model Qwen/Qwen3.5-9B
```

Results land in `results/mech/Qwen3.5-9B/`.

## Follow-up: does the model use the reference range? (`range_posneg.py`)

Behavior only, all 40 pool cases. The numeric result keeps its value while the stated range
changes ("9 mm (normal under 6 mm)" vs "9 mm (normal under 12 mm)" vs "9 mm", and the same
for the low value). If the range is used, changing only the range moves the answer like
changing the value does. About 5 min on an A100.

```
sbatch range_posneg.slurm
```

Results land in `results/range/Qwen3.5-9B/` (`range_summary.md` = per-case tables).

## Follow-up 2: joint patching, logit lens, heads (`followup_posneg.py`)

Same five cases x 3 forms:

- joint patching: residual at the differing token, at the words after it, and at both at once
  (one intervention, not a sum), layers 0-16, both directions
- logit lens: each position's residual at each layer read through the model's own final
  norm + unembedding; which words N raises over P
- heads: swap or zero the heads that attend more to "no" (19.13, 19.14, 27.10) vs a control
  group preferring the positive word (23.12, 23.14, 27.1); and whether holding the "no" heads
  at their P values removes the effect of putting N's notes at the differing token
- cumulative component patching: all DeltaNet / attention / MLP writes into a position over
  layers 0..L at once (L = 3..12), against replacing the residual at every layer 0..L
  (identical at result_tail by construction): was the information carried in or built in place?

```
sbatch followup_posneg.slurm        # A100, 8 CPU, 64G, 1 h; results in results/followup/Qwen3.5-9B/
```

## Local smoke test

Runs on CPU with a small model. Slow (about 14 min per case), so it is only for catching bugs:

```
python patch_posneg.py --model Qwen/Qwen3.5-0.8B --pool-per-test 1 --select-per-test 1
```

Check: patching the `final` segment at the last layer must give recovery 1.000. The script
prints `WARNING sanity` if it does not.
