# Handoff, round 3: causal tests with activation interventions

You are the same researcher as in rounds 1-2. Read these first:
- `briefs/round1_weight_edit.md` for the rules;
- `$SCRATCH/harness/work/ROUND2.md`;
- the Round 2 part of `$SCRATCH/harness/work/NOTEBOOK.md`.

Round 2 asked for two server features. Both now exist.

This round is **interpretation, not repair**. There is no final edit to deliver. Every intervention below is a
temporary forward hook: it changes no weights and is removed with `clear`.

## Setup for this round

The human restarts the server on a fresh output directory, which gives a fresh budget of 60 score calls and
4000 probes:

```
nohup python -m harness.server --model Qwen/Qwen3.5-9B --out $SCRATCH/harness_r3 \
  > $SCRATCH/harness_r3/server.log 2>&1 &
```

- Pass `--out $SCRATCH/harness_r3` to every client call.
- Put your files in `$SCRATCH/harness_r3/work/`.
- You may read anything you produced in `$SCRATCH/harness/work/` (cached activations, the dV vector,
  your scripts).
- Same rules as before:
  - no test split;
  - no edits to `harness/`;
  - all forward passes go through the server.

## New server tools

```
python -m harness.client --out $SCRATCH/harness_r3 intervene spec.json   # install (free)
python -m harness.client --out $SCRATCH/harness_r3 clear                 # remove (free)
```

`spec.json` = `{"interventions": [ ... ], "freeze": [module names]}`. Installing a new spec replaces the old
one. While a spec is active it applies to every `score`, `logits` and `acts` call, and the reports and the log
record it.

| type | fields | effect |
|---|---|---|
| `add` | `layer` (or `module`), `vector` (list, or path to .pt + optional `key`), `alpha`, `positions` | `h[pos] += alpha * v` |
| `project` | `layer`/`module`, `vector`, `strength` (default 1), `center` (default 0), `positions` | `h[pos] -= strength * ((h·v̂) − center) v̂`; with `center` = the mean B projection, this sets the component to its B level |
| `patch` | `layer`/`module`, `positions` (`"last"` or negative ints) | `h[pos]` = the same module, same position counted from the end, in that prompt's `patch_from` source. Every probe prompt must then carry `"patch_from": <prompt>`. The source run costs one extra probe. Not usable with `score`. |
| `ablate_neurons` | `layer`, `neurons`, `scale` (default 1) | MLP neurons scaled by (1 − scale) at the input of `down_proj` |
| `ablate_heads` | `layer` (full-attention layers only), `heads`, `scale` | heads scaled by (1 − scale) at the input of `o_proj` |
| `freeze` (top-level list) | module names, e.g. `"<prefix>.31.mlp"` | these modules output their value on the ORIGINAL model (no weight edit, no interventions) for the same prompt |

Notes:
- **`layer: L`** hooks the output of decoder layer L (0-based), which is the residual stream after layer L. The
  layer prefix is in `model_info.json`. Check that this matches what you called "L29" in round 2 before reusing
  your dV vector.
- **`positions`**: `"last"` (default), `"all"`, or a list of ints.
- **`freeze` works with both weight edits (`apply`) and hook interventions.** Each prompt then runs twice: once
  clean to record the frozen modules, once for real. A score call takes about 2x as long.
- **Verified on 0.8B:**
  - `add` with alpha 0 and `project` with strength 0 leave the output unchanged;
  - patching the last layer reproduces the source margin;
  - `ablate_neurons` and `ablate_heads` equal zeroing the same weights;
  - freezing an edited MLP restores the original output exactly.

## Tasks

### A. Is the shared negative direction dV the cause of `down`?

Round 2 showed dV is present, but the only causal test removed about 9% of it.

1. **Remove it.** On practice N prompts, `project` dV at the last position of the layer where you measured it,
   with `center` = the mean projection on matched B prompts and strength 0.5 and 1.0. Prediction if dV carries
   `down`: down shrinks toward 0 while `up` and B barely move.
   - The signal may be rebuilt from the result tokens in later layers. So also try removing it at several
     layers at once (e.g. every layer from L23 to the last) and/or at all positions.
   - Report how much of `down` each variant removes.
2. **Add it.** On B prompts (and some P prompts), `add` alpha · dV at the same place, with alpha set so that
   the added projection equals the mean N−B projection, plus a dose series. Prediction: B drops by about the
   mean `down`. What happens to P?
3. **Patch.** Patch the full residual at the last position from each case's N prompt into its B prompt
   (`patch_from`), at a sweep of layers. This gives the fraction of `down` carried by the last position at
   each depth, as an upper bound for what dV can explain.
4. **Controls.** For 1 and 2, use random directions with the same norm, including random directions
   orthogonal to dP. They should do nothing comparable.

Use `logits` probes for exploration and spend `score` calls on the decisive versions. `project` and `add`
work with `score`; `patch` does not.

### B. Online check of self-repair

With MLP31 frozen:
- ablate head 31.12, then head 31.14 (`ablate_heads`);
- apply the round-1 `FINAL_edit.pt` at ×1.5 (`apply`), keeping the freeze spec installed.

Compare the measured `up`/`down` with your offline predictions:
- 31.12 frozen: +1.03 / +0.73;
- FINAL frozen: down +2.24 on your own tests.

If they match, self-repair is confirmed online.

### C. Distinguishing experiment: "evidence against the hypothesis" vs "the planned answer is No"

dV loads as much on general questions whose answer is No as on real negative results. Two readings remain:

- **(E)** dV encodes *evidence against the asked proposition*.
- **(A)** dV encodes *the model's forming answer "No"*.

Separate them with a 2×2 design:

| | answer stays Yes | answer is No |
|---|---|---|
| evidence against | strong-prior history + a weak or unreliable negative, selected so that the final margin stays > 0 | ordinary N |
| no evidence | high-prior B (result pending, margin > 0) | low-prior B (result pending, margin < 0) |

Add positive results whose answer stays No (low prior + weak positive) as a fifth cell if you can.

- **Measure** the dV projection at the last position at your layer, plus a few earlier and later layers.
  Compare absolute projections with high-prior B as the reference.
- **Regress** the projection on (evidence: −1 / 0 / +1) and on the final margin across all prompts. Report
  which one explains it, and whether that changes with depth (an answer representation should track the margin
  more strongly near the output).
- **Causal follow-up.** Add dV to general yes/no questions whose answer is Yes. Does it flip them as easily as
  it lowers B in the medical prompts? Interpret this carefully: both readings predict some effect.

Write your own prompts for C (practice histories are fine; never the test split). Keep 10+ cases per cell.

## Report

Log everything in `$SCRATCH/harness_r3/work/NOTEBOOK.md`. Finish with `work/ROUND3.md`:
- a plain-language summary first, then the numbers;
- a clear split between what was causally tested and what was only observed;
- for C, which reading the evidence favours and how strongly.
