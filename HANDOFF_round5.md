# Handoff, round 5: where does the rest of the negative result go?

You are the same researcher as in rounds 1-4. Before starting, read:
- `HANDOFF_weight_edit.md` (rules);
- `HANDOFF_round3.md` (tools);
- `/scratch/dlf8982/harness_r3/work/ROUND3.md`;
- `/scratch/dlf8982/harness_r3/work/ROUND4.md`.

This round is **interpretation, not repair**. There is no edit to deliver. Keep the scope narrow: the three
questions below and one contrast. Do not start new lines of inquiry. If something surprising comes up, note it in
NOTEBOOK.md under "open leads" and continue.

## Setup
- **Server.** The human starts it on a fresh directory, which gives a fresh budget of 60 score calls and 4000
  probes:

  ```
  mkdir -p /scratch/dlf8982/harness_r5 && chmod 700 /scratch/dlf8982/harness_r5
  nohup python -m harness.server --model Qwen/Qwen3.5-9B --out /scratch/dlf8982/harness_r5 \
    > /scratch/dlf8982/harness_r5/server.log 2>&1 &
  ```

- **Paths.** Pass `--out /scratch/dlf8982/harness_r5` to every client call. Put your files in
  `/scratch/dlf8982/harness_r5/work/`. You may read anything in `/scratch/dlf8982/harness/work/` and
  `/scratch/dlf8982/harness_r3/work/`: cached activations, dV, dP, R, O, your scripts.
- **Rules.** Same as before:
  - no test split;
  - no edits to `harness/`;
  - every forward pass goes through the server.
- **Before you start,** check `status`: there should be no active edit and no active interventions.

## What is already established (do not re-test)
1. Result information moves from the result tokens to the answer position between L12 and L20, on the same
   schedule for P and N (round 4, task 5).
2. At the answer position, removing dV abolishes `down` and adding dV reproduces it (round 3). It is written
   gradually over L19–29 by many components (round 2).
3. L30–31 read the result text again and add the other half of the push, but only when dV is present (round 3).
4. The L31 squash is mostly symmetric norm dilution; MLP31 self-repairs (rounds 2–3). It is not the source of the
   asymmetry.
5. Neutral "result present" (R) and "organ mentioned" (O) components do not push Yes (round 4).
6. **The fact this round starts from.** |dP| ≈ |dN| (ratio 1.1–1.26 at L19–31). But per unit of displacement,
   N's logit-lens readout toward No is only 1/2 to 1/5 of P's readout toward Yes at L23–29 (round 4, R4.5).

**So the question is: N moves the state as much as P, but most of that movement is not on the readout axis. What
is the rest, is it used, and who writes it?**

## Definitions (avoid circularity)
- **Readout axis u_L.** Define it from the model's own output head, not from P/N data. Use the same linearized
  logit lens as `r4_mag.py`: the Yes−No unembedding difference, passed through the final norm weight and scaled
  by the B-prompt rms at layer L.
- **Do NOT use dV as the axis.** dV was defined from P and N: it is dN orthogonalised to dP.
- **Decomposition.** For each layer L in {19, 21, 23, 25, 27, 29}:
  - `N_on = (dN·û) û` and `N_off = dN − N_on`;
  - likewise `P_on` and `P_off`.

  Here dN and dP are the mean (N − B) and (P − B) at the answer position over practice mid-prior cases, per cell.
  Fit on half the cases and check stability (cosine) on the other half.
- **Allowed and forbidden comparisons.** You may report cos(N_off, P_off) as a description. Do NOT define any
  "shared part" from N_off and P_off and then claim it explains the asymmetry; that is the round-3 error.
- **Caveat.** The logit lens at intermediate layers only approximates what L30–31 actually read. Question 2
  exists to check this.

## Q1. What is N_off? (mostly free, from caches; ≤300 probes for new references)
- Report per layer:
  - `|N_off|/|dN|` and `|P_off|/|dP|`;
  - stability across halves.
- Logit lens of N_off (top promoted and suppressed tokens, excluding Yes/No), and the same for P_off.
- Cosine of N_off with reference directions defined from **other prompts**. Reuse round-2 and round-4 caches
  where possible:
  - R, R_admin, O (round 4);
  - history negation ("Denies fever") − matched affirmation;
  - positive finding of a different disease (ureteral stone when asked about appendicitis) − pending;
  - general-knowledge answer No − answer Yes;
  - **hedge**: "X cannot be excluded" / "equivocal" − a definitive negative, using your own texts;
  - **disease word alone**: the negative sentence with the negator removed, i.e. its affirmative twin, written so
    that the token count matches, − pending. This tests the "affirmative first" reading. Report it for each
    layer separately: an early high cosine that falls later is the pattern to look for.
- **Per cell:** does `|N_off|/|dN|` (or N's readout per unit) predict the cell's under-weighting? Correlate
  across the 8 practice cells with that cell's `down / lit_down`. `findings`, the worst cell, is the key point.

## Q2. Is N_off used downstream? (score calls)
- **Remove N_off** from N prompts at L29, answer position. Use `project` along the unit N_off with `center` = the
  mean B projection, at strengths 0.5 and 1. Predictions:
  - if `down` rises, N_off actively counteracts No downstream (a Yes-leaning extra part that L30–31 read);
  - if `down` is unchanged, N_off is wasted (written but not read);
  - if `down` falls, the logit lens missed a real No contribution.

  Report `up`, `down`, B drift and guards.
- **Add N_off** to B (alpha 1 and 2). Does the B margin move, and in which direction?
- **Control.** Do the same with P_off on P prompts. Is P's off-axis part also unused?
- **Random controls:** same norm; and random orthogonal to u, N_off and P_off.
- Repeat at L25 only if L29 is informative.

## Q3. Who writes N_on and N_off? (≤1500 probes, then a few score calls)
- Capture per-component outputs at the answer position for N, P and B prompts at L19–29:
  - each MLP;
  - each mixer (DeltaNet or attention);
  - per head at the full-attention layers 19, 23 and 27 (use `o_proj` input).
- For each component, project its (N − B) write onto û and onto the unit N_off, and its (P − B) write onto û and
  the unit P_off.
- Report:
  - the top writers of N_on, N_off and P_on;
  - whether the same components write P_on and N_on (P with the sign flipped);
  - whether the main N_off writers are components that also write P_on. That would mean "a positive-evidence
    writer that fires on the disease word, regardless of the negator".
- **Causal check.** Ablate the top N_off writers that you can ablate (MLP neurons via `ablate_neurons`, heads at
  full-attention layers via `ablate_heads`). Score, with a random control of matched size. You cannot ablate
  DeltaNet mixers by head; say so if they dominate.

## Contrast: a lab negative that is NOT under-weighted (≤600 probes)
In the earlier held-out test, lipase negatives were not under-weighted:
- "within normal limits" and "32 U/L" gave asym −0.32/−0.37;
- CT negatives gave +0.8 to +1.2.

Use this contrast without touching the test split:
- Write your **own** lab-negative and lab-positive wordings. Do not copy the test-split sentences. Graft them onto
  practice histories with the question "acute pancreatitis", as in round 2.
- Add one or two CT negatives that contain no negator and no disease word ("The appendix is normal in caliber
  and enhancement").
- First check behaviorally, with `logits`, that your lab negatives are again not under-weighted relative to the
  literature on practice histories. If they are, report that and skip the rest of this contrast.
- Then compute the same decomposition. Is `|N_off|/|dN|` much smaller for the lab negatives than for the CT
  negatives? Does the lab N_off resemble the CT N_off?

## Budget plan (suggested)
- Q1 references: about 300 probes.
- Q3 captures: about 1500 probes.
- Contrast: about 600 probes.
- Q2 and Q3 causal tests: about 20–25 score calls.
- Keep at least 10 score calls and 500 probes in reserve.

## Report: `work/ROUND5.md`
- **Plain-language summary first.** Answer each question in one or two sentences:
  - What is the off-axis part of a negative result?
  - Is it read downstream?
  - Who writes it?
  - Is it smaller for the lab negative?
- **Then the numbers:**
  - per-layer decomposition and stability;
  - the cosine table with references;
  - the per-cell correlation;
  - score tables with controls;
  - the writer tables.
- **Then the evidence split.** Keep what was causally tested separate from what was only observed. List the
  caveats: the logit-lens approximation, the choice of reference texts, and the grafted lab cases.
- Add a Round 5 section to `work/NOTEBOOK.md` as you go.
