# Handoff, round 4: is there a "result present" (or "organ mentioned") component that pushes Yes?

You are the same researcher as in rounds 1-3. Before starting, read:
- `briefs/round1_weight_edit.md` (rules);
- `briefs/round3.md` (tools);
- `$SCRATCH/harness_r3/work/ROUND3.md`.

This round is again **interpretation, not repair**. You will use interventions only; there is no edit to
deliver.

## Setup
- **Server.** Same directory as round 3: `$SCRATCH/harness_r3`. The budget carries over: 46 score calls
  and 714 probes.
- **Probes are scarce.** Use `score` calls for the decisive tests. Plan your probe spending before you start.
- **Files.** Put your files in `$SCRATCH/harness_r3/work/`. Write a new section "Round 4" in
  `NOTEBOOK.md` and finish with `work/ROUND4.md`.
- **Rules.** Do not use the test split. Do not edit `harness/`. Every forward pass goes through the server.
- **After a restart,** check `status` first: there should be no active edit and no active interventions.

## The question and why the obvious test is circular

Round 3 gave a hint. Adding dV alone to B lowered it by 1.66; adding the full mean N−B vector lowered it by only
0.72. So the part of N−B that N shares with P seems to push toward Yes. That would explain the bias directly:
- for a positive result, this part adds to the Yes push;
- for a negative result, it cancels about half of the No push.

**Do NOT define the shared part from P and N themselves.** For example, S = (dN + dP)/2 is circular. Suppose
there is only one yes/no axis, and P moves +5 along it while N moves −2. Then S = +1.5 toward Yes, and removing
S makes up = down = 3.5. The "Yes push" and the "symmetry after removal" would both follow from the very
asymmetry we want to explain. Your round-3 split (the component of dN along dP) is not safe either. Take a 2-D toy: the x axis is yes/no, and
the y axis is "result present" with NO effect on yes/no. Let P = (5, 4) and N = (−2, 4). Then cos(P, N) > 0, as
in the data. Yet the component of N along P̂ is about (0.73, 0.59), which pushes Yes. That push comes only from
measuring along P's direction, which carries P's own valence. So round 3's "add dN −0.72 vs add dV −1.66"
does not show that a shared component pushes Yes. Only the positive cosine itself is solid evidence that P and
N share something.

**Define the candidate components from prompts that are neither positive nor negative.** Two candidates:
- **R, "a result is present".** R = mean(neutral result − pending), at the answer position, per layer. Neutral
  results carry no evidence for or against the target. Examples:
  - "CT abdomen/pelvis: study completed, final report to follow."
  - "CT abdomen/pelvis: technically adequate examination."
  - an irrelevant finding: "Degenerative changes of the lumbar spine."
  - a finding in an organ unrelated to the 4 diseases.

  Use 4-6 such texts, written by you, on practice histories.
- **O, "the organ or disease is mentioned".** O = mean(neutral text that names the target organ − matched
  neutral text that does not). Examples:
  - "The appendix was not specifically evaluated." vs "study completed".
  - "The sigmoid colon was not specifically evaluated." vs "study completed".

  Round 1 found that negatives which name the organ ("The sigmoid colon and rectum are normal") barely move
  belief, and sometimes raise it. O tests whether a mention by itself pushes Yes.

Fit R and O on half of the practice cases you use. Check on the other half that they are stable (cosine between
the halves). You may also report S = (dN + dP)/2 and your round-3 split as descriptive numbers. If you do, state
explicitly that their Yes lean is partly true by construction and is not evidence.

## Tasks, in priority order

### 1. Build R and O and check that they are real (probes, about 200-300)
- Measure each neutral prompt's margin change against pending. Round 1 found that neutral results slightly
  LOWER belief. Record which way each text moves.
- Report per layer:
  - stability across the two halves;
  - |R|;
  - cos(R, dV), cos(R, dP), cos(R, dN);
  - the same quantities for O.

### 2. Are R and O inside P−B and N−B? (free, from the practice cache)
- Project each practice P−B and N−B (per cell, per case) onto R̂ and Ô.
  - If R is a shared "result present" component, P and N should both load positively, by similar amounts.
  - If O is the organ-mention effect, wordings that name the organ should load more. Compare `findings`, which
    names the sigmoid colon or appendix, with `impression`.
- This is a genuine test: the loadings could be zero, unequal or negative.

### 3. Do R and O push Yes? (score calls)
- **Logit lens** of R and O per layer (free).
- **Add R** (and separately O) to all prompts at L29, at the answer position, with alpha equal to the mean
  loading found in task 2 (×1 and ×2). Watch B in particular.
- **Controls:** a random direction with the same norm, and a random direction orthogonal to R, O, dV and dP.

### 4. The decisive test: remove R (and separately O) from P and N (score calls)
- **Setup:** `project` with R̂ (or Ô), with `center` = the mean projection on pending prompts, at the answer
  position, strength 0.5 and 1.0. Run L29 first, then L25 and L23–31 if L29 is informative.
- **Prediction if the component causes the bias:**
  - `up` falls and `down` rises;
  - `asym_mean` falls;
  - `neg_raises` falls;
  - B and the guards barely move.

  For O specifically: the organ-naming wordings (`findings`, `lexical`) should improve most, especially
  `ct_sigmoid/findings`.
- **Controls:** the same random controls as in task 3.
- **How to read the outcome:**
  - Report what fraction of the original asym (+1.12) each removal accounts for.
  - If removal raises both `up` and `down`, or moves B a lot, say so plainly: the hypothesis is then wrong or
    incomplete.
  - If neither R nor O pushes Yes, also say so plainly. Then the bias may be purely a magnitude asymmetry on the
    yes/no axis: P writes its valence more strongly than N. In that case, compare how strongly P and N write
    along their own valence directions in L19–29. Use your round-3 cache; this is free.

### 5. Optional, only if budget remains
- Patch only the result-token positions (negative indices from the end, covering the result sentence) from N
  into B at a few layers. This shows whether what writes dV (and R or O) is specific to the result text.

## Report: `work/ROUND4.md`
- **Plain-language summary first:**
  - Is there a component that pushes Yes for every result, or for every organ mention?
  - How much of the positive bias does removing it explain?
  - If neither, what is left: a magnitude asymmetry in how P and N write?
- **Then the numbers:**
  - geometry and stability of R and O;
  - loadings of P and N by cell;
  - the decisive score tables with controls.
- **Then the evidence split:** keep what was causally tested separate from what was only observed. List any
  remaining caveats, including how much the conclusions depend on your choice of neutral texts.
