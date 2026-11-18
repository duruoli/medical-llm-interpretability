# Handoff, round 6: REPAIR — make each test move belief by the right amount

You are the same kind of researcher as in rounds 1-5, but this round is **repair, not interpretation**. You must
deliver edits, and a score decides whether they work. Before starting, read:
- `HANDOFF_weight_edit.md` (rules, tools, workflow; still valid unless changed below);
- `HANDOFF_round3.md` (diagnostic interventions; still available for diagnosis);
- the reports `/scratch/dlf8982/harness/work/{FINAL.md,ROUND2.md}`,
  `/scratch/dlf8982/harness_r3/work/{ROUND3.md,ROUND4.md}` and `/scratch/dlf8982/harness_r5/work/ROUND5.md`.

## Why this round, and what changed
- **Round 1's edit** worked on CT but pushed *every* negative toward No. It overshot lipase, a test whose
  negatives were already weighted correctly. Practice never contained such a test, so you could not see it.
- **Round 5 explains why.** Lab and CT negatives write the same off-axis "definite result, points elsewhere"
  content (cos 0.89). They differ only in the small part on the readout axis (lab −0.12, CT −0.03 to −0.05).
  An edit that reads the negative-result signal therefore cannot tell lab from CT.
- **Where the on-axis difference lives.** It is written in L23–29:
  - heads 27.11 and 27.13 write Yes for positives and almost nothing for negatives;
  - head 27.12 writes both ways;
  - the L28 DeltaNet mixer writes +1.76 for P and 0 for N;
  - small No contributions come from L23 attention and MLPs 26–28.

  In L30–31, P's off-axis content is turned into Yes again; N's is not turned into No.

**Changes to the harness (suite `repair`):**
1. **Practice now contains lipase** (numeric and lexical wordings, question "acute pancreatitis", grafted onto
   the practice histories). It is the "do not break this" control.
2. **New primary metric `abs_asym`:** the mean over cells of |asym|. Over-correction now costs as much as
   under-correction. The score report also gives `abs_asym` per test. `asym_mean` is still shown.
3. **Fresh held-out set** (you never see it): new wordings, including negatives with no negator or disease word,
   and new tests (imaging and lab).
4. **Edit arms.** `apply` takes `--arm`, and the server rejects parameters outside the arm:

   | arm | allowed parameters | hypothesis it tests |
   |---|---|---|
   | `A_readout` | anything in layers 30–31 (attention, mixer, MLP) | the fault is in the final readout (round-1 style) |
   | `B_translate` | attention / DeltaNet mixer parameters in layers 23–29, no MLPs | the fault is that the L23–29 "translators" (27.11, 27.13, L28 mixer) only translate positives |
   | `C_free` | any layer, any attention / mixer / MLP parameter | whatever works best |

   The global rules still hold: no embeddings, lm_head or final norm, and at most 5,000,000 changed weights.

## Setup
The human prepares the data and starts the server:

```
python -m harness.make_splits --suite repair      # writes data/harness/repair_practice.jsonl (+ held-out test)
mkdir -p /scratch/dlf8982/harness_r6 && chmod 700 /scratch/dlf8982/harness_r6
nohup python -m harness.server --model Qwen/Qwen3.5-9B --practice data/harness/repair_practice.jsonl \
  --out /scratch/dlf8982/harness_r6 --score-calls 80 --probe-prompts 5000 \
  > /scratch/dlf8982/harness_r6/server.log 2>&1 &
```

- **Paths.** Pass `--out /scratch/dlf8982/harness_r6` to every client call. Put files in
  `/scratch/dlf8982/harness_r6/work/`. You may read your earlier work directories, including the round-1 edit
  `FINAL_edit.pt` and the dV, R, O and N_off vectors.
- **Prompts.** Practice prompts can address lipase as `{"case": i, "test": "lipase", "pair": "numeric",
  "variant": "N"}`.
- **Speed.** A score call now takes about 3.5 minutes, because practice has about 30% more prompts.
- **Budget:** 80 score calls and 5000 probes.
- **Rules.** No test split: `data/harness_test/` is off limits, including `repair_test.jsonl`. No edits to
  `harness/`. Every forward pass goes through the server.

## Step 0: aim before cutting (≤700 probes, no score calls)
Step 0 has two parts. Both shape where the edits should go.

**Step 0 may be run separately.** The round-5 researcher may run it on the round-5 server and write
`/scratch/dlf8982/harness_r5/work/STEP0.md`.
- If that file exists, read it and skip Step 0 here.
- If the human tells you it is still running, start with arm **A**, which does not depend on Step 0. Do arm B
  once STEP0.md is there.
- The budget, server and output directory of this round are unaffected.

### 0a. Which components translate the lab negative? (≤300 probes)
Round 5 did not measure *which components translate the lab negative into No*. Do that first.
- Use lipase negatives from practice, and CT negatives for the *same* pancreatitis question on the same histories.
  Write the CT text yourself, as in round 5.
- Use the same per-component projection onto û as in round 5. Report the heads at L23 and L27, the DeltaNet
  mixers at L24–29, and the MLPs at L23–29.
- **The question:** does the lab negative's extra on-axis No come from the same translators (27.11, 27.13, L28
  mixer) responding to the lab negative, or from other components?
  - If the same translators: arm B should be able to fix CT without breaking lab.
  - If other components: say so, and aim arm B at what you find.

### 0b. Why is `ct_sigmoid/findings` so weak? The "inference distance" hypothesis (≤400 probes)
Look at the negative sentences and how much of the literature move each one gets:

| cell | negative sentence | down / lit_down |
|---|---|---|
| sigmoid impression | "No evidence of diverticulitis." | 0.75 |
| sigmoid lexical | "sigmoid colon wall normal." | **0.20** |
| sigmoid findings | "The sigmoid colon and rectum are normal in appearance." | **0.08** |
| appendix impression | "No evidence of appendicitis." | 0.64 |
| appendix lexical | "appendix normal." | 0.79 |
| appendix findings | "The appendix is normal in caliber and appearance." | 0.65 |

Saying "the organ is normal" works for the appendix but not for the sigmoid colon.

**Hypothesis (untested).** "Appendix normal" means "no appendicitis" in one step, because appendicitis is
inflammation of the appendix itself. Diverticulitis is inflammation of diverticula, small pouches in the colon
wall. So "sigmoid colon normal" needs one more step: no inflamed pouches, therefore no diverticulitis. The word
"diverticula" never appears in the sentence. The weakness would then be in *linking the result to the
question*. That fits round 5's finding that the negative's displacement is already short at L19, before the
translators.

**Test.** Fix the histories (practice, mid-prior), change only the negative sentence, and measure the move with
`logits` and the displacement size and on-axis share at L19, L23 and L29:

| variant | example | prediction if the hypothesis holds |
|---|---|---|
| name the pouches | "No diverticula or inflammation in the sigmoid colon." | much stronger |
| drop "and rectum" | "The sigmoid colon is normal in appearance." | about the same |
| make the appendix indirect | "The right lower abdomen is normal." (no appendix mentioned) | the appendix weakens too |
| simplify the positive | "Focal wall thickening of the sigmoid colon." only | the P/N size ratio improves, `down` unchanged |

The third row is the key one. If the indirect appendix sentence also weakens, the rule is general: **the
further the negative evidence is from the asked disease, the less the model uses it.** That is not specific to
the sigmoid colon.

Use 2–3 wordings per variant, written by you.

Other explanations not yet excluded:
- the sigmoid findings positive is unusually strong: it lists three abnormalities and names diverticula;
- "and rectum" reads like boilerplate;
- in `full_impression` the negative says "diverticulosis without diverticulitis", and pouches are themselves a
  risk factor.

**Consequence for the arms.** If the hypothesis holds, arms A and B are not expected to fix this cell, because
the fault is before the translators. Say so, and check in step 4 whether arm C found something earlier.

### Update after Step 0 (human, 11-17): how to aim the arms
Step 0 is done; see `/scratch/dlf8982/harness_r5/work/STEP0.md`. It changes the aiming:

- **Arm B: aim at the route the lab negative already uses, not at 27.11/27.13.** The Yes-only translators barely
  sign-flip even for the lab negative (18% of its extra No, mostly the L28 mixer). The existing No route runs
  through:
  - H27.12, 15%;
  - H23.5, about 10%;
  - the L28 mixer;
  - MLPs L23–28, about 40%. Arm B cannot reach these.

  So arm B's natural edit makes **H27.12, H23.5 and the L28 mixer respond to CT negatives the way they already
  respond to lab negatives**. That means changing what they read, not scaling their output. Scaling their output
  would also push lipase further, and lipase is already over-moved: `down` 2.7 vs lit 1.39.
- **Arm C: the first thing to try is the same route including the MLPs.** That is attention, mixers and MLPs in
  L23–29, which no other arm covers. The rest of C stays free. Say which you did.
- **`ct_sigmoid/findings` is a linking failure, not a translation failure.** Naming the pouches fixes it (0.01 →
  0.86), and making the appendix indirect weakens the appendix (0.74 → 0.50). The failing sentences are short at
  L19 and never align with the readout. Do not spend arm A or B budget on this cell. If arm C has budget left, an
  attempt at L12–20 is welcome but optional.
- **A caveat on the 0a contrast.** CT negatives for pancreatitis move the wrong way (+0.21) even when they name the
  disease. In real medicine a normal CT does not rule out mild pancreatitis, and lipase is the diagnostic test. So
  part of that contrast may be the model's medical knowledge, not a flaw. The practice score uses only the
  sigmoid and appendix CTs and lipase, so this does not affect scoring. But do not use CT-for-pancreatitis as a
  repair target.

## Steps 1–3: one best edit per arm (about 20–22 score calls each)
Suggested order: **B first** (the main hypothesis), then **A** (the baseline), then **C**.

For each arm:
- Build and test edits as in round 1: analysis is free, then `apply --arm ...`, then `score`.
- **Self-repair.** MLP31 compensates late changes. Edits in A that sit before MLP31 lose about half their effect.
  Use `freeze` (one score call) to diagnose it if you need to.
- **Deliverable:** save the best edit as `work/FINAL_A.pt`, `FINAL_B.pt`, `FINAL_C.pt`. If an arm cannot beat
  the original on `abs_asym` without breaking a guard, deliver nothing for it and explain why. A clear negative
  result for an arm is informative.

**What counts as success, on practice:**
- `abs_asym` clearly below the original;
- the lipase cells' |asym| no worse than the original by more than 0.1;
- `neg_raises` falls;
- guards: `baseline_drift` < 0.15, `general_acc` unchanged, history NLL change < 0.01, and `move_err` not worse
  by more than 0.1.

**Not the target.** Do not try to fix the model's overall overconfidence (e.g. positives moving 5 log-odds where
the literature says 2). Just don't make it worse.

**Report `ct_sigmoid/findings` separately.** Its negative moves the state only 0.57× as far as its positive, and
this is visible from L19, so the weakness starts before the translators. Do not sacrifice other cells to fix
it. Do say whether any arm helps it.

## Step 4: look back at each delivered edit (free, plus ≤600 probes and ≤6 score calls)
For each delivered edit, describe what it does:
- what it reads (cosine of its read vector with û, N_off, P_off, R, dV);
- which heads or components it changes;
- how it changes the on-axis write of lab and CT negatives (repeat the Step 0 projection with the edit applied);
- whether MLP31 cancels part of it (freeze).

The point of the comparison between arms:
- If **B** reaches per-test calibration and **A** overshoots lipase, the fault is in the L23–29 translation.
- If only **A** works, the information is translated, but the readout ignores it.
- If neither, the lab/CT difference lies elsewhere.

## Report: `work/ROUND6.md`
- **Plain-language summary first:**
  - which arms produced a working edit;
  - how each did on CT versus lipase;
  - what that implies about where the fault is.
- **Then:**
  - the Step 0a table (lab vs CT translators) and the Step 0b variant table, saying whether the
    "inference distance" hypothesis held;
  - one score table per arm, with the per-test `abs_asym`;
  - the Step 4 analysis;
  - `ct_sigmoid/findings`.
- **Then the evidence split:** keep what was causally tested separate from what was only observed. List the
  caveats.
- Keep a Round 6 section in `work/NOTEBOOK.md` as you go.

The human will run `harness.final_eval --suite repair` on the held-out set, for all delivered edits, with
random controls.
