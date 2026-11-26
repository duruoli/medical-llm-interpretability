# DDW 2027 abstract — draft v4 (2026-11-25): focus on agent-driven weight editing

Deadline: Wed Dec 2, 2026, 21:00 ET. Body <= 2,900 characters incl. spaces and headings. Title <= 255 chars, ALL CAPS.

## Title

AN LLM AGENT PRECISELY CORRECTS UNDER-WEIGHTED NEGATIVE IMAGING IN AN OPEN LANGUAGE MODEL BY DIRECTLY EDITING ITS WEIGHTS

## Authors

Duruo Li, Dennis L. Shung, Bradly C. Stadie

## Body

<!-- BODY START -->
Background: Large language models (LLMs) are increasingly used for diagnostic reasoning, yet accuracy benchmarks do not show whether a model weighs each test result by the right amount. Bayes' rule gives an objective target: a result should multiply the pretest odds of disease by its likelihood ratio (LR). We used this target to measure how an LLM weighs positive versus negative test results, and whether an LLM agent can correct an imbalance by directly editing model weights.

Methods: We took 188 emergency department histories from MIMIC-IV (appendicitis, cholecystitis, diverticulitis, pancreatitis) with all test results and diagnoses removed. Each was paired with 7 tests (CT for diverticulitis, appendicitis and kidney stone; ultrasound for cholecystitis and appendicitis; serum lipase and amylase), reported as pending, positive or negative, and the open-weight model Qwen3.5-9B was asked "Does this patient have [disease]?". The main metric was the imbalance between the shifts caused by positive and negative results, relative to the ratio predicted by literature LRs (1-fold = balanced). Success criteria were fixed in advance: reduce the imaging imbalance, do not worsen lab tests, and leave pre-result beliefs and general knowledge unchanged. Under these rules, an automated LLM agent searched for a minimal change to the model's internal parameters (a targeted weight edit, not retraining), using half of the patients and 3 tests. We then scored the change on the other 94 patients, new report wordings and 4 tests never used to develop the edit, against random edits of the same size.

Results: Positive imaging reports shifted belief as much as their LR warrants or more, whereas negative reports were under-weighted relative to positives in all 19 imaging wordings; a negative CT achieved only 33-68% of its expected shift. Negative lab results showed no such deficit. At worst, "The sigmoid colon and rectum are normal in appearance" should lower a 50% suspicion of diverticulitis to 6%, but on average it fell only to 44%, and in 48 of 103 patients even rose. The selected edit changed 0.02% of parameters. On held-out data, the overall imbalance fell from 2.5-fold to 1.35-fold (random edits: 2.4-2.5-fold), including 3 imaging tests never seen, and negative results that wrongly raised belief fell from 79 to 2. Lab tests, pre-result beliefs and general knowledge were essentially unchanged.

Conclusions: The model under-weighted negative imaging, at times failing to rule disease out. An automated LLM agent, given a clinically defined target and fixed safety checks, can directly edit an open model's weights to correct such a failure precisely: the edit fixed imaging without disturbing lab tests or general knowledge, and generalized to new patients, wordings and tests. This makes targeted, verifiable repair of clinical reasoning biases practical.
<!-- BODY END -->

## Fact sources (not part of submission; all from paper/main.tex)

- 188 histories, 4 diseases, 7 tests, 2-6 wordings: §2
- 19/19 imaging wordings, CT down 33-68% of ideal, labs no deficit: §3, Table 1
- sigmoid 0.25 vs 2.80, 50% -> 44% (ideal 6%), 48/103: §3
- practice split: 94 histories, CT diverticulitis + CT appendicitis + lipase; held-out new tests: US x2, CT kidney stone, amylase: §4 Protocol
- two rank-1 parts, 0.02%: §4 "The edit"
- imbalance 0.91 -> 0.30 mean |asym|; exp(0.91)=2.48, exp(0.30)=1.35; random 0.89-0.92 -> 2.4-2.5-fold: §4, Fig. 2
- 3 imaging tests unseen (US cholecystitis, US appendicitis, CT kidney stone); wrong-way 79 -> 2: §4
- labs: lipase 0.40 -> 0.40, amylase 0.22 -> 0.28; pending drift 0.13 (guard 0.15); 40/40 general questions; history loss change <= 0.0015: §4
