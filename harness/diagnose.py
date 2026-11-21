"""PIPELINE STAGE 1 (see PIPELINE.md): does model X have the positive-evidence bias?  Behaviour only.

Same prompts, literature yardsticks and asym definition as the repair harness (harness/config.py,
suite "repair"), but run on ALL 188 patients x ALL tests / wordings (nothing has been fitted on the
new model, so there is no held-out issue). Model-specific parts are generic: chat template, Yes/No
token ids, multimodal checkpoints (MedGemma 4B), left-padded batching. Ends with an automatic GATE
verdict (GATE below): proceed to stage 2 (repair) or not.

  python -m harness.diagnose --model google/medgemma-1.5-4b-it --out /scratch/dlf8982/diagnose/medgemma-1.5-4b-it
  python -m harness.diagnose --model Qwen/Qwen3.5-9B        # reference model, same script

Output: results/diagnose/<model>/{REPORT.md, report.json, margins.jsonl}
margins.jsonl holds only ids + numbers (no patient text). results/ is git-ignored anyway.

Sections of REPORT.md (what to look at, in order):
  0 sanity     Yes/No are single tokens, P(Yes)+P(No) at the answer is high, 40 general yes/no correct
  1 history    does the model read the history at all (prior spread, AUC of the no-result belief
               for the patient's true disease) -- Qwen 0.8B failed here
  2 per test   asym, up/ideal, down/ideal, negatives that RAISED belief; all 188 and the 94-patient
               test half (the half the Qwen page reports)
  3 per cell   test x wording
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.config import GENERAL_SUFFIX, GENERAL_YESNO, LIT, PB_RANGE, SUITES, lit_moves
from harness.core import cell_stats
from harness.make_splits import item_from_pool

# Stage-1 gate (fixed across models; change only with a written reason in PIPELINE.md)
GATE = {
    "min_yesno_mass": 0.90,    # P(Yes)+P(No) at the answer, mean over no-result prompts
    "min_general_acc": 0.85,   # 40 general yes/no questions
    "min_auc": 0.65,           # median over tests of AUC(no-result belief, true disease): reads history
    "min_mid": 30,             # mid-prior cases per test, else that test is under-powered
}

IMAGING = ["ct_sigmoid", "ct_kidney_stone", "ct_appendix", "us_gallbladder", "us_appendix"]
LAB = ["lipase", "amylase"]
TEST_DISEASE = {"ct_sigmoid": "diverticulitis", "ct_appendix": "appendicitis", "us_appendix": "appendicitis",
                "us_gallbladder": "cholecystitis", "lipase": "pancreatitis", "amylase": "pancreatitis",
                "ct_kidney_stone": None}


# ---- model ---------------------------------------------------------------------------------
class Runner:
    def __init__(self, name, device, batch):
        self.batch = batch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.proc = None
        if not self.tok.chat_template:             # multimodal checkpoints keep the template on the processor
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(name)
        cuda = device.startswith("cuda")
        # on GPU load straight into GPU memory (needs accelerate) so a 27B model never sits in CPU RAM
        kw = {"dtype": torch.bfloat16 if cuda else torch.float32, **({"device_map": device} if cuda else {})}
        try:
            self.hf = AutoModelForCausalLM.from_pretrained(name, **kw)
        except ValueError as e:
            if "Unrecognized configuration" not in str(e):
                raise
            from transformers import AutoModelForImageTextToText
            self.hf = AutoModelForImageTextToText.from_pretrained(name, **kw)
        if not cuda:
            self.hf.to(device)
        self.hf.eval()
        self.device = next(self.hf.parameters()).device
        self.cfg = getattr(self.hf.config, "text_config", self.hf.config)
        self.yn = self.tok.convert_tokens_to_ids(["Yes", "No"])
        self.tok.padding_side = "left"
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def chat(self, text):
        if self.proc is not None:
            return self.proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": text}]}],
                tokenize=False, add_generation_prompt=True)
        kw = {"enable_thinking": False} if "enable_thinking" in self.tok.chat_template else {}
        return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True, **kw)

    def token_check(self):
        out = {}
        for w, i in zip(["Yes", "No"], self.yn):
            enc = self.tok(w, add_special_tokens=False).input_ids
            out[w] = {"id": i, "unk": i == self.tok.unk_token_id, "encodes_as": enc, "single": enc == [i]}
        return out

    @torch.no_grad()
    def margins(self, texts, topk_for=()):
        """m = logit(Yes) - logit(No) at the answer position, P(Yes)+P(No), optional top-5 tokens."""
        ids = [self.tok(self.chat(t), add_special_tokens=False).input_ids for t in texts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        res = [None] * len(ids)
        y, n = self.yn
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            L = max(len(ids[i]) for i in idx)
            x = torch.full((len(idx), L), self.pad, dtype=torch.long)
            am = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                x[r, L - len(ids[i]):] = torch.tensor(ids[i])
                am[r, L - len(ids[i]):] = 1
            x, am = x.to(self.device), am.to(self.device)
            try:
                lg = self.hf(input_ids=x, attention_mask=am, logits_to_keep=1).logits[:, -1].float()
            except TypeError:
                lg = self.hf(input_ids=x, attention_mask=am).logits[:, -1].float()
            p = torch.softmax(lg, -1)
            for r, i in enumerate(idx):
                d = {"m": (lg[r, y] - lg[r, n]).item(), "mass": (p[r, y] + p[r, n]).item(), "len": len(ids[i])}
                if i in topk_for:
                    tp = torch.topk(p[r], 5)
                    d["top5"] = [(self.tok.decode([int(t)]), round(float(v), 4)) for v, t in zip(tp.values, tp.indices)]
                res[i] = d
        return res


# ---- data ------------------------------------------------------------------------------------
def build_items(pool_path, limit):
    suite = SUITES["repair"]
    pool = [json.loads(l) for l in open(pool_path)]
    # same patient split as harness/make_splits.py (seed 0) -> mark the 94-patient test half
    rng, test_ids = random.Random(0), set()
    for dz in sorted({c["true_disease"] for c in pool}):
        cs = [c for c in pool if c["true_disease"] == dz]
        rng.shuffle(cs)
        test_ids |= {c["hadm_id"] for c in cs[len(cs) // 2:]}
    items = [item_from_pool(c, suite, True) for c in pool]
    for it in items:
        it["split"] = "test" if it["hadm_id"] in test_ids else "practice"
    return items[:limit] if limit else items


def flatten(items):
    keys, texts = [], []
    for i, it in enumerate(items):
        for test, t in it["tests"].items():
            keys.append((i, test, "B", None)); texts.append(t["B"])
            for pair, pr in t["pairs"].items():
                for side in ("P", "N"):
                    keys.append((i, test, pair, side)); texts.append(pr[side])
    return keys, texts


def unflatten(items, keys, res):
    rows = [{} for _ in items]
    for (i, test, pair, side), r in zip(keys, res):
        row = rows[i].setdefault(test, {})
        if pair == "B":
            row["B"], row["mass_B"] = r["m"], r["mass"]
        else:
            row.setdefault(pair, {})[side] = r["m"]
    return rows


# ---- analysis --------------------------------------------------------------------------------
def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def mid_idx(rows, idx, test):
    return [i for i in idx if PB_RANGE[0] <= 1 / (1 + math.exp(-rows[i][test]["B"])) <= PB_RANGE[1]]


def per_test(rows, items, idx, n_boot=1000, seed=0):
    sub_items = [items[i] for i in idx]
    sub_rows = [rows[i] for i in idx]
    mid = {t: mid_idx(sub_rows, range(len(sub_rows)), t) for t in sub_items[0]["tests"]}
    cells = cell_stats(sub_rows, sub_items, mid)
    rng = np.random.default_rng(seed)
    out = {}
    for test in sub_items[0]["tests"]:
        cs = {k.split("/")[1]: c for k, c in cells.items() if k.split("/")[0] == test}
        if not cs:
            out[test] = {"n_mid": 0}
            continue
        lu, ld = lit_moves(test)
        # bootstrap over mid-prior cases: mean over wordings of asym
        m = mid[test]
        boots = []
        for _ in range(n_boot):
            b = list(rng.choice(m, size=len(m), replace=True))
            cb = cell_stats(sub_rows, sub_items, {**{t: [] for t in mid}, test: b})
            boots.append(np.mean([c["asym"] for c in cb.values()]))
        out[test] = {
            "n_mid": len(m), "n_total": len(sub_rows),
            "asym_mean": float(np.mean([c["asym"] for c in cs.values()])),
            "asym_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "asym_min": min(c["asym"] for c in cs.values()), "asym_max": max(c["asym"] for c in cs.values()),
            "up_over_ideal": float(np.mean([c["up"] for c in cs.values()]) / lu),
            "down_over_ideal": float(np.mean([c["down"] for c in cs.values()]) / ld),
            "wordings_pos": sum(c["asym"] > 0 for c in cs.values()), "n_wordings": len(cs),
            "neg_raises": sum(c["neg_raises"] for c in cs.values()),
            "neg_pairs": sum(c["n"] for c in cs.values()),
        }
    return out, cells


def history_check(rows, items):
    out = {}
    for test in items[0]["tests"]:
        b = np.array([r[test]["B"] for r in rows])
        pb = 1 / (1 + np.exp(-b))
        dz = TEST_DISEASE[test]
        a = (auc([b[i] for i, it in enumerate(items) if it["true_disease"] == dz],
                 [b[i] for i, it in enumerate(items) if it["true_disease"] != dz]) if dz else float("nan"))
        out[test] = {"pB_median": float(np.median(pb)), "pB_iqr": [float(np.percentile(pb, 25)), float(np.percentile(pb, 75))],
                     "m_sd": float(b.std()), "frac_mid": float(np.mean((pb >= PB_RANGE[0]) & (pb <= PB_RANGE[1]))),
                     "auc_true_disease": a}
    return out


def fmt_test_table(pt, title):
    L = [f"### {title}", "",
         "| test | sens/spec | n mid | asym mean [95% CI] | asym range | up/ideal | down/ideal | wordings asym>0 | negative raised belief |",
         "|---|---|---|---|---|---|---|---|---|"]
    for test in IMAGING + LAB:
        r = pt.get(test)
        if not r or not r["n_mid"]:
            L.append(f"| {test} | | 0 | – | | | | | |")
            continue
        s, sp = LIT[test]
        L.append(f"| {test} | {s:.2f}/{sp:.2f} | {r['n_mid']}/{r['n_total']} | {r['asym_mean']:+.2f} "
                 f"[{r['asym_ci'][0]:+.2f}, {r['asym_ci'][1]:+.2f}] | {r['asym_min']:+.2f} … {r['asym_max']:+.2f} | "
                 f"{r['up_over_ideal']:.2f} | {r['down_over_ideal']:.2f} | {r['wordings_pos']}/{r['n_wordings']} | "
                 f"{r['neg_raises']}/{r['neg_pairs']} |")
    im = [pt[t] for t in IMAGING if pt.get(t, {}).get("n_mid")]
    lb = [pt[t] for t in LAB if pt.get(t, {}).get("n_mid")]
    L += ["", f"- imaging wordings with asym > 0: {sum(r['wordings_pos'] for r in im)}/{sum(r['n_wordings'] for r in im)}"
              f"; mean asym {np.mean([r['asym_mean'] for r in im]):+.2f}" if im else "",
          f"- lab mean asym {np.mean([r['asym_mean'] for r in lb]):+.2f}" if lb else "", ""]
    return L


def gate(tc, mass_B, gen_acc, hist, pt):
    """Stage-1 verdict. Returns (checks, decision)."""
    aucs = [h["auc_true_disease"] for h in hist.values() if not math.isnan(h["auc_true_disease"])]
    im = [pt[t] for t in IMAGING if pt.get(t, {}).get("n_mid")]
    lb = [pt[t] for t in LAB if pt.get(t, {}).get("n_mid")]
    checks = {
        "Yes/No single tokens": all(v["single"] and not v["unk"] for v in tc.values()),
        f"yes/no mass >= {GATE['min_yesno_mass']}": float(np.mean(mass_B)) >= GATE["min_yesno_mass"],
        f"general accuracy >= {GATE['min_general_acc']}": gen_acc >= GATE["min_general_acc"],
        f"reads history (median AUC >= {GATE['min_auc']})": bool(aucs) and float(np.median(aucs)) >= GATE["min_auc"],
        f"every test has >= {GATE['min_mid']} mid-prior cases": all(pt.get(t, {}).get("n_mid", 0) >= GATE["min_mid"]
                                                                  for t in IMAGING + LAB),
    }
    valid = all(checks.values())
    # bias = most imaging tests have a 95% CI above 0
    im_pos = sum(r["asym_ci"][0] > 0 for r in im)
    im_neg = sum(r["asym_ci"][1] < 0 for r in im)
    lab_mean = float(np.mean([r["asym_mean"] for r in lb])) if lb else float("nan")
    if not valid:
        dec = "STOP: measurement not valid on this model (see failed checks); fix the setup before reading asym"
    elif im and im_pos > len(im) / 2 and not lab_mean > 0.3:
        dec = (f"PROCEED to stage 2 (repair): same bias pattern ({im_pos}/{len(im)} imaging tests CI > 0, "
               f"lab mean asym {lab_mean:+.2f})")
    elif im and im_pos > len(im) / 2:
        dec = (f"DISCUSS: imaging biased ({im_pos}/{len(im)}) but labs too (mean {lab_mean:+.2f}); "
               "different pattern from Qwen, the round-6 recipe (lab = control) does not apply as is")
    elif im_neg > len(im) / 2:
        dec = f"DISCUSS: reversed bias ({im_neg}/{len(im)} imaging tests CI < 0)"
    else:
        dec = f"NO BIAS: only {im_pos}/{len(im)} imaging tests CI > 0; report as a negative result, no repair"
    checks["bias: imaging tests with CI > 0"] = f"{im_pos}/{len(im)}"
    checks["lab mean asym"] = round(lab_mean, 3)
    return checks, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--out", default=None, help="default results/diagnose/<model name>")
    ap.add_argument("--batch", type=int, default=8, help="1 = no padding (exact); >1 = left-padded batches")
    ap.add_argument("--limit-cases", type=int, default=0, help="smoke test on the first N patients")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out or f"results/diagnose/{args.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    items = build_items(args.pool, args.limit_cases)
    keys, texts = flatten(items)
    print(f"{len(items)} patients, {len(texts)} prompts; loading {args.model}", flush=True)
    R = Runner(args.model, args.device, args.batch)
    tc = R.token_check()
    print(f"loaded ({time.time() - t0:.0f}s); Yes/No tokens {tc}", flush=True)

    first_B = [k for k, key in enumerate(keys) if key[2] == "B"][:3]
    res = R.margins(texts, topk_for=set(first_B))
    print(f"scored ({time.time() - t0:.0f}s)", flush=True)
    gen = R.margins([q + GENERAL_SUFFIX for q, _ in GENERAL_YESNO], topk_for={0, 3})
    rows = unflatten(items, keys, res)

    with (out / "margins.jsonl").open("w") as f:
        for it, r in zip(items, rows):
            f.write(json.dumps({"hadm_id": it["hadm_id"], "true_disease": it["true_disease"],
                                "split": it["split"], "tests": r}) + "\n")

    # ---- analysis
    all_idx = list(range(len(items)))
    test_idx = [i for i, it in enumerate(items) if it["split"] == "test"]
    pt_all, cells_all = per_test(rows, items, all_idx)
    pt_test, _ = per_test(rows, items, test_idx) if test_idx else ({}, {})
    hist = history_check(rows, items)
    gold = [a for _, a in GENERAL_YESNO]
    gen_acc = float(np.mean([(g["m"] > 0) == (a == "Yes") for g, a in zip(gen, gold)]))
    mass_B = [r["mass"] for k, r in zip(keys, res) if k[2] == "B"]
    mass_all = [r["mass"] for r in res]
    abs_asym = float(np.mean([abs(c["asym"]) for c in cells_all.values()])) if cells_all else float("nan")

    checks, decision = gate(tc, mass_B, gen_acc, hist, pt_all)
    L = [f"# Stage 1 diagnose: {args.model}", "",
         f"{len(items)} patients ({len(test_idx)} in the harness test half), {len(texts)} prompts, batch {args.batch}, "
         f"{time.time() - t0:.0f}s. Mid prior = no-result P(Yes) in {PB_RANGE}. "
         "asym = ln(up/down) − ln(ideal up/ideal down); > 0 = positives over-weighted relative to negatives.", "",
         "## GATE", "", f"**{decision}**", ""] + [f"- {k}: {v}" for k, v in checks.items()] + ["",
         "## 0 Sanity", "",
         f"- model: {type(R.hf).__name__}, {getattr(R.cfg, 'num_hidden_layers', '?')} layers, "
         f"final_logit_softcapping={getattr(R.cfg, 'final_logit_softcapping', None)}",
         f"- Yes/No tokens: {json.dumps(tc)}",
         f"- P(Yes)+P(No) at answer: no-result prompts mean {np.mean(mass_B):.3f} (min {np.min(mass_B):.3f}); "
         f"all prompts mean {np.mean(mass_all):.3f} (min {np.min(mass_all):.3f}). If low, m is not the model's real answer.",
         f"- general yes/no accuracy: {gen_acc:.3f} ({len(gold)} questions)",
         "- top-5 next tokens, first no-result prompts: " + "; ".join(str(res[k].get("top5")) for k in first_B),
         "- top-5 next tokens, general Q 0 and 3: " + "; ".join(str(gen[k].get("top5")) for k in (0, 3)), "",
         "## 1 Does it read the history? (no-result prompt)", "",
         "AUC = how well the no-result belief separates patients whose true diagnosis is the asked disease "
         "(0.5 = ignores history).", "",
         "| test | median P(Yes) | IQR | sd of log-odds | fraction mid-prior | AUC true disease |", "|---|---|---|---|---|---|"]
    for test in IMAGING + LAB:
        h = hist.get(test)
        if h:
            L.append(f"| {test} | {h['pB_median']:.3f} | {h['pB_iqr'][0]:.3f}–{h['pB_iqr'][1]:.3f} | {h['m_sd']:.2f} | "
                     f"{h['frac_mid']:.2f} | {h['auc_true_disease']:.2f} |")
    L += ["", "## 2 Per test", "",
          "up/ideal and down/ideal: model's mean move ÷ literature move (1 = exact). "
          "'negative raised belief' = mid-prior (case, wording) pairs where the negative moved belief toward Yes.", ""]
    L += fmt_test_table(pt_all, f"All {len(items)} patients")
    if pt_test:
        L += fmt_test_table(pt_test, f"Harness test half ({len(test_idx)} patients; comparable to the Qwen page §1.3)")
    L += [f"Harness score on all patients (mean |asym| over cells): {abs_asym:.3f}", "",
          "## 3 Per cell (all patients)", "",
          "| cell | n | up | down | ideal up/down | asym | neg raised |", "|---|---|---|---|---|---|---|"]
    for k, c in cells_all.items():
        L.append(f"| {k} | {c['n']} | {c['up']:.2f} | {c['down']:.2f} | {c['lit_up']:.2f}/{c['lit_down']:.2f} | "
                 f"{c['asym']:+.2f} | {c['neg_raises']} |")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(
        {"model": args.model, "token_check": tc, "mass_B_mean": float(np.mean(mass_B)), "general_acc": gen_acc,
         "history": hist, "per_test_all": pt_all, "per_test_testhalf": pt_test, "cells_all": cells_all,
         "abs_asym_all": abs_asym, "gate": {"checks": checks, "decision": decision}}, indent=1, default=str))
    print(f"GATE: {decision}\n-> {out / 'REPORT.md'} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
