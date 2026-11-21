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
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.config import DOWN_FLOOR, GENERAL_SUFFIX, GENERAL_YESNO, LIT, PB_RANGE, SUITES, lit_moves
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
        cuda = device.startswith("cuda") or device == "auto"
        # on GPU load straight into GPU memory (needs accelerate) so a 27B model never sits in CPU RAM;
        # device "auto" splits the layers over all visible GPUs (e.g. 27B on 2x A100 40GB)
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
# Question wordings. "orig" is the one every earlier result (Qwen page, harness) uses. The others test
# whether a model reads "Does this patient have X?" as "is X already established?" (-> No while the
# result is pending) rather than "how likely is X?".
QUESTIONS = {
    "orig":   "Does this patient have {d}?",
    "likely": "Is {d} a likely diagnosis for this patient?",
    "could":  "Could this patient have {d}?",
}
Q_RE = re.compile(r"Does this patient have (.*)\?(\n\nAnswer Yes or No only\.)$", re.S)


def reword(text, q):
    m = Q_RE.search(text)
    if not m:
        raise ValueError("prompt does not end with the standard question")
    return text[:m.start()] + QUESTIONS[q].format(d=m.group(1)) + m.group(2)


def build_items(pool_path, limit, question="orig"):
    suite = SUITES["repair"]
    pool = [json.loads(l) for l in open(pool_path)]
    # same patient split as harness/make_splits.py (seed 0) -> mark the 94-patient test half
    rng, test_ids = random.Random(0), set()
    for dz in sorted({c["true_disease"] for c in pool}):
        cs = [c for c in pool if c["true_disease"] == dz]
        rng.shuffle(cs)
        test_ids |= {c["hadm_id"] for c in cs[len(cs) // 2:]}
    if limit:   # stratified subset: round-robin over true diseases, so AUCs are defined
        by = {dz: [c for c in pool if c["true_disease"] == dz] for dz in sorted({c["true_disease"] for c in pool})}
        pool = [by[dz][k] for k in range(max(map(len, by.values()))) for dz in by if k < len(by[dz])][:limit]
    items = [item_from_pool(c, suite, True) for c in pool]
    for it in items:
        it["split"] = "test" if it["hadm_id"] in test_ids else "practice"
        if question != "orig":
            for t in it["tests"].values():
                t["B"] = reword(t["B"], question)
                for pr in t["pairs"].values():
                    pr["P"], pr["N"] = reword(pr["P"], question), reword(pr["N"], question)
    return items


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


def mid_idx(rows, idx, test, pb=PB_RANGE):
    return [i for i in idx if pb[0] <= 1 / (1 + math.exp(-rows[i][test]["B"])) <= pb[1]]


def source(up_r, down_r):
    """Where the imbalance comes from, from move / ideal ratios (1 = literature)."""
    if down_r < 0.8 and up_r <= 1.5:
        return "neg under-used"
    if up_r > 1.5 and down_r >= 0.8:
        return "pos over-used"
    if up_r > 1.5 and down_r < 0.8:
        return "both"
    return "near ideal"


def per_test(rows, items, idx, pb=PB_RANGE, n_boot=1000, seed=0):
    sub_items = [items[i] for i in idx]
    sub_rows = [rows[i] for i in idx]
    mid = {t: mid_idx(sub_rows, range(len(sub_rows)), t, pb) for t in sub_items[0]["tests"]}
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
        out[test]["source"] = source(out[test]["up_over_ideal"], out[test]["down_over_ideal"])
    return out, cells


def history_check(rows, items, pb_range=PB_RANGE):
    """No-result belief: spread, ranking (AUC for the true disease) and LEVEL (calibration vs true diagnoses).
    Level reference = share of pool patients with that diagnosis (the pool is enriched: every patient has
    one of four diseases, so a model calibrated to real ED prevalence should sit BELOW it, but not far below)."""
    out = {}
    for test in items[0]["tests"]:
        b = np.array([r[test]["B"] for r in rows])
        pb = 1 / (1 + np.exp(-b))
        dz = TEST_DISEASE[test]
        y = np.array([it["true_disease"] == dz for it in items])
        a = auc(list(b[y]), list(b[~y])) if dz else float("nan")
        out[test] = {"pB_median": float(np.median(pb)), "pB_iqr": [float(np.percentile(pb, 25)), float(np.percentile(pb, 75))],
                     "m_sd": float(b.std()), "frac_mid": float(np.mean((pb >= pb_range[0]) & (pb <= pb_range[1]))),
                     "auc_true_disease": a, "pB_mean": float(pb.mean()),
                     "prevalence_pool": float(y.mean()) if dz else float("nan"),
                     "pB_median_true": float(np.median(pb[y])) if dz and y.any() else float("nan"),
                     "pB_median_other": float(np.median(pb[~y])) if dz and (~y).any() else float("nan")}
    return out


def prior_bands(rows, items, n_bands=4):
    """Saturation check on ALL cases: split by no-result belief (quantiles); per band the mean move up / down
    (case moves averaged over wordings) and the band's asym. If the lowest band is not compressed (moves and
    asym similar to the higher bands), a low prior does not distort this model and a wider range is safe."""
    out = {}
    for test in items[0]["tests"]:
        lu, ld = lit_moves(test)
        pairs = list(items[0]["tests"][test]["pairs"])
        B = np.array([r[test]["B"] for r in rows])
        up = np.array([np.mean([r[test][p]["P"] - r[test]["B"] for p in pairs]) for r in rows])
        dn = np.array([np.mean([r[test]["B"] - r[test][p]["N"] for p in pairs]) for r in rows])
        raise_ = np.array([np.mean([r[test]["B"] - r[test][p]["N"] < 0 for p in pairs]) for r in rows])
        edges = np.quantile(B, np.linspace(0, 1, n_bands + 1))
        band = np.clip(np.searchsorted(edges, B, side="right") - 1, 0, n_bands - 1)
        bands = []
        for k in range(n_bands):
            s = band == k
            mu, md = float(up[s].mean()), float(dn[s].mean())
            bands.append({"n": int(s.sum()), "pB_median": float(np.median(1 / (1 + np.exp(-B[s])))),
                          "up": mu, "down": md, "up_over_ideal": mu / lu, "down_over_ideal": md / ld,
                          "asym": math.log(max(mu, DOWN_FLOOR) / max(md, DOWN_FLOOR)) - math.log(lu / ld),
                          "neg_raises_frac": float(raise_[s].mean())})
        # correlation of each case's move with its prior (Spearman via ranks)
        rk = lambda x: np.argsort(np.argsort(x))
        out[test] = {"bands": bands,
                     "rho_up_prior": float(np.corrcoef(rk(B), rk(up))[0, 1]),
                     "rho_down_prior": float(np.corrcoef(rk(B), rk(dn))[0, 1])}
    return out


def fmt_bands(pbands):
    L = ["## 1b Does a low starting belief compress the moves? (all patients, quartiles of no-result belief)", "",
         "Each test: four bands from lowest to highest no-result P(Yes). up/ideal, down/ideal and asym per band. "
         "If the lowest band looks like the others, low priors are not saturated on this model and the mid-prior "
         "range can be widened (record it in PIPELINE.md change log). rho = rank correlation of a case's move "
         "with its prior.", "",
         "| test | band | n | median P(Yes) | up/ideal | down/ideal | asym | neg raised |", "|---|---|---|---|---|---|---|---|"]
    for test in IMAGING + LAB:
        if test not in pbands:
            continue
        for k, b in enumerate(pbands[test]["bands"]):
            L.append(f"| {test if k == 0 else ''} | {k + 1} | {b['n']} | {b['pB_median']:.4f} | {b['up_over_ideal']:.2f} | "
                     f"{b['down_over_ideal']:.2f} | {b['asym']:+.2f} | {b['neg_raises_frac']:.2f} |")
        L.append(f"| | rho | | | {pbands[test]['rho_up_prior']:+.2f} | {pbands[test]['rho_down_prior']:+.2f} | | |")
    return L + [""]


def fmt_test_table(pt, title):
    L = [f"### {title}", "",
         "| test | sens/spec | n mid | asym mean [95% CI] | asym range | up/ideal | down/ideal | source | wordings asym>0 | negative raised belief |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for test in IMAGING + LAB:
        r = pt.get(test)
        if not r or not r["n_mid"]:
            L.append(f"| {test} | | 0 | – | | | | | | |")
            continue
        s, sp = LIT[test]
        L.append(f"| {test} | {s:.2f}/{sp:.2f} | {r['n_mid']}/{r['n_total']} | {r['asym_mean']:+.2f} "
                 f"[{r['asym_ci'][0]:+.2f}, {r['asym_ci'][1]:+.2f}] | {r['asym_min']:+.2f} … {r['asym_max']:+.2f} | "
                 f"{r['up_over_ideal']:.2f} | {r['down_over_ideal']:.2f} | {r['source']} | "
                 f"{r['wordings_pos']}/{r['n_wordings']} | "
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
    srcs = [pt[t]["source"] for t in IMAGING if pt.get(t, {}).get("n_mid")]
    checks["imbalance source (imaging)"] = ", ".join(f"{s} {srcs.count(s)}" for s in dict.fromkeys(srcs))
    if im and srcs.count("pos over-used") > len(srcs) / 2:
        dec += (" | NOTE: imbalance comes mainly from OVER-USED POSITIVES, not under-used negatives; "
                "a 'push imaging negatives to No' repair would fix asym but overshoot the negative moves")
    return checks, dec


def run_model(args, items, out):
    """GPU part: score every prompt, write margins.jsonl. Returns rows + sanity info."""
    t0 = time.time()
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
    gold = [a for _, a in GENERAL_YESNO]
    sanity = {
        "token_check": tc,
        "general_acc": float(np.mean([(g["m"] > 0) == (a == "Yes") for g, a in zip(gen, gold)])),
        "mass_all_mean": float(np.mean([r["mass"] for r in res])), "mass_all_min": float(np.min([r["mass"] for r in res])),
        "model_class": type(R.hf).__name__, "n_layers": getattr(R.cfg, "num_hidden_layers", None),
        "softcap": getattr(R.cfg, "final_logit_softcapping", None),
        "top5_B": [res[k].get("top5") for k in first_B], "top5_general": [gen[k].get("top5") for k in (0, 3)],
        "n_prompts": len(texts), "batch": args.batch, "seconds": round(time.time() - t0),
    }
    (out / "sanity.json").write_text(json.dumps(sanity, indent=1))
    return rows, sanity


def load_margins(out, items):
    """Re-analysis without a GPU: rows from margins.jsonl, sanity from sanity.json (or an older report.json)."""
    by_id = {json.loads(l)["hadm_id"]: json.loads(l) for l in open(out / "margins.jsonl")}
    rows = [by_id[it["hadm_id"]]["tests"] for it in items if it["hadm_id"] in by_id]
    items[:] = [it for it in items if it["hadm_id"] in by_id]
    if (out / "sanity.json").exists():
        sanity = json.loads((out / "sanity.json").read_text())
    else:   # first MedGemma run predates sanity.json
        old = json.loads((out / "report.json").read_text())
        sanity = {"token_check": old["token_check"], "general_acc": old["general_acc"]}
    return rows, sanity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--out", default=None, help="default results/diagnose/<model name>")
    ap.add_argument("--batch", type=int, default=8, help="1 = no padding (exact); >1 = left-padded batches")
    ap.add_argument("--limit-cases", type=int, default=0, help="smoke test on the first N patients")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="cuda (one GPU), auto (split over all visible GPUs), or cpu")
    ap.add_argument("--from-margins", action="store_true",
                    help="no GPU: re-analyse <out>/margins.jsonl (e.g. after copying it back)")
    ap.add_argument("--pb-range", type=float, nargs=2, default=list(PB_RANGE),
                    help=f"mid-prior range (default {PB_RANGE}); a non-default value writes REPORT_pb<lo>-<hi>.md")
    ap.add_argument("--question", default="orig", choices=list(QUESTIONS),
                    help="question wording (orig = all earlier results); "
                    + "; ".join(f"{k}: {v}" for k, v in QUESTIONS.items()))
    args = ap.parse_args()
    out = Path(args.out or f"results/diagnose/{args.model.split('/')[-1]}"
               + ("" if args.question == "orig" else f"/q_{args.question}"))
    out.mkdir(parents=True, exist_ok=True)
    pb = tuple(args.pb_range)
    suffix = "" if pb == tuple(PB_RANGE) else f"_pb{pb[0]:g}-{pb[1]:g}"

    items = build_items(args.pool, args.limit_cases, args.question)
    rows, sanity = load_margins(out, items) if args.from_margins else run_model(args, items, out)
    tc, gen_acc = sanity["token_check"], sanity["general_acc"]

    # ---- analysis
    all_idx = list(range(len(items)))
    test_idx = [i for i, it in enumerate(items) if it["split"] == "test"]
    pt_all, cells_all = per_test(rows, items, all_idx, pb)
    pt_test, _ = per_test(rows, items, test_idx, pb) if test_idx else ({}, {})
    hist = history_check(rows, items, pb)
    pbands = prior_bands(rows, items)
    gold = [a for _, a in GENERAL_YESNO]
    mass_B = [r[t]["mass_B"] for r in rows for t in r]
    abs_asym = float(np.mean([abs(c["asym"]) for c in cells_all.values()])) if cells_all else float("nan")

    checks, decision = gate(tc, mass_B, gen_acc, hist, pt_all)
    L = [f"# Stage 1 diagnose: {args.model}  (question: {QUESTIONS[args.question]})", "",
         f"{len(items)} patients ({len(test_idx)} in the harness test half). Mid prior = no-result P(Yes) in {pb}"
         f"{'' if not suffix else ' (NON-DEFAULT; default ' + str(PB_RANGE) + ')'}. "
         "asym = ln(up/down) − ln(ideal up/ideal down); > 0 = positives over-weighted relative to negatives. "
         "source: neg under-used = down/ideal < 0.8; pos over-used = up/ideal > 1.5 with down/ideal ≥ 0.8.", "",
         "## GATE", "", f"**{decision}**", ""] + [f"- {k}: {v}" for k, v in checks.items()] + ["",
         "## 0 Sanity", "",
         f"- model: {sanity.get('model_class', '?')}, {sanity.get('n_layers', '?')} layers, "
         f"final_logit_softcapping={sanity.get('softcap')}",
         f"- Yes/No tokens: {json.dumps(tc)}",
         f"- P(Yes)+P(No) at answer: no-result prompts mean {np.mean(mass_B):.3f} (min {np.min(mass_B):.3f}); "
         f"all prompts mean {sanity.get('mass_all_mean', float('nan')):.3f} (min {sanity.get('mass_all_min', float('nan')):.3f}). "
         "If low, m is not the model's real answer.",
         f"- general yes/no accuracy: {gen_acc:.3f} ({len(gold)} questions)",
         f"- top-5 next tokens, first no-result prompts: {sanity.get('top5_B', 'n/a')}",
         f"- top-5 next tokens, general Q 0 and 3: {sanity.get('top5_general', 'n/a')}", "",
         "## 1 Does it read the history? (no-result prompt)", "",
         "AUC = how well the no-result belief separates patients whose true diagnosis is the asked disease "
         "(0.5 = ignores history).", "",
         "Level: mean P(Yes) vs the share of pool patients who truly have the disease (enriched pool, so a model "
         "calibrated to real ED prevalence should be somewhat below it, not orders of magnitude below).", "",
         "| test | median P(Yes) | IQR | sd of log-odds | fraction mid-prior | AUC true disease | "
         "mean P(Yes) / pool prevalence | median P(Yes) true cases / others |", "|---|---|---|---|---|---|---|---|"]
    for test in IMAGING + LAB:
        h = hist.get(test)
        if h:
            L.append(f"| {test} | {h['pB_median']:.3f} | {h['pB_iqr'][0]:.3f}–{h['pB_iqr'][1]:.3f} | {h['m_sd']:.2f} | "
                     f"{h['frac_mid']:.2f} | {h['auc_true_disease']:.2f} | {h['pB_mean']:.3f} / {h['prevalence_pool']:.3f} | "
                     f"{h['pB_median_true']:.4f} / {h['pB_median_other']:.4f} |")
    L += [""] + fmt_bands(pbands)
    L += ["## 2 Per test (mid-prior cases)", "",
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
    (out / f"REPORT{suffix}.md").write_text("\n".join(L) + "\n")
    (out / f"report{suffix}.json").write_text(json.dumps(
        {"model": args.model, "pb_range": pb, "token_check": tc, "mass_B_mean": float(np.mean(mass_B)),
         "general_acc": gen_acc, "history": hist, "prior_bands": pbands, "per_test_all": pt_all,
         "per_test_testhalf": pt_test, "cells_all": cells_all,
         "abs_asym_all": abs_asym, "gate": {"checks": checks, "decision": decision}}, indent=1, default=str))
    print(f"GATE: {decision}\n-> {out / f'REPORT{suffix}.md'}", flush=True)


if __name__ == "__main__":
    main()
