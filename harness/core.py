"""Model wrapper, editing with rollback, scoring and guards. Shared by server.py and final_eval.py."""
import json
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.config import (DOWN_FLOOR, FORBIDDEN, GENERAL_SUFFIX, GENERAL_YESNO, MAX_NNZ,
                            NLL_MAX_TOKENS, NLL_TEXTS, PB_RANGE, lit_moves)


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def load_items(path, limit=0):
    items = [json.loads(l) for l in open(path)]
    return items[:limit] if limit else items


class Model:
    def __init__(self, name, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
        self.yn = self.tok.convert_tokens_to_ids(["Yes", "No"])
        self.params = dict(self.hf.named_parameters())
        self.orig = {}          # name -> CPU copy of the ORIGINAL weight (filled on first edit)
        self.active = []        # names currently carrying an edit
        self._ids = {}

    # ---- forward passes ------------------------------------------------------------
    def ids(self, text, raw=False):
        key = (text, raw)
        if key not in self._ids:
            s = text if raw else chat(self.tok, text)
            self._ids[key] = self.tok(s, add_special_tokens=False).input_ids
        return self._ids[key]

    @torch.no_grad()
    def yesno(self, ids):
        lg = self.hf(torch.tensor([ids], device=self.device), logits_to_keep=1).logits[0, -1].float()
        p = torch.softmax(lg, -1)
        y, n = self.yn
        return {"m": (lg[y] - lg[n]).item(), "mass": (p[y] + p[n]).item(), "logits": lg}

    @torch.no_grad()
    def nll(self, text):
        ids = self.tok(text, add_special_tokens=False).input_ids[:NLL_MAX_TOKENS]
        x = torch.tensor([ids], device=self.device)
        lg = self.hf(x).logits[0, :-1].float()
        return torch.nn.functional.cross_entropy(lg, x[0, 1:]).item()

    # ---- editing ---------------------------------------------------------------------
    def original(self, name):
        return self.orig[name] if name in self.orig else self.params[name].detach().cpu()

    def validate(self, deltas):
        nnz = 0
        for n, d in deltas.items():
            if n not in self.params:
                raise ValueError(f"unknown parameter {n!r} (see model_info.json for names)")
            if FORBIDDEN.search(n):
                raise ValueError(f"parameter {n!r} is off limits (embedding / lm_head / final norm / vision)")
            if tuple(d.shape) != tuple(self.params[n].shape):
                raise ValueError(f"{n}: delta shape {tuple(d.shape)} != weight {tuple(self.params[n].shape)}")
            if not torch.isfinite(d).all():
                raise ValueError(f"{n}: delta has NaN/inf")
            nnz += int((d != 0).sum())
        if nnz > MAX_NNZ:
            raise ValueError(f"edit changes {nnz:,} weights > limit {MAX_NNZ:,}")
        return nnz

    @torch.no_grad()
    def reset(self):
        for n in self.active:
            self.params[n].data.copy_(self.orig[n].to(self.device))
        self.active = []

    @torch.no_grad()
    def apply(self, deltas, scale=1.0):
        """Replace any current edit by W = W_orig + scale * delta. Returns stats."""
        nnz = self.validate(deltas)
        self.reset()
        eff, rel = 0, []
        for n, d in deltas.items():
            p = self.params[n]
            if n not in self.orig:
                self.orig[n] = p.detach().to("cpu", copy=True)
            w0 = self.orig[n].to(self.device, torch.float32)
            p.data.copy_((w0 + scale * d.to(self.device, torch.float32)).to(p.dtype))
            self.active.append(n)
            eff += int((p.data != self.orig[n].to(self.device)).sum())
            rel.append((n, (scale * d.float()).norm().item() / max(w0.norm().item(), 1e-12)))
        return {"nnz": nnz, "effective_changed": eff, "relative_norm": rel}


def load_edit(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    deltas = obj["deltas"] if isinstance(obj, dict) and "deltas" in obj else obj
    if not isinstance(deltas, dict) or not all(torch.is_tensor(v) for v in deltas.values()):
        raise ValueError("edit file must be {param_name: delta_tensor} or {'deltas': {...}}")
    return deltas


def random_control(deltas, seed=0):
    """Same parameters, same support, same Frobenius norm per parameter, random content."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, d in deltas.items():
        r = torch.randn(d.shape, generator=g) * (d != 0)
        out[n] = r * (d.float().norm() / r.norm().clamp_min(1e-12))
    return out


# ---- scoring ----------------------------------------------------------------------------
def run_items(model, items):
    """margins[i][test] = {"B": m, "mass_B": .., pair: {"P": m, "N": m}}"""
    out = []
    for it in items:
        row = {}
        for test, t in it["tests"].items():
            b = model.yesno(model.ids(t["B"]))
            r = {"B": b["m"], "mass_B": b["mass"]}
            for pair, pr in t["pairs"].items():
                r[pair] = {"P": model.yesno(model.ids(pr["P"]))["m"],
                           "N": model.yesno(model.ids(pr["N"]))["m"]}
            row[test] = r
        out.append(row)
    return out


def run_guards(model, items):
    gen = [model.yesno(model.ids(q + GENERAL_SUFFIX)) for q, _ in GENERAL_YESNO]
    texts = [it["history"] for it in items[:NLL_TEXTS]]
    return {"general_m": [g["m"] for g in gen], "general_mass": [g["mass"] for g in gen],
            "nll": [model.nll(t) for t in texts]}


def mid_cases(ref, items):
    """Per test: case indices whose ORIGINAL baseline P(Yes) is in PB_RANGE (fixed once)."""
    mid = {}
    for test in items[0]["tests"]:
        mid[test] = [i for i, r in enumerate(ref)
                     if PB_RANGE[0] <= 1 / (1 + math.exp(-r[test]["B"])) <= PB_RANGE[1]]
    return mid


def cell_stats(m, items, mid):
    cells = {}
    for test in items[0]["tests"]:
        lu, ld = lit_moves(test)
        idx = mid[test]
        if not idx:
            continue
        for pair in items[0]["tests"][test]["pairs"]:
            up = np.array([m[i][test][pair]["P"] - m[i][test]["B"] for i in idx])
            dn = np.array([m[i][test]["B"] - m[i][test][pair]["N"] for i in idx])
            mu, md = float(up.mean()), float(dn.mean())
            asym = math.log(max(mu, DOWN_FLOOR) / max(md, DOWN_FLOOR)) - math.log(lu / ld)
            cells[f"{test}/{pair}"] = {
                "n": len(idx), "up": mu, "down": md, "lit_up": lu, "lit_down": ld, "asym": asym,
                "neg_raises": int((dn < 0).sum()), "pos_lowers": int((up < 0).sum())}
    return cells


def report(cur, ref, g_cur, g_ref, items, mid, worst=8):
    """Everything the researcher sees after a score call (dict + plain-text table)."""
    c_now, c_ref = cell_stats(cur, items, mid), cell_stats(ref, items, mid)
    drift = float(np.mean([abs(cur[i][t]["B"] - ref[i][t]["B"])
                           for i in range(len(items)) for t in items[i]["tests"]]))
    mass = float(np.mean([cur[i][t]["mass_B"] for i in range(len(items)) for t in items[i]["tests"]]))
    gold = [a for _, a in GENERAL_YESNO]
    acc = lambda ms: float(np.mean([(mm > 0) == (a == "Yes") for mm, a in zip(ms, gold)]))
    head = {
        "asym_mean": float(np.mean([c["asym"] for c in c_now.values()])),
        "asym_mean_orig": float(np.mean([c["asym"] for c in c_ref.values()])),
        "move_err": float(np.mean([abs(c["up"] - c["lit_up"]) + abs(c["down"] - c["lit_down"])
                                   for c in c_now.values()])),
        "move_err_orig": float(np.mean([abs(c["up"] - c["lit_up"]) + abs(c["down"] - c["lit_down"])
                                        for c in c_ref.values()])),
        "neg_raises": sum(c["neg_raises"] for c in c_now.values()),
        "neg_raises_orig": sum(c["neg_raises"] for c in c_ref.values()),
        "baseline_drift": drift,
        "yesno_mass": mass,
        "general_acc": acc(g_cur["general_m"]), "general_acc_orig": acc(g_ref["general_m"]),
        "general_shift": float(np.mean(np.array(g_cur["general_m"]) - np.array(g_ref["general_m"]))),
        "nll": float(np.mean(g_cur["nll"])), "nll_orig": float(np.mean(g_ref["nll"])),
    }
    # concrete failures: the worst negative moves (mid prior), so they can be read one by one
    fails = []
    for test, idx in mid.items():
        for pair in items[0]["tests"][test]["pairs"]:
            for i in idx:
                d = cur[i][test]["B"] - cur[i][test][pair]["N"]
                fails.append({"case": i, "hadm_id": items[i]["hadm_id"],
                              "true_disease": items[i]["true_disease"], "test": test, "pair": pair,
                              "down": round(d, 2),
                              "down_orig": round(ref[i][test]["B"] - ref[i][test][pair]["N"], 2)})
    fails.sort(key=lambda x: x["down"])

    L = ["HEADLINE (now vs original)",
         f"  asym_mean      {head['asym_mean']:+.3f}  (orig {head['asym_mean_orig']:+.3f}; 0 = literature ratio, >0 = positive bias)",
         f"  move_err       {head['move_err']:.3f}  (orig {head['move_err_orig']:.3f}; |up-lit|+|down-lit|)",
         f"  neg_raises     {head['neg_raises']}  (orig {head['neg_raises_orig']}; negative result RAISED belief)",
         "GUARDS",
         f"  baseline_drift {head['baseline_drift']:.3f}  (mean |change| of no-result log-odds; keep small)",
         f"  yesno_mass     {head['yesno_mass']:.3f}  (P(Yes)+P(No) at answer)",
         f"  general_acc    {head['general_acc']:.3f}  (orig {head['general_acc_orig']:.3f}; {len(gold)} general yes/no)",
         f"  general_shift  {head['general_shift']:+.3f}  (mean change of general yes-no log-odds; >0 = toward Yes)",
         f"  history_nll    {head['nll']:.4f}  (orig {head['nll_orig']:.4f})",
         "CELLS (mid-prior cases; up/down in log-odds; lit = literature ideal)",
         f"  {'cell':32s} {'n':>4s} {'up':>13s} {'down':>13s} {'lit up/down':>12s} {'asym':>13s} {'neg_raises':>10s}"]
    for k, c in c_now.items():
        r = c_ref[k]
        L.append(f"  {k:32s} {c['n']:4d} {r['up']:5.2f}->{c['up']:5.2f} {r['down']:5.2f}->{c['down']:5.2f} "
                 f"{c['lit_up']:5.2f}/{c['lit_down']:4.2f} {r['asym']:+5.2f}->{c['asym']:+5.2f} "
                 f"{r['neg_raises']:4d}->{c['neg_raises']:<4d}")
    L.append(f"WORST NEGATIVE MOVES (case = index into the item file)")
    for f in fails[:worst]:
        L.append(f"  case {f['case']:3d} {f['true_disease']:14s} {f['test']}/{f['pair']:16s} "
                 f"down {f['down_orig']:+.2f} -> {f['down']:+.2f}")
    return {"headline": head, "cells": c_now, "cells_orig": c_ref, "worst": fails[:worst]}, "\n".join(L)
