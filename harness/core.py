"""Model wrapper, editing with rollback, scoring and guards. Shared by server.py and final_eval.py.

Diagnostic interventions (forward hooks, no weight change, removable) -- see Model.set_interventions:
  add            h[pos] += alpha * v                        at a module output (default: residual after layer L)
  project        h[pos] -= strength * ((h.v^) - center) v^  (remove / clamp a direction)
  patch          h[pos] = same module, same position-from-end, in a SOURCE prompt run (per-prompt patch_from)
  ablate_neurons MLP neurons of layer L scaled by (1 - scale) (input of down_proj)
  ablate_heads   attention heads of a full-attention layer L scaled by (1 - scale) (input of o_proj)
  freeze         listed modules output what they output on the ORIGINAL model (no weight edit, no
                 interventions) for the same prompt -> tests self-repair / compensation.
"""
import json
import math
import re

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
        self.edited = {}        # name -> device copy of the EDITED weight (to swap back after a freeze run)
        self.active = []        # names currently carrying an edit
        self._ids = {}
        # interventions
        self.layers_prefix = next(n[:-2] for n, _ in self.hf.named_modules() if re.fullmatch(r".*\.layers\.0", n))
        self.cfg = getattr(self.hf.config, "text_config", self.hf.config)
        self.iv_specs, self.freeze_mods, self._iv_hooks = [], [], []
        self._mode = "on"       # "on" | "capture_patch" | "capture_freeze"
        self._patch_src, self._freeze_src = {}, {}

    # ---- interventions -------------------------------------------------------------
    def layer_module(self, layer, sub=""):
        return f"{self.layers_prefix}.{layer}" + (f".{sub}" if sub else "")

    def _vector(self, s):
        v = s["vector"]
        if isinstance(v, str):
            obj = torch.load(v, map_location="cpu", weights_only=False)
            v = obj[s["key"]] if isinstance(obj, dict) else obj
        return torch.as_tensor(v, dtype=torch.float32).flatten().to(self.device)

    @staticmethod
    def _positions(pos, T):
        if pos == "all":
            return list(range(T))
        if pos == "last":
            return [T - 1]
        return [p if p >= 0 else T + p for p in pos]

    def _out_hook(self, k, s):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if self._mode == "capture_patch":
                if s["type"] == "patch":
                    self._patch_src[k] = h[0].detach().clone()
                return None
            if self._mode != "on":
                return None
            h2 = h.clone()
            T = h2.shape[1]
            if s["type"] == "patch":
                src = self._patch_src[k]
                pos = s.get("positions", "last")
                for p in ([-1] if pos == "last" else pos):
                    h2[0, T + p] = src[src.shape[0] + p]
            else:
                idx = self._positions(s.get("positions", "last"), T)
                x = h2[:, idx].float()
                if s["type"] == "add":
                    x = x + s["alpha"] * s["_v"]
                else:  # project
                    c = x @ s["_vhat"]
                    x = x - s.get("strength", 1.0) * (c - s.get("center", 0.0))[..., None] * s["_vhat"]
                h2[:, idx] = x.to(h2.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    def _ablate_hook(self, idx, scale):
        def pre(mod, args):
            if self._mode != "on":
                return None
            x = args[0].clone()
            x[..., idx] = x[..., idx] * (1 - scale)
            return (x,) + tuple(args[1:])
        return pre

    def _freeze_hook(self, name):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if self._mode == "capture_freeze":
                self._freeze_src[name] = h.detach().clone()
                return None
            if self._mode != "on":
                return None
            src = self._freeze_src[name]
            return (src,) + tuple(out[1:]) if isinstance(out, tuple) else src
        return hook

    def clear_interventions(self):
        for h in self._iv_hooks:
            h.remove()
        self.iv_specs, self.freeze_mods, self._iv_hooks = [], [], []
        self._patch_src, self._freeze_src = {}, {}

    def set_interventions(self, specs, freeze=()):
        """Replace any active interventions. Returns one description line per spec."""
        self.clear_interventions()
        desc = []
        try:
            for k, s in enumerate(specs):
                s, t = dict(s), s["type"]
                if t in ("add", "project", "patch"):
                    pos = s.get("positions", "last")
                    if t == "patch" and pos != "last" and any(p >= 0 for p in pos):
                        raise ValueError("patch positions must be 'last' or negative ints (counted from the end)")
                    name = s.get("module") or self.layer_module(s["layer"])
                    mod = self.hf.get_submodule(name)
                    if t != "patch":
                        s["_v"] = self._vector(s)
                        s["_vhat"] = s["_v"] / s["_v"].norm().clamp_min(1e-12)
                    self._iv_hooks.append(mod.register_forward_hook(self._out_hook(k, s)))
                    extra = {"add": lambda: f"alpha={s['alpha']}, |v|={s['_v'].norm():.3g}",
                             "project": lambda: f"strength={s.get('strength', 1.0)}, center={s.get('center', 0.0)}",
                             "patch": lambda: "source = each prompt's patch_from"}[t]()
                    desc.append(f"{t} @ {name} positions={pos} ({extra})")
                elif t == "ablate_neurons":
                    name = self.layer_module(s["layer"], "mlp.down_proj")
                    mod = self.hf.get_submodule(name)
                    self._iv_hooks.append(mod.register_forward_pre_hook(
                        self._ablate_hook(list(s["neurons"]), s.get("scale", 1.0))))
                    desc.append(f"ablate {len(s['neurons'])} MLP neurons of layer {s['layer']} x{s.get('scale', 1.0)}")
                elif t == "ablate_heads":
                    name = self.layer_module(s["layer"], "self_attn.o_proj")
                    try:
                        mod = self.hf.get_submodule(name)
                    except AttributeError:
                        raise ValueError(f"layer {s['layer']} has no self_attn.o_proj (not a full-attention layer)")
                    hd = mod.in_features // self.cfg.num_attention_heads
                    idx = [c for h in s["heads"] for c in range(h * hd, (h + 1) * hd)]
                    self._iv_hooks.append(mod.register_forward_pre_hook(
                        self._ablate_hook(idx, s.get("scale", 1.0))))
                    desc.append(f"ablate heads {s['heads']} of layer {s['layer']} x{s.get('scale', 1.0)} (head_dim {hd})")
                else:
                    raise ValueError(f"unknown intervention type {t!r}")
                self.iv_specs.append(s)
            for name in freeze:
                mod = self.hf.get_submodule(name)
                self._iv_hooks.append(mod.register_forward_hook(self._freeze_hook(name)))
                self.freeze_mods.append(name)
                desc.append(f"freeze {name} (outputs its value on the original model, no edit, no interventions)")
        except Exception:
            self.clear_interventions()
            raise
        return desc

    @torch.no_grad()
    def _swap(self, to_original):
        for n in self.active:
            self.params[n].data.copy_((self.orig[n] if to_original else self.edited[n]).to(self.device))

    @torch.no_grad()
    def _forward(self, x, patch_ids=None, **kw):
        """One forward pass with the active interventions (plus freeze / patch source runs)."""
        try:
            if any(s["type"] == "patch" for s in self.iv_specs):
                if patch_ids is None:
                    raise ValueError("a patch intervention is active: every prompt needs 'patch_from'")
                self._mode = "capture_patch"
                self.hf(torch.tensor([patch_ids], device=self.device), logits_to_keep=1)
            if self.freeze_mods:
                self._mode = "capture_freeze"
                self._swap(True)
                try:
                    self.hf(x, **kw)
                finally:
                    self._swap(False)
            self._mode = "on"
            return self.hf(x, **kw)
        finally:
            self._mode = "on"

    # ---- forward passes ------------------------------------------------------------
    def ids(self, text, raw=False):
        key = (text, raw)
        if key not in self._ids:
            s = text if raw else chat(self.tok, text)
            self._ids[key] = self.tok(s, add_special_tokens=False).input_ids
        return self._ids[key]

    @torch.no_grad()
    def yesno(self, ids, patch_ids=None):
        x = torch.tensor([ids], device=self.device)
        lg = self._forward(x, patch_ids, logits_to_keep=1).logits[0, -1].float()
        p = torch.softmax(lg, -1)
        y, n = self.yn
        return {"m": (lg[y] - lg[n]).item(), "mass": (p[y] + p[n]).item(), "logits": lg}

    @torch.no_grad()
    def nll(self, text):
        ids = self.tok(text, add_special_tokens=False).input_ids[:NLL_MAX_TOKENS]
        x = torch.tensor([ids], device=self.device)
        if any(s["type"] == "patch" for s in self.iv_specs):
            return float("nan")   # patching needs a per-prompt source; not defined for plain text
        lg = self._forward(x).logits[0, :-1].float()
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
        self.active, self.edited = [], {}

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
            self.edited[n] = p.data.clone()
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
