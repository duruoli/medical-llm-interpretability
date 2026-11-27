"""Model wrapper, editing with rollback, scoring and guards. Shared by server.py and final_eval.py.

Diagnostic interventions (forward hooks, no weight change, removable) -- see Model.set_interventions:
  add            h[pos] += alpha * v                        at a module output (default: residual after layer L)
  project        h[pos] -= strength * ((h.v^) - center) v^  (remove / clamp a direction)
  patch          h[pos] = same module, same position-from-end, in a SOURCE prompt run (per-prompt patch_from)
  ablate_neurons MLP neurons of layer L scaled by (1 - scale) (input of down_proj)
  ablate_heads   attention heads of a full-attention layer L scaled by (1 - scale) (input of o_proj)
  freeze         listed modules output what they output on the ORIGINAL model (no weight edit, no
                 interventions) for the same prompt -> tests self-repair / compensation.
  patch_units    at the answer position, set chosen units (MLP neurons = input of down_proj, or attention heads =
                 input of o_proj / out_proj) to  value on the SOURCE prompt + offset[cell].  Source = the same
                 patient's no-result (B) prompt; offset per "test|pair|side" (server can fill it as the mean over
                 mid cases of value(result prompt) - value(B prompt)  ->  "add the patient information back").
                 Prompts without a source (B prompts, general questions, NLL texts) are left untouched.
Attribution (Model.attrib): direct contribution of every block (embedding, each attention, each MLP) to the
yes-no margin at the answer position, final norm linearised per prompt; optionally per unit of chosen blocks.
"""
import json
import math
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.config import (DOWN_FLOOR, FORBIDDEN, GENERAL_SUFFIX, GENERAL_YESNO, MAX_NNZ,
                            NLL_MAX_TOKENS, NLL_TEXTS, PB_RANGE, arm_check, lit_moves)


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def attn_child(layer):
    """Name of a decoder layer's attention (token-mixing) sub-module."""
    return next(n for n, _ in layer.named_children() if "attn" in n or n == "mixer")


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
        self._mode = "on"       # "on" | "capture_patch" | "capture_freeze" | "capture_units"
        self._patch_src, self._freeze_src = {}, {}
        self._unit_src, self._unit_key = {}, None     # patch_units: captured source values, current cell key

    # ---- unit addressing (MLP neurons / attention heads) -------------------------------
    def unit_module(self, layer, kind):
        """Module whose INPUT holds the units: mlp.down_proj (neurons) or the attention output projection
        (heads). Returns (name, module, features per unit)."""
        if kind == "mlp":
            name = self.layer_module(layer, "mlp.down_proj")
            return name, self.hf.get_submodule(name), 1
        layer_mod = self.hf.get_submodule(self.layer_module(layer))
        att_name = attn_child(layer_mod)
        att = getattr(layer_mod, att_name)
        proj = next(s for s in ("o_proj", "out_proj") if hasattr(att, s))
        mod = getattr(att, proj)
        hd = (getattr(att, "head_v_dim", None) or getattr(att, "head_dim", None) or getattr(self.cfg, "head_dim", None)
              or self.cfg.hidden_size // self.cfg.num_attention_heads)
        return self.layer_module(layer, f"{att_name}.{proj}"), mod, hd

    def unit_index(self, layer, kind, units):
        name, mod, hd = self.unit_module(layer, kind)
        if units == "all":
            return name, mod, torch.arange(mod.in_features)
        return name, mod, torch.as_tensor([u * hd + j for u in units for j in range(hd)], dtype=torch.long)

    def has_unit_patch(self):
        return any(s["type"] == "patch_units" for s in self.iv_specs)

    def _units_hook(self, k, s):
        idx = s["_idx"].to(self.device)

        def pre(mod, args):
            if self._mode == "capture_units":
                self._unit_src[k] = args[0][0, -1, idx].detach().float().clone()
                return None
            if self._mode != "on" or self._unit_key is None or k not in self._unit_src:
                return None
            off = s["_off"].get(self._unit_key)
            if off is None:
                return None
            x = args[0].clone()
            x[0, -1, idx] = (self._unit_src[k] + off.to(self.device)).to(x.dtype)
            return (x,) + tuple(args[1:])
        return pre

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
        self._unit_src, self._unit_key = {}, None

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
                elif t == "patch_units":
                    kind = s.get("kind", "mlp")
                    if kind not in ("mlp", "heads"):
                        raise ValueError("patch_units kind must be 'mlp' or 'heads'")
                    name, mod, idx = self.unit_index(s["layer"], kind, s["units"])
                    s["_idx"] = idx
                    offs = s.get("offsets") or {}
                    if isinstance(offs, str):
                        offs = torch.load(offs, map_location="cpu", weights_only=False)
                    s["_off"] = {}
                    for key, v in offs.items():
                        v = torch.as_tensor(v, dtype=torch.float32).flatten()
                        if v.numel() != len(idx):
                            raise ValueError(f"patch_units offset {key}: {v.numel()} values, expected {len(idx)}")
                        s["_off"][key] = v
                    if not s["_off"]:
                        raise ValueError("patch_units needs offsets (or 'auto' on the server): "
                                         "{'test|pair|side': [values per feature]}")
                    self._iv_hooks.append(mod.register_forward_pre_hook(self._units_hook(k, s)))
                    nu = "all" if s["units"] == "all" else len(s["units"])
                    desc.append(f"patch_units {kind} layer {s['layer']} ({name}): {nu} units, {len(idx)} features, "
                                f"source = same patient's B prompt, offsets for {len(s['_off'])} cells")
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
    def _forward(self, x, patch_ids=None, unit_ctx=None, **kw):
        """One forward pass with the active interventions (plus freeze / patch / patch_units source runs).
        unit_ctx = (source ids, cell key) for patch_units; None -> patch_units leave this prompt alone."""
        try:
            self._unit_key, self._unit_src = None, {}
            if any(s["type"] == "patch" for s in self.iv_specs):
                if patch_ids is None:
                    raise ValueError("a patch intervention is active: every prompt needs 'patch_from'")
                self._mode = "capture_patch"
                self.hf(torch.tensor([patch_ids], device=self.device), logits_to_keep=1)
            if unit_ctx is not None and self.has_unit_patch():
                self._mode = "capture_units"
                self.hf(torch.tensor([unit_ctx[0]], device=self.device), logits_to_keep=1)
                self._unit_key = unit_ctx[1]
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
            self._unit_key = None

    # ---- forward passes ------------------------------------------------------------
    def ids(self, text, raw=False):
        key = (text, raw)
        if key not in self._ids:
            s = text if raw else chat(self.tok, text)
            self._ids[key] = self.tok(s, add_special_tokens=False).input_ids
        return self._ids[key]

    @torch.no_grad()
    def yesno(self, ids, patch_ids=None, unit_ctx=None):
        x = torch.tensor([ids], device=self.device)
        lg = self._forward(x, patch_ids, unit_ctx, logits_to_keep=1).logits[0, -1].float()
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

    @torch.no_grad()
    def capture_inputs(self, ids, targets):
        """targets: [(layer, kind, units)]. Plain forward (call with no interventions installed); returns the
        answer-position input values of those units, one float32 CPU tensor per target."""
        got, hooks = {}, []
        for j, (layer, kind, units) in enumerate(targets):
            _, mod, idx = self.unit_index(layer, kind, units)
            idx = idx.to(self.device)
            hooks.append(mod.register_forward_pre_hook(
                lambda m, a, j=j, idx=idx: got.__setitem__(j, a[0][0, -1, idx].detach().float().cpu())))
        try:
            self.hf(torch.tensor([ids], device=self.device), logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        return [got[j] for j in range(len(targets))]

    def _attrib_setup(self):
        if getattr(self, "_attr", None) is not None:
            return self._attr
        base = self.hf.get_submodule(self.layers_prefix.rsplit(".", 1)[0])
        norm, embed = base.norm, base.embed_tokens
        dev = next(norm.parameters()).device
        d = self.cfg.hidden_size
        with torch.no_grad():
            g = norm(torch.ones(1, 1, d, device=dev, dtype=next(norm.parameters()).dtype))[0, 0].float()
            W = self.hf.get_output_embeddings().weight
            y, n = self.yn
            wg = (W[y].float() - W[n].float()).to(dev) * g
        blocks = [("embed", embed)]
        for L in range(self.cfg.num_hidden_layers):
            layer = self.hf.get_submodule(self.layer_module(L))
            blocks += [(f"a{L}", getattr(layer, attn_child(layer))), (f"m{L}", layer.mlp)]
        eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
        self._attr = {"norm": norm, "blocks": blocks, "wg": wg, "eps": eps}
        return self._attr

    @torch.no_grad()
    def attrib(self, ids, units=(), unit_ctx=None, patch_ids=None):
        """Direct contribution of each block to the margin at the answer position (with active edit and
        interventions). units: block names ("m24", "a27") to split per neuron / head. Returns dict."""
        S = self._attrib_setup()
        cap, hooks = {}, []
        for name, mod in S["blocks"]:
            hooks.append(mod.register_forward_hook(
                lambda m, a, o, name=name: cap.__setitem__(name, (o[0] if isinstance(o, tuple) else o)[0, -1].detach().float())))
        hooks.append(S["norm"].register_forward_pre_hook(lambda m, a: cap.__setitem__("h_final", a[0][0, -1].detach().float())))
        uin = {}
        for u in units:
            kind = "mlp" if u[0] == "m" else "heads"
            _, mod, hd = self.unit_module(int(u[1:]), kind)
            hooks.append(mod.register_forward_pre_hook(
                lambda m, a, u=u: uin.__setitem__(u, a[0][0, -1].detach().float())))
        try:
            m = self.yesno(ids, patch_ids, unit_ctx)["m"]
        finally:
            for h in hooks:
                h.remove()
        h = cap["h_final"]
        rms = torch.sqrt((h * h).mean() + S["eps"])
        contrib = {name: float((cap[name] * S["wg"]).sum() / rms) for name, _ in S["blocks"]}
        out = {"m": m, "blocks": contrib, "decomp_err": abs(sum(contrib.values()) - m), "rms": float(rms), "units": {}}
        for u in units:
            kind = "mlp" if u[0] == "m" else "heads"
            _, mod, hd = self.unit_module(int(u[1:]), kind)
            read = mod.weight.float().t() @ S["wg"]                # (in_features,)
            per = uin[u] * read / rms
            out["units"][u] = (per if kind == "mlp" else per.view(-1, hd).sum(-1)).cpu()
        return out

    # ---- editing ---------------------------------------------------------------------
    def original(self, name):
        return self.orig[name] if name in self.orig else self.params[name].detach().cpu()

    def validate(self, deltas, arm=None):
        nnz = 0
        for n, d in deltas.items():
            if n not in self.params:
                raise ValueError(f"unknown parameter {n!r} (see model_info.json for names)")
            if FORBIDDEN.search(n):
                raise ValueError(f"parameter {n!r} is off limits (embedding / lm_head / final norm / vision)")
            arm_check(n, arm)
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
    def apply(self, deltas, scale=1.0, arm=None):
        """Replace any current edit by W = W_orig + scale * delta. Returns stats."""
        nnz = self.validate(deltas, arm)
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
    units = model.has_unit_patch()      # patch_units: source = the same patient's B prompt
    for it in items:
        row = {}
        for test, t in it["tests"].items():
            bids = model.ids(t["B"])
            b = model.yesno(bids)
            r = {"B": b["m"], "mass_B": b["mass"]}
            for pair, pr in t["pairs"].items():
                ctx = lambda s: (bids, f"{test}|{pair}|{s}") if units else None
                r[pair] = {"P": model.yesno(model.ids(pr["P"]), unit_ctx=ctx("P"))["m"],
                           "N": model.yesno(model.ids(pr["N"]), unit_ctx=ctx("N"))["m"]}
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
    by_test = lambda cs: {t: float(np.mean([abs(c["asym"]) for k, c in cs.items() if k.split("/")[0] == t]))
                          for t in dict.fromkeys(k.split("/")[0] for k in cs)}
    head = {
        "abs_asym": float(np.mean([abs(c["asym"]) for c in c_now.values()])),
        "abs_asym_orig": float(np.mean([abs(c["asym"]) for c in c_ref.values()])),
        "abs_asym_by_test": by_test(c_now), "abs_asym_by_test_orig": by_test(c_ref),
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
         f"  abs_asym       {head['abs_asym']:.3f}  (orig {head['abs_asym_orig']:.3f}; mean over cells of |asym|: "
         "over- AND under-correction both count)",
         "  abs_asym by test: " + "  ".join(f"{t} {head['abs_asym_by_test_orig'][t]:.2f}->{v:.2f}"
                                           for t, v in head["abs_asym_by_test"].items()),
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
