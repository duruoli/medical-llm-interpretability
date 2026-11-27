"""GPU service for the weight-edit researcher: holds the model, enforces the rules and budget.

Start once per GPU session (repo root, inside tmux):
  python -m harness.server --model Qwen/Qwen3.5-9B --out /projects/p53430/medllm/harness
It loads the model, scores the ORIGINAL model on the practice split once (free, the
reference), writes <out>/model_info.json, then listens on 127.0.0.1:<port>.
Talk to it with  python -m harness.client <command>  (see client.py / HANDOFF_weight_edit.md).

Endpoints (POST, JSON):
  status                      budget left, active edit
  score                       full practice evaluation of the current weights   [score_calls]
  apply   {path, scale, arm}  W = W_orig + scale*delta  (replaces any active edit; free); optional arm
                              (config.ARMS) rejects parameters outside that arm
  reset                       back to original weights (free)
  logits  {prompts, top_k}    yes/no margin + top tokens per prompt              [probe_prompts]
  acts    {prompts, modules, positions, kind}  save activations to a .pt file    [probe_prompts]
  weights {names}             save ORIGINAL weights to a .pt file (free; analysis is CPU-side)
  intervene {interventions, freeze}   install DIAGNOSTIC hooks (no weight change; free); they stay on for
                              every later score / logits / acts call until `clear` or the next `intervene`.
                              patch_units with "offsets": "auto" -> the server computes them on the practice
                              mid cases (charged to probe_prompts)
  clear                       remove all interventions (free)
  attrib  {prompts, units}    direct contribution of every block (+ units of chosen blocks) -> .pt [probe_prompts]
--target attractor (with --profile llama|qwen): `score` reports pull / move / r (harness/attractor_score.py)
instead of asym.
A prompt is {"case": i, "test": t, "variant": "B"} or {..., "pair": p, "variant": "P"|"N"}
(index into practice.jsonl) or {"text": "...", "raw": false} (chat template unless raw).
With a `patch` intervention active, each prompt also needs "patch_from": <another prompt>.
Intervention specs: see harness/core.py docstring and HANDOFF_round3.md.
"""
import argparse
import hashlib
import json
import shutil
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from harness.attractor_score import attractor_report
from harness.config import BUDGET, MAX_PROBE_PER_CALL
from harness.core import Model, load_edit, load_items, mid_cases, report, run_guards, run_items


class State:
    def __init__(self, args):
        self.out = Path(args.out)
        self.target, self.profile = args.target, args.profile
        for d in ("edits", "acts", "weights", "scores"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "log.jsonl"
        self.budget_path = self.out / "budget.json"
        self.used = (json.loads(self.budget_path.read_text()) if self.budget_path.exists()
                     else {k: 0 for k in BUDGET})
        self.limits = dict(BUDGET, **{k: v for k, v in (("score_calls", args.score_calls),
                                                         ("probe_prompts", args.probe_prompts)) if v})
        print("loading model ...", flush=True)
        self.model = Model(args.model, args.device)
        self.items = load_items(args.practice, args.limit_cases)
        self.active = None
        self.iv = None            # description of active interventions
        self.n_saved = len(list((self.out / "edits").glob("*.pt")))
        self.write_info(args.model)
        print(f"scoring original model on {len(self.items)} practice patients (free) ...", flush=True)
        t0 = time.time()
        self.ref = run_items(self.model, self.items)
        self.g_ref = run_guards(self.model, self.items)
        self.mid = mid_cases(self.ref, self.items)
        rep, text = self.report(self.ref, self.g_ref)
        (self.out / "scores" / "original.txt").write_text(text)
        self.log({"cmd": "start", "model": args.model, "secs": round(time.time() - t0),
                  "mid": {k: len(v) for k, v in self.mid.items()}, "headline": rep["headline"]})
        print(text, flush=True)

    def report(self, cur, g):
        if self.target == "attractor":
            return attractor_report(cur, self.ref, g, self.g_ref, self.items, self.mid, self.profile)
        return report(cur, self.ref, g, self.g_ref, self.items, self.mid)

    def write_info(self, name):
        m = self.model
        info = {"model": name, "config": m.hf.config.to_dict(),
                "parameters": {n: list(p.shape) for n, p in m.params.items()},
                "modules": {n: type(mod).__name__ for n, mod in m.hf.named_modules() if n}}
        (self.out / "model_info.json").write_text(json.dumps(info, indent=1, default=str))

    def log(self, rec):
        rec = dict(rec, time=time.strftime("%Y-%m-%d %H:%M:%S"), edit=self.active, interventions=self.iv)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def spend(self, key, n):
        if self.used[key] + n > self.limits[key]:
            raise ValueError(f"budget exhausted: {key} used {self.used[key]} + {n} > {self.limits[key]}")
        self.used[key] += n
        self.budget_path.write_text(json.dumps(self.used))

    def budget_left(self):
        return {k: self.limits[k] - self.used[k] for k in self.limits}

    # ---- prompts -----------------------------------------------------------------------
    def prompt_ids(self, p):
        if "text" in p:
            return self.model.ids(p["text"], raw=bool(p.get("raw")))
        t = self.items[p["case"]]["tests"][p["test"]]
        text = t["B"] if p["variant"] == "B" else t["pairs"][p["pair"]][p["variant"]]
        return self.model.ids(text)

    # ---- commands ------------------------------------------------------------------------
    def cmd_status(self, req):
        return {"budget_left": self.budget_left(), "active_edit": self.active, "interventions": self.iv,
                "practice_patients": len(self.items), "mid_cases": {k: len(v) for k, v in self.mid.items()}}

    def cmd_score(self, req):
        self.spend("score_calls", 1)
        t0 = time.time()
        cur = run_items(self.model, self.items)
        g = run_guards(self.model, self.items)
        rep, text = self.report(cur, g)
        tag = f"{self.used['score_calls']:03d}"
        (self.out / "scores" / f"{tag}.txt").write_text(
            f"edit: {self.active}\ninterventions: {self.iv}\n\n{text}")
        (self.out / "scores" / f"{tag}.json").write_text(
            json.dumps({"edit": self.active, "interventions": self.iv, **rep}))
        self.log({"cmd": "score", "n": tag, "secs": round(time.time() - t0), "headline": rep["headline"]})
        return {"text": f"[score {tag}, edit {self.active}, interventions {self.iv}, {time.time() - t0:.0f}s, "
                        f"budget left {self.budget_left()}]\n{text}"}

    def auto_offsets(self, specs):
        """patch_units with offsets "auto": per cell, mean over the practice mid cases of
        value(result prompt) - value(B prompt), at the current weights, no interventions.
        Then value = B value + offset = (cell mean with result) + (patient's no-result deviation from its mean)."""
        auto = [s for s in specs if s.get("type") == "patch_units" and s.get("offsets") == "auto"]
        if not auto:
            return 0
        targets = [(s["layer"], s.get("kind", "mlp"), s["units"]) for s in auto]
        tests = self.items[0]["tests"]
        n = sum(len(self.mid[t]) * (1 + 2 * len(tests[t]["pairs"])) for t in tests)
        self.spend("probe_prompts", n)
        self.model.clear_interventions()
        self.iv = None
        sums = [dict() for _ in auto]
        for t in tests:
            for i in self.mid[t]:
                T = self.items[i]["tests"][t]
                vb = self.model.capture_inputs(self.model.ids(T["B"]), targets)
                for pair, pr in T["pairs"].items():
                    for side in "PN":
                        vs = self.model.capture_inputs(self.model.ids(pr[side]), targets)
                        key = f"{t}|{pair}|{side}"
                        for j in range(len(auto)):
                            d = vs[j] - vb[j]
                            sums[j][key] = sums[j][key] + d if key in sums[j] else d
        for j, s in enumerate(auto):
            s["offsets"] = {k: (v / len(self.mid[k.split("|")[0]])).tolist() for k, v in sums[j].items()}
        return n

    def cmd_intervene(self, req):
        specs = [dict(s) for s in req.get("interventions", [])]
        n_auto = self.auto_offsets(specs)
        desc = self.model.set_interventions(specs, req.get("freeze", []))
        self.iv = desc or None
        self.log({"cmd": "intervene", "spec": {**req, "interventions": [
            {k: (v if k != "offsets" else f"<{len(v)} cells>") for k, v in s.items()} for s in specs]},
            "auto_offset_prompts": n_auto})
        extra = f"\n  (auto offsets: {n_auto} probe prompts charged)" if n_auto else ""
        return {"text": "interventions active (diagnostic only, weights unchanged):\n" +
                        "\n".join(f"  {d}" for d in desc) + extra}

    def cmd_clear(self, req):
        self.model.clear_interventions()
        self.iv = None
        self.log({"cmd": "clear"})
        return {"text": "all interventions removed"}

    def cmd_apply(self, req):
        src = Path(req["path"])
        scale = float(req.get("scale", 1.0))
        deltas = load_edit(src)
        arm = req.get("arm") or None
        stats = self.model.apply(deltas, scale, arm)
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:10]
        self.n_saved += 1
        kept = self.out / "edits" / f"{self.n_saved:03d}_{src.stem}_{digest}.pt"
        shutil.copy(src, kept)
        self.active = {"file": str(kept), "scale": scale, "nnz": stats["nnz"], "arm": arm}
        self.log({"cmd": "apply", **stats, "arm": arm, "note": req.get("note", "")})
        warn = ""
        if stats["effective_changed"] < 0.5 * stats["nnz"]:
            warn = " WARNING: most of the delta was lost to bf16 rounding; use a larger delta."
        return {"text": f"applied {kept.name} x{scale} (arm {arm}): {stats['nnz']:,} weights in delta, "
                        f"{stats['effective_changed']:,} actually changed.{warn}\n"
                        "relative norm |delta|/|W| per parameter:\n" +
                        "\n".join(f"  {n}: {r:.4f}" for n, r in stats["relative_norm"])}

    def cmd_reset(self, req):
        self.model.reset()
        self.active = None
        self.log({"cmd": "reset"})
        return {"text": "weights restored to original"}

    def _probe(self, req):
        prompts = req["prompts"]
        if len(prompts) > MAX_PROBE_PER_CALL:
            raise ValueError(f"at most {MAX_PROBE_PER_CALL} prompts per call")
        ids = [self.prompt_ids(p) for p in prompts]
        src = [self.prompt_ids(p["patch_from"]) if "patch_from" in p else None for p in prompts]
        if any(s["type"] == "patch" for s in self.model.iv_specs) and None in src:
            raise ValueError("a patch intervention is active: every prompt needs 'patch_from' (nothing charged)")
        # patch_units: result prompts of practice cases take the same patient's B prompt as source
        ctx = [None] * len(prompts)
        if self.model.has_unit_patch():
            for j, p in enumerate(prompts):
                if "case" in p and p.get("variant") in ("P", "N"):
                    b = self.prompt_ids({"case": p["case"], "test": p["test"], "variant": "B"})
                    ctx[j] = (b, f"{p['test']}|{p['pair']}|{p['variant']}")
        self.spend("probe_prompts", len(prompts) + sum(s is not None for s in src) + sum(c is not None for c in ctx))
        return prompts, [(x, s, c) for x, s, c in zip(ids, src, ctx)]

    def cmd_logits(self, req):
        prompts, ids = self._probe(req)
        k = int(req.get("top_k", 5))
        rows = []
        for p, (x, s, c) in zip(prompts, ids):
            r = self.model.yesno(x, s, c)
            top = torch.topk(torch.softmax(r["logits"], -1), k)
            rows.append({"prompt": p, "yes_minus_no": round(r["m"], 3), "yesno_mass": round(r["mass"], 3),
                         "top": [(self.model.tok.decode([int(i)]), round(float(v), 3))
                                 for v, i in zip(top.values, top.indices)]})
        self.log({"cmd": "logits", "n": len(rows)})
        return {"rows": rows, "interventions": self.iv, "budget_left": self.budget_left()}

    @torch.no_grad()
    def cmd_acts(self, req):
        prompts, ids = self._probe(req)
        mods, kind, pos = req["modules"], req.get("kind", "output"), req.get("positions", "last")
        cur, hooks = {}, []

        def grab(name):
            def out_hook(mod, args, out):
                cur[name] = out[0] if isinstance(out, tuple) else out

            def in_hook(mod, args, kwargs):
                cur[name] = args[0] if args else next(v for v in kwargs.values() if torch.is_tensor(v))
            return in_hook if kind == "input" else out_hook

        for m in mods:
            mod = self.model.hf.get_submodule(m)
            hooks.append(mod.register_forward_pre_hook(grab(m), with_kwargs=True) if kind == "input"
                         else mod.register_forward_hook(grab(m)))
        acts = {m: [] for m in mods}
        margins, toks = [], []
        try:
            for x, s, c in ids:
                cur.clear()
                margins.append(self.model.yesno(x, s, c)["m"])
                sel = (slice(None) if pos == "all" else [-1] if pos == "last" else list(pos))
                for m in mods:
                    t = cur[m][0]
                    acts[m].append(t[sel].detach().to("cpu", torch.float16 if pos == "all" else torch.float32))
                toks.append(self.model.tok.convert_ids_to_tokens(x))
        finally:
            for h in hooks:
                h.remove()
        path = self.out / "acts" / f"{time.strftime('%H%M%S')}_{len(list((self.out / 'acts').glob('*.pt'))):04d}.pt"
        torch.save({"acts": acts, "prompts": prompts, "tokens": toks, "yes_minus_no": margins,
                    "kind": kind, "positions": pos, "edit": self.active, "interventions": self.iv}, path)
        self.log({"cmd": "acts", "n": len(ids), "modules": mods, "file": str(path)})
        return {"text": f"saved {path} ({len(ids)} prompts x {len(mods)} modules); "
                        f"budget left {self.budget_left()}", "path": str(path)}

    def cmd_attrib(self, req):
        prompts, ids = self._probe(req)
        units = list(req.get("units", []))
        rows = [self.model.attrib(x, units, c, s) for x, s, c in ids]
        names = list(rows[0]["blocks"])
        path = self.out / "acts" / f"attrib_{time.strftime('%H%M%S')}_{len(list((self.out / 'acts').glob('*.pt'))):04d}.pt"
        torch.save({"prompts": prompts, "blocks": names,
                    "contrib": torch.tensor([[r["blocks"][n] for n in names] for r in rows]),
                    "units": {u: torch.stack([r["units"][u] for r in rows]) for u in units},
                    "yes_minus_no": [r["m"] for r in rows], "decomp_err": [r["decomp_err"] for r in rows],
                    "rms": [r["rms"] for r in rows], "edit": self.active, "interventions": self.iv}, path)
        self.log({"cmd": "attrib", "n": len(rows), "units": units, "file": str(path)})
        return {"text": f"saved {path} ({len(rows)} prompts; contrib = prompts x {len(names)} blocks "
                        f"[embed, a0, m0, ...]; units: {units}); max decomposition error "
                        f"{max(r['decomp_err'] for r in rows):.3g}; budget left {self.budget_left()}", "path": str(path)}

    def cmd_weights(self, req):
        names = req["names"]
        bad = [n for n in names if n not in self.model.params]
        if bad:
            raise ValueError(f"unknown parameters: {bad}")
        path = self.out / "weights" / f"{time.strftime('%H%M%S')}.pt"
        torch.save({n: self.model.original(n).float() for n in names}, path)
        self.log({"cmd": "weights", "names": names, "file": str(path)})
        return {"text": f"saved ORIGINAL weights to {path}", "path": str(path)}


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            cmd = self.path.strip("/")
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                fn = getattr(state, f"cmd_{cmd}", None)
                if fn is None:
                    raise ValueError(f"unknown command {cmd!r}")
                body, code = fn(req), 200
            except Exception as e:
                body, code = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}, 400
            data = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--practice", default="data/harness/practice.jsonl")
    ap.add_argument("--out", default="/projects/p53430/medllm/harness")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-cases", type=int, default=0, help="smoke test: first N practice patients")
    ap.add_argument("--score-calls", type=int, default=0, help="override budget")
    ap.add_argument("--probe-prompts", type=int, default=0, help="override budget")
    ap.add_argument("--target", default="asym", choices=["asym", "attractor"])
    ap.add_argument("--profile", default="llama", help="attractor target profile (config.ATTRACTOR_PROFILES)")
    args = ap.parse_args()

    state = State(args)
    (state.out / "server.json").write_text(json.dumps({"port": args.port, "model": args.model}))
    print(f"ready on 127.0.0.1:{args.port}; budget left {state.budget_left()}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
