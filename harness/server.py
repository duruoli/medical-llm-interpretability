"""GPU service for the weight-edit researcher: holds the model, enforces the rules and budget.

Start once per GPU session (repo root, inside tmux):
  python -m harness.server --model Qwen/Qwen3.5-9B --out /scratch/dlf8982/harness
It loads the model, scores the ORIGINAL model on the practice split once (free, the
reference), writes <out>/model_info.json, then listens on 127.0.0.1:<port>.
Talk to it with  python -m harness.client <command>  (see client.py / HANDOFF_weight_edit.md).

Endpoints (POST, JSON):
  status                      budget left, active edit
  score                       full practice evaluation of the current weights   [score_calls]
  apply   {path, scale}       W = W_orig + scale*delta  (replaces any active edit; free)
  reset                       back to original weights (free)
  logits  {prompts, top_k}    yes/no margin + top tokens per prompt              [probe_prompts]
  acts    {prompts, modules, positions, kind}  save activations to a .pt file    [probe_prompts]
  weights {names}             save ORIGINAL weights to a .pt file (free; analysis is CPU-side)
A prompt is {"case": i, "test": t, "variant": "B"} or {..., "pair": p, "variant": "P"|"N"}
(index into practice.jsonl) or {"text": "...", "raw": false} (chat template unless raw).
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

from harness.config import BUDGET, MAX_PROBE_PER_CALL
from harness.core import Model, load_edit, load_items, mid_cases, report, run_guards, run_items


class State:
    def __init__(self, args):
        self.out = Path(args.out)
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
        self.n_saved = len(list((self.out / "edits").glob("*.pt")))
        self.write_info(args.model)
        print(f"scoring original model on {len(self.items)} practice patients (free) ...", flush=True)
        t0 = time.time()
        self.ref = run_items(self.model, self.items)
        self.g_ref = run_guards(self.model, self.items)
        self.mid = mid_cases(self.ref, self.items)
        rep, text = report(self.ref, self.ref, self.g_ref, self.g_ref, self.items, self.mid)
        (self.out / "scores" / "original.txt").write_text(text)
        self.log({"cmd": "start", "model": args.model, "secs": round(time.time() - t0),
                  "mid": {k: len(v) for k, v in self.mid.items()}, "headline": rep["headline"]})
        print(text, flush=True)

    def write_info(self, name):
        m = self.model
        info = {"model": name, "config": m.hf.config.to_dict(),
                "parameters": {n: list(p.shape) for n, p in m.params.items()},
                "modules": {n: type(mod).__name__ for n, mod in m.hf.named_modules() if n}}
        (self.out / "model_info.json").write_text(json.dumps(info, indent=1, default=str))

    def log(self, rec):
        rec = dict(rec, time=time.strftime("%Y-%m-%d %H:%M:%S"), edit=self.active)
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
        return {"budget_left": self.budget_left(), "active_edit": self.active,
                "practice_patients": len(self.items), "mid_cases": {k: len(v) for k, v in self.mid.items()}}

    def cmd_score(self, req):
        self.spend("score_calls", 1)
        t0 = time.time()
        cur = run_items(self.model, self.items)
        g = run_guards(self.model, self.items)
        rep, text = report(cur, self.ref, g, self.g_ref, self.items, self.mid)
        tag = f"{self.used['score_calls']:03d}"
        (self.out / "scores" / f"{tag}.txt").write_text(f"edit: {self.active}\n\n{text}")
        (self.out / "scores" / f"{tag}.json").write_text(json.dumps({"edit": self.active, **rep}))
        self.log({"cmd": "score", "n": tag, "secs": round(time.time() - t0), "headline": rep["headline"]})
        return {"text": f"[score {tag}, edit {self.active}, {time.time() - t0:.0f}s, "
                        f"budget left {self.budget_left()}]\n{text}"}

    def cmd_apply(self, req):
        src = Path(req["path"])
        scale = float(req.get("scale", 1.0))
        deltas = load_edit(src)
        stats = self.model.apply(deltas, scale)
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:10]
        self.n_saved += 1
        kept = self.out / "edits" / f"{self.n_saved:03d}_{src.stem}_{digest}.pt"
        shutil.copy(src, kept)
        self.active = {"file": str(kept), "scale": scale, "nnz": stats["nnz"]}
        self.log({"cmd": "apply", **stats, "note": req.get("note", "")})
        warn = ""
        if stats["effective_changed"] < 0.5 * stats["nnz"]:
            warn = " WARNING: most of the delta was lost to bf16 rounding; use a larger delta."
        return {"text": f"applied {kept.name} x{scale}: {stats['nnz']:,} weights in delta, "
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
        self.spend("probe_prompts", len(prompts))
        return prompts, [self.prompt_ids(p) for p in prompts]

    def cmd_logits(self, req):
        prompts, ids = self._probe(req)
        k = int(req.get("top_k", 5))
        rows = []
        for p, x in zip(prompts, ids):
            r = self.model.yesno(x)
            top = torch.topk(torch.softmax(r["logits"], -1), k)
            rows.append({"prompt": p, "yes_minus_no": round(r["m"], 3), "yesno_mass": round(r["mass"], 3),
                         "top": [(self.model.tok.decode([int(i)]), round(float(v), 3))
                                 for v, i in zip(top.values, top.indices)]})
        self.log({"cmd": "logits", "n": len(rows)})
        return {"rows": rows, "budget_left": self.budget_left()}

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
            for x in ids:
                cur.clear()
                margins.append(self.model.yesno(x)["m"])
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
                    "kind": kind, "positions": pos, "edit": self.active}, path)
        self.log({"cmd": "acts", "n": len(ids), "modules": mods, "file": str(path)})
        return {"text": f"saved {path} ({len(ids)} prompts x {len(mods)} modules); "
                        f"budget left {self.budget_left()}", "path": str(path)}

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
    ap.add_argument("--out", default="/scratch/dlf8982/harness")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-cases", type=int, default=0, help="smoke test: first N practice patients")
    ap.add_argument("--score-calls", type=int, default=0, help="override budget")
    ap.add_argument("--probe-prompts", type=int, default=0, help="override budget")
    args = ap.parse_args()

    state = State(args)
    (state.out / "server.json").write_text(json.dumps({"port": args.port, "model": args.model}))
    print(f"ready on 127.0.0.1:{args.port}; budget left {state.budget_left()}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
