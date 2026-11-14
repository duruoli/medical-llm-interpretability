"""Command-line client for harness.server.

  python -m harness.client status
  python -m harness.client score
  python -m harness.client apply /scratch/dlf8982/harness/work/my_edit.pt [--scale 0.5] [--note "..."]
  python -m harness.client reset
  python -m harness.client logits prompts.json [--top-k 5]
  python -m harness.client acts spec.json          # {"prompts": [...], "modules": [...], "positions": "last", "kind": "output"}
  python -m harness.client weights NAME [NAME ...]

prompts.json = a JSON list of prompts, each {"case": i, "test": t, "variant": "B"},
{"case": i, "test": t, "pair": p, "variant": "P"|"N"}, or {"text": "..."}.
From Python:  from harness.client import call;  call("logits", prompts=[...])
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path("/scratch/dlf8982/harness")


def call(cmd, out=OUT, **req):
    port = json.loads((Path(out) / "server.json").read_text())["port"]
    r = urllib.request.Request(f"http://127.0.0.1:{port}/{cmd}", data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=7200) as f:
            return json.loads(f.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "score", "apply", "reset", "logits", "acts", "weights"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--note", default="")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=str(OUT), help="harness output dir (holds server.json)")
    a = ap.parse_args()

    if a.cmd == "apply":
        req = {"path": str(Path(a.args[0]).resolve()), "scale": a.scale, "note": a.note}
    elif a.cmd == "logits":
        req = {"prompts": json.loads(Path(a.args[0]).read_text()), "top_k": a.top_k}
    elif a.cmd == "acts":
        req = json.loads(Path(a.args[0]).read_text())
    elif a.cmd == "weights":
        req = {"names": a.args}
    else:
        req = {}
    res = call(a.cmd, a.out, **req)
    if "error" in res:
        print("ERROR:", res["error"], file=sys.stderr)
        sys.exit(1)
    print(res["text"] if "text" in res else json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
