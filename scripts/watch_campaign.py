#!/usr/bin/env python3
"""Live progress view for a running campaign.

Campaign logs are far too verbose to follow with `tail -f` (one line per generation,
hundreds of KB), so this parses the log and renders a compact per-run status table that
refreshes in place.

    pixi run python scripts/watch_campaign.py <logfile>
    pixi run python scripts/watch_campaign.py <logfile> --once     # single snapshot

Stdlib only and no project imports, so it starts instantly rather than paying JAX's
import cost just to read a text file.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RUN_DIR = re.compile(r"^run_dir:\s*(\S+)")
SECTION = re.compile(r"^=> (.+)")
OPT_LIST = re.compile(r":\s*((?:jax|torch)/[\w,/]+)\s*\|")
GEN = re.compile(r"\[([\w.]+)\]\[INFO\] - Generation (\d+) \| train fitness min=([\d.eE+-]+)")
EPOCH = re.compile(r"\[([\w.]+)\]\[INFO\] - Epoch (\d+) \| train=([\d.eE+-]+)")
VAL = re.compile(r"\[([\w.]+)\]\[INFO\] - (?:Generation|Epoch) (\d+) \| val loss=([\d.eE+-]+)(?: accuracy=([\d.eE+-]+))?")
SAVED = re.compile(r"\[([\w.]+)\]\[INFO\] - Final best model saved")
TOTAL_GEN = re.compile(r"^generations:\s*(\d+)")
TOTAL_EPOCH = re.compile(r"^epochs:\s*(\d+)")

# Campaign scripts name optimizers by config slug, while exp_name uses the class name
# (see configs/base/run_layout.yaml). Most differ only in case/underscores and compare
# fine after normalizing; these are the ones that genuinely diverge.
SLUG_ALIASES = {"particle_swarm": "PSO"}

C = {
    "dim": "\033[2m", "bold": "\033[1m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "red": "\033[31m", "reset": "\033[0m",
}
if not sys.stdout.isatty():
    C = {k: "" for k in C}


def parse(log_path: Path) -> dict:
    runs = {}      # exp_name -> dict
    order = []
    planned = []
    sections = []
    totals = {"generations": None, "epochs": None}

    with open(log_path, errors="replace") as f:
        for line in f:
            m = SECTION.match(line)
            if m:
                sections.append(m.group(1).strip())
                o = OPT_LIST.search(line)
                if o:
                    planned.extend(x.split("/")[-1] for x in o.group(1).split(","))
                continue
            m = TOTAL_GEN.match(line)
            if m:
                totals["generations"] = int(m.group(1))
            m = TOTAL_EPOCH.match(line)
            if m:
                totals["epochs"] = int(m.group(1))

            for rx, kind in ((GEN, "gen"), (EPOCH, "epoch")):
                m = rx.search(line)
                if m:
                    name, step, train = m.group(1), int(m.group(2)), m.group(3)
                    r = runs.setdefault(name, {"kind": kind, "step": -1, "val": None,
                                               "acc": None, "done": False})
                    if name not in order:
                        order.append(name)
                    r["kind"], r["step"], r["train"] = kind, step, train
                    break

            m = VAL.search(line)
            if m:
                r = runs.get(m.group(1))
                if r:
                    r["val"] = m.group(3)
                    r["acc"] = m.group(4)
            m = SAVED.search(line)
            if m and m.group(1) in runs:
                runs[m.group(1)]["done"] = True

    return {"runs": runs, "order": order, "planned": planned,
            "sections": sections, "totals": totals}


def alive() -> bool:
    r = subprocess.run(["pgrep", "-f", "tools/train.py"], capture_output=True)
    return r.returncode == 0


def short(name: str) -> str:
    # jax_NE_Sep_CMA_ES_baseline_mnist -> jax_NE/Sep_CMA_ES
    parts = name.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}/" + "_".join(parts[2:-2]) if len(parts) > 4 else name
    return name


def render(state: dict, log_path: Path, running: bool) -> str:
    out = []
    mtime = log_path.stat().st_mtime
    age = time.time() - mtime
    size = log_path.stat().st_size

    status = f"{C['green']}● running{C['reset']}" if running else (
        f"{C['red']}■ stopped{C['reset']}" if age > 60 else f"{C['yellow']}◌ idle{C['reset']}")
    out.append(f"{C['bold']}{log_path.name}{C['reset']}   {status}"
               f"   {C['dim']}last write {int(age)}s ago · {size // 1024} KB{C['reset']}")
    if state["sections"]:
        out.append(f"{C['dim']}{state['sections'][-1][:78]}{C['reset']}")
    out.append("")

    seen_algos = set()
    for name in state["order"]:
        r = state["runs"][name]
        total = state["totals"]["generations"] if r["kind"] == "gen" else state["totals"]["epochs"]
        prog = f"{r['kind']} {r['step'] + 1}" + (f"/{total}" if total else "")
        if r["done"]:
            mark, col = "✓", C["green"]
        elif running:
            mark, col = "●", C["blue"]
        else:
            mark, col = "✗", C["red"]
        metric = ""
        if r.get("acc"):
            metric = f"val acc {float(r['acc']):.4f}"
        elif r.get("val"):
            metric = f"val loss {float(r['val']):.4f}"
        out.append(f" {col}{mark}{C['reset']} {short(name):<28} {C['dim']}{prog:<16}{C['reset']} {metric}")
        for p in state["planned"]:
            key = SLUG_ALIASES.get(p, p).lower().replace("_", "")
            if key in name.lower().replace("_", ""):
                seen_algos.add(p)

    pending = [p for p in state["planned"] if p not in seen_algos]
    for p in pending:
        out.append(f" {C['dim']}·  {p:<28} pending{C['reset']}")

    done = sum(1 for r in state["runs"].values() if r["done"])
    total_runs = len(state["planned"]) or len(state["runs"])
    out.append("")
    out.append(f"{C['bold']}{done}/{total_runs} complete{C['reset']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile")
    ap.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    log_path = Path(args.logfile)
    if not log_path.exists():
        raise SystemExit(f"No such log: {log_path}")

    try:
        while True:
            frame = render(parse(log_path), log_path, alive())
            if args.once:
                print(frame)
                return
            os.system("clear" if os.name != "nt" else "cls")
            print(frame)
            print(f"\n{C['dim']}ctrl-c to stop watching (does not affect the run){C['reset']}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
