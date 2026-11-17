# SPDX-License-Identifier: Apache-2.0

"""Stop the lanes when the runs of an experiment have cost more than its cap.

    nohup python -m experiment.cost_cap paper/jobs/R-runs.tsv --cap 300 >> runs/lanes/cost-cap.out 2>&1 < /dev/null &

Every ten minutes the spending of the runs the table names is summed from their own event
logs (a run's last ``round`` event carries what its roles and its evaluations have cost so
far). Above the cap the file ``runs/lanes/STOP`` is written, and every lane ends after its
current job (``experiment/lane.sh``). The watch ends when no lane is alive. (The lanes are
found by a pattern written so that it does not match this command's own line.)"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from .config import ROOT


def spent(names: list[str]) -> tuple[float, int, int]:
    """Dollars spent by the runs so far, how many have started, how many have ended."""
    total, started, ended = 0.0, 0, 0
    for name in names:
        log = ROOT / "runs" / name / "log" / "events.jsonl"
        if not log.exists():
            continue
        started += 1
        last = 0.0
        for line in log.read_text(encoding="utf-8").splitlines():
            if '"kind": "round"' in line:
                last = float(json.loads(line).get("total_cost_usd") or 0.0)
            elif '"kind": "end"' in line:
                ended += 1
        total += last
    return total, started, ended


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.cost_cap")
    parser.add_argument("runs", type=Path, help="the table of the experiment's runs (experiment/jobs.py)")
    parser.add_argument("--cap", type=float, required=True, help="dollars")
    parser.add_argument("--lanes", default="lane[.]sh R-", help="a pattern the lanes' command lines match and this command's own does not")
    parser.add_argument("--every", type=float, default=600.0, help="seconds between two sums")
    args = parser.parse_args(argv)
    names = [row["name"] for row in csv.DictReader(args.runs.open(encoding="utf-8"), delimiter="\t")]
    while True:
        total, started, ended = spent(names)
        alive = subprocess.run(["pgrep", "-f", args.lanes], capture_output=True, text=True).stdout.split()
        print(f"{time.strftime('%m-%d %H:%M')} spent ${total:.2f} of ${args.cap:.0f}; runs started {started}, ended {ended} of {len(names)}; lanes alive {len(alive)}", flush=True)
        if total > args.cap:
            (ROOT / "runs" / "lanes" / "STOP").write_text(f"the runs have cost ${total:.2f}, above the cap of ${args.cap:.0f}\n", encoding="utf-8")
            print("the cap is passed: runs/lanes/STOP written; every lane ends after its current job", flush=True)
            return 1
        if not alive:
            print("no lane is alive: the watch ends", flush=True)
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
