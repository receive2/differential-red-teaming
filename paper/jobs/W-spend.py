# SPDX-License-Identifier: Apache-2.0

"""Stop the lanes of the second backend's experiment when its runs have cost more than the cap.

    nohup python paper/jobs/W-spend.py --cap 300 >> runs/lanes/spend.out 2>&1 < /dev/null &

Run from the repository root. Every ten minutes the spending of the runs of
``paper/jobs/W-runs.tsv`` is summed from their own records: what the roles have cost (a run's
last ``round`` event) and every evaluation the run has made (``log/subject_costs.jsonl``: the
gate's, and the ones the roles and the red team ask for), which is what the reading reports
as a run's cost. ``experiment/cost_cap.py`` counts the roles and the gate's evaluations
only: in the first experiment 108 of the 192 dollars the runs cost. Above the cap the file
``runs/lanes/STOP`` is written and every lane ends after its current job. The watch ends when
no lane is alive."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def spent(names: list[str]) -> tuple[float, float, int, int]:
    """Dollars the roles and the evaluations of the runs have cost so far, how many runs have
    started, how many have ended."""
    roles = evaluations = 0.0
    started = ended = 0
    for name in names:
        log = Path("runs") / name / "log" / "events.jsonl"
        if not log.exists():
            continue
        started += 1
        last = 0.0
        for line in log.read_text(encoding="utf-8").splitlines():
            if '"kind": "round"' in line:
                last = float(json.loads(line).get("roles_cost_usd") or 0.0)
            elif '"kind": "end"' in line:
                ended += 1
        roles += last
        ledger = log.with_name("subject_costs.jsonl")
        if ledger.exists():
            evaluations += sum(float(json.loads(line).get("cost_usd") or 0.0) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())
    return roles, evaluations, started, ended


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python paper/jobs/W-spend.py")
    parser.add_argument("--cap", type=float, required=True, help="dollars")
    parser.add_argument("--runs", type=Path, default=Path("paper/jobs/W-runs.tsv"))
    parser.add_argument("--lanes", default="lane[.]sh W-", help="a pattern the lanes' command lines match and this command's own does not")
    parser.add_argument("--every", type=float, default=600.0, help="seconds between two sums")
    args = parser.parse_args(argv)
    with args.runs.open(encoding="utf-8") as table:
        names = [row["name"] for row in csv.DictReader(table, delimiter="\t")]
    while True:
        roles, evaluations, started, ended = spent(names)
        alive = subprocess.run(["pgrep", "-f", args.lanes], capture_output=True, text=True).stdout.split()
        print(f"{time.strftime('%m-%d %H:%M')} spent ${roles + evaluations:.2f} of ${args.cap:.0f} (roles {roles:.2f}, evaluations {evaluations:.2f}); runs started {started}, ended {ended} of {len(names)}; lane processes {len(alive)}", flush=True)
        if roles + evaluations > args.cap:
            Path("runs/lanes/STOP").write_text(f"the runs have cost ${roles + evaluations:.2f}, above the cap of ${args.cap:.0f}\n", encoding="utf-8")
            print("the cap is passed: runs/lanes/STOP written; every lane ends after its current job", flush=True)
            return 1
        if not alive:
            print("no lane is alive: the watch ends", flush=True)
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
