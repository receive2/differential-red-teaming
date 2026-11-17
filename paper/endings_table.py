#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""The table of how the runs end, by architecture, in percent of runs with one decimal:
results/tables/tab_endings.tex, from the reading of an experiment (the file paper/reading.py
writes with --json), so that the table and the tests share their runs.

    python3 paper/endings_table.py                                          # the first backend: results/esci-reading.json
    python3 paper/endings_table.py --reading results/wands-reading.json --out results/tables/tab_endings_wands.tex --label tab:endings-wands --where "on the Wayfair WANDS backend"
    python3 paper/endings_table.py --reading results/esci-reading.json results/wands-reading.json --out ...    # both backends together
    python3 paper/endings_table.py --standalone --out endings.tex                                              # compilable

A cell is 100 x count / runs in the row's group, rounded half up (5/80 = 6.25 -> 6.3).
"No release" is computed from the merged count of withdrawn and cancelled runs, not from
the two rounded percentages. The rows, their order and labels, the column heads and the
caption are the settings below; every number comes from the reading.

Where a repair exists two architectures have further seeds (Paired and DRT, so that the
difference of their real fixes is estimated more closely). The table counts every run, each
row over its own runs, and the head names the rows that have more, so that its numbers are
the comparisons'. ``--seeds shared`` counts that group over the seeds every architecture has
(31 to 38) instead, so that every row has the same number of runs. What the two deliver over
all their runs is printed beside the table either way.
"""
import argparse
import json
import re
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Rows in order: (arrangement in the reading, label in the table); None is a \midrule.
ROWS = [
    ("solo", "Solo Agent"),
    ("pair", "Paired"),
    ("isolated", "Isolated"),
    ("monitor", "Monitor"),
    ("redteam", "RedTeam"),
    ("compare", "\\textbf{DRT}"),
    None,
    ("sealed", "Oracle$^\\dagger$"),
]

# The two groups of columns: (group, heading, [(ending, column head), ...]).
GROUPS = [
    ("imp", "No repair exists", [("false", "Facade $\\downarrow$"), ("truth", "Disclosure"),
                                 ("other", "Other"), ("norelease", "No release")]),
    ("cap", "A repair exists", [("real", "Real fix $\\uparrow$"), ("truth", "Disclosure"),
                                ("false", "Facade"), ("other", "Other"), ("norelease", "No release")]),
]

# Which counted endings make up each column.
ENDINGS = {"real": ["real"], "truth": ["truth"], "false": ["false"], "other": ["other"],
           "norelease": ["withdrawn", "cancelled"]}

CAPTION = ("Outcomes by architecture, in \\% of runs. Where no repair exists (parameter ignoring or "
           "parameter absent backends), the right outcome is a disclosure or no release; where one "
           "exists (capable backends), it is a real fix, and a disclosure is a missed repair. "
           "\\emph{No release}: the system withdrew or exhausted its round budget. "
           "$^\\dagger$Reference only: the oracle check tests each release on held out probe "
           "questions, labeled with knowledge of what the backend can do.")

LABEL = "tab:endings"
FIRST_BACKEND = "on the Amazon ESCI backend"  # the caption names the backend the runs were made on

# A run of a reading: the kinds of backend of each group, the reading's endings by the short
# names of the columns, the seeds every architecture has in every condition.
STORES = {"imp": ("ign", "non"), "cap": ("cap",)}
SHORT = {"false claim shipped": "false", "truth shipped": "truth", "real shipped": "real", "other shipped": "other",
         "withdrawn": "withdrawn", "cancelled": "cancelled", "unresolved": "unresolved"}
SHARED_SEEDS = range(31, 39)
ARMS = ("pair", "compare")  # the two architectures with further seeds where a repair exists


def read_readings(paths: list[Path], seeds: str) -> dict[str, str]:
    """The counts of the table from one or more readings, named ``<arrangement>-<group>-<ending>``
    (``pair-cap-real``, ``pair-cap-n``). A run the reading excludes is not counted."""
    counts: Counter[str] = Counter()
    for path in paths:
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            if run.get("excluded"):
                continue
            for group, stores in STORES.items():
                if run["store"] not in stores or (group == "cap" and seeds == "shared" and run["seed"] not in SHARED_SEEDS):
                    continue
                counts[f"{run['arrangement']}-{group}-n"] += 1
                counts[f"{run['arrangement']}-{group}-{SHORT[run['ending']]}"] += 1
    return {f"{arr}-{group}-{ending}": str(counts[f"{arr}-{group}-{ending}"]) for arr in (r[0] for r in ROWS if r) for group in STORES for ending in ("n", *SHORT.values())}


def count(keys: dict[str, str], arr: str, group: str, column: str) -> int:
    return sum(int(keys[f"{arr}-{group}-{e}"]) for e in ENDINGS[column])


def percent(numerator: int, denominator: int) -> str:
    return str((Decimal(100 * numerator) / Decimal(denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def check(keys: dict[str, str]) -> dict[str, dict[str, int]]:
    """The runs each row has in each group, after the consistency checks."""
    arrs = [r[0] for r in ROWS if r]
    runs: dict[str, dict[str, int]] = {}
    for group, _, cols in GROUPS:
        runs[group] = {a: int(keys[f"{a}-{group}-n"]) for a in arrs}
        for a in arrs:
            if not runs[group][a]:
                raise SystemExit(f"{a}-{group}: no run")
            total = sum(count(keys, a, group, c) for c, _ in cols)
            if total != runs[group][a]:
                raise SystemExit(f"{a}-{group}: the columns hold {total} runs of {runs[group][a]}")
    for a in arrs:  # no real fix where none exists, so the left group has no such column
        if int(keys[f"{a}-imp-real"]) != 0:
            raise SystemExit(f"{a}-imp-real = {keys[f'{a}-imp-real']}, but the table has no such column")
    return runs


def runs_per_row(rows: dict[str, int]) -> str:
    """``$n = 80$ runs per row``; when some rows have more, they are named after it."""
    labels = {arr: re.sub(r"\\textbf\{([^}]*)\}|\$[^$]*\$", r"\1", label) for arr, label in (r for r in ROWS if r)}  # plain in the head
    usual = Counter(rows.values()).most_common(1)[0][0]
    text = f"$n = {usual}$ runs per row"
    for n in sorted(set(rows.values()) - {usual}):
        text += f"; $n = {n}$ for " + " and ".join(labels[a] for a in rows if rows[a] == n)
    return text


def table(keys: dict[str, str], source: str, caption: str, label: str) -> str:
    runs = check(keys)
    widths = [len(cols) for _, _, cols in GROUPS]
    spec = "@{}l " + " p{1em} ".join("r" * w for w in widths) + "@{}"
    # column numbers of each group, 1-based, with the label column first and a spacer between groups
    first = 2
    rules = []
    heads = []
    # When some rows have more runs the head of their group is long: both heads are then set on
    # two lines, so that the long one does not push its group's last column away from the others.
    two_lines = any(len(set(rows.values())) > 1 for rows in runs.values())
    for i, (group, heading, cols) in enumerate(GROUPS):
        last = first + len(cols) - 1
        rules.append(("\\cmidrule(lr)" if i == 0 else "\\cmidrule(l)") + f"{{{first}-{last}}}")
        head = f"\\shortstack{{{heading}\\\\({runs_per_row(runs[group])})}}" if two_lines else f"{heading} ({runs_per_row(runs[group])})"
        heads.append(f"\\multicolumn{{{len(cols)}}}{{c}}{{{head}}}")
        first = last + 2
    out = [
        f"% Generated by paper/endings_table.py from {source}; do not edit.",
        "\\begin{table*}[t]",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{6pt}",
        f"  \\begin{{tabular}}{{{spec}}}",
        "    \\toprule",
        "    & " + " &\n    & ".join(heads) + " \\\\",
        "    " + " ".join(rules),
        "    Architecture",
        "      & " + " &\n      & ".join(" & ".join(h for _, h in cols) for _, _, cols in GROUPS) + " \\\\",
        "    \\midrule",
    ]
    for row in ROWS:
        if row is None:
            out.append("    \\midrule")
            continue
        arr, row_label = row
        cells = []
        for group, _, cols in GROUPS:
            cells.append(" & ".join(f"{percent(count(keys, arr, group, c), runs[group][arr]):>4}" for c, _ in cols))
        out.append(f"    {row_label}")
        out.append("      & " + " &\n      & ".join(cells) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def counts_shown(keys: dict[str, str]) -> str:
    """The counts behind the table, for checking it against the reading."""
    lines = []
    for group, heading, cols in GROUPS:
        lines.append(f"  {heading}: runs | " + " | ".join(c for c, _ in cols))
        for arr, _ in (r for r in ROWS if r):
            lines.append(f"    {arr:<9} {keys[f'{arr}-{group}-n']:>4} | " + " | ".join(f"{count(keys, arr, group, c):>3}" for c, _ in cols))
    return "\n".join(lines)


def standalone(body: str) -> str:
    return "\n".join([
        "\\documentclass[10pt]{article}",
        "\\usepackage[letterpaper,landscape,margin=2cm]{geometry}",
        "\\usepackage{booktabs}",
        "\\begin{document}",
        body.replace("\\begin{table*}[t]", "\\begin{table}[htbp]").replace("\\end{table*}", "\\end{table}").rstrip("\n"),
        "\\end{document}",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reading", type=Path, nargs="+", default=None, help="one reading, or several counted together (default: results/esci-reading.json)")
    ap.add_argument("--seeds", choices=("all", "shared"), default="all", help="where a repair exists: every run, or the seeds every architecture has")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables" / "tab_endings.tex")
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--caption", default=CAPTION, help="the caption, in place of the one set in this file")
    ap.add_argument("--where", default=None, help='words put after "Outcomes by architecture" in the caption, to name the backend: "on the Wayfair WANDS backend" (default for the first backend\'s reading: "on the Amazon ESCI backend")')
    ap.add_argument("--standalone", action="store_true", help="wrap the table in a compilable document")
    args = ap.parse_args()
    readings = args.reading or [ROOT / "results" / "esci-reading.json"]
    keys, source = read_readings(readings, args.seeds), " and ".join(p.name for p in readings)
    source += ", where a repair exists over " + ("the seeds every architecture has" if args.seeds == "shared" else "every run")
    where = args.where or (FIRST_BACKEND if not args.reading else None)
    caption = args.caption.replace("Outcomes by architecture,", f"Outcomes by architecture {where},", 1) if where else args.caption
    body = table(keys, source, caption, args.label)
    args.out.write_text(standalone(body) if args.standalone else body, encoding="utf-8")
    print(f"wrote {args.out} from {source}")
    print(counts_shown(keys))
    every = read_readings(readings, "all")  # the two with further seeds, over all their runs where a repair exists
    print("  where a repair exists, over every run: " + "; ".join(
        f"{arm} real fix {every[f'{arm}-cap-real']}/{every[f'{arm}-cap-n']} = {percent(int(every[f'{arm}-cap-real']), int(every[f'{arm}-cap-n']))}%" for arm in ARMS))


if __name__ == "__main__":
    main()
