#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""The table of how the checker's release decisions depend on its thresholds:
results/tables/tab_sensitivity.tex, from the file experiment/sensitivity.py writes for an
experiment (``--out <stem>`` gives ``<stem>.json``), so that the table holds no number of
its own.

    python3 paper/sensitivity_table.py                                      # the first backend: results/esci-sensitivity.json
    python3 paper/sensitivity_table.py --sensitivity results/esci-sensitivity.json results/wands-sensitivity.json --names ESCI WANDS --out results/tables/tab_sensitivity_both.tex
    python3 paper/sensitivity_table.py --no-redteam                         # without the last row

A row changes one threshold of the rule and leaves the others as run. "Lost" counts the
releases the runs rejected that the variant accepts. "Added" counts the releases the runs
delivered that the variant rejects, and of those the ones whose delivered build is labelled
a facade, a real fix and a disclosure. The last row is the plain red team's own rule at
half its threshold. The rows, their labels, the heads and the caption are the settings
below; every number comes from the file. The table is one column wide."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The rule as run (experiment/compare.py) and the variants shown: (label, what is changed).
# A label reads "threshold: as run -> variant".
AS_RUN = {"threshold": 2, "vote": "half", "min_showing": 1, "before_max": 2}
ROWS = [
    ("Asserting grades: 1--2 $\\to$ 1", {"threshold": 1}),
    ("Asserting grades: 1--2 $\\to$ 1--3", {"threshold": 3}),
    ("Asserting share: half $\\to$ all", {"vote": "all"}),
    ("Asserting share: half $\\to$ any", {"vote": "any"}),
    ("Min.\\ qualifying: 1 $\\to$ 2", {"min_showing": 2}),
]
# The plain red team's rule (experiment/loop.py): as run, and the variant shown.
FINDINGS_AS_RUN = {"share": 0.2, "min_asks": 6}
REDTEAM_ROW = ("RedTeam$^\\dagger$: 20\\% $\\to$ 10\\%", {"share": 0.1, "min_asks": 6})

# The labels of a delivered build counted in a cell, in the order of the cell.
COUNTED = ("facade", "real", "honest")
ADDED_HEAD = "Added (F/R/D)"

CAPTION = (
    "Release decisions with one threshold changed (as run $\\to$ variant), recomputed from the recorded "
    "judge grades. \\emph{Lost}: rejections that become acceptances. \\emph{Added}: delivered releases "
    "that become rejections (facades/real fixes/disclosures among them). In the head: releases DRT "
    "rejected as run."
)
CAPTION_REDTEAM = " $^\\dagger$RedTeam's own threshold (as run: {redteam} rejected)."
LABEL = "tab:sensitivity"
FIRST_BACKEND = "ESCI"


def variant(data: dict, change: dict) -> dict:
    """The file's row for the rule as run with ``change`` applied."""
    want = {**AS_RUN, **change}
    rows = [row for row in data["comparison_rule"] if all(row[k] == v for k, v in want.items())]
    if len(rows) != 1:
        raise SystemExit(f"{len(rows)} rows for {want}")
    return rows[0]


def findings(data: dict, want: dict) -> dict:
    rows = [row for row in data["findings_rule"] if all(row[k] == v for k, v in want.items())]
    if len(rows) != 1:
        raise SystemExit(f"{len(rows)} rows for {want}")
    return rows[0]["redteam"]


def check(data: dict, name: str) -> None:
    """The recorded decisions are the rule's, and the rule as run changes none."""
    consistency = data["consistency"]
    if consistency["comparisons_mismatched"] or consistency["releases_mismatched"]:
        raise SystemExit(f"{name}: the rule as run does not give the recorded decisions")
    as_run = variant(data, {})
    if as_run["deliveries_held_back"] or as_run["rejections_let_through"] or not as_run["as_run"]:
        raise SystemExit(f"{name}: the rule as run changes a decision")
    plain = findings(data, FINDINGS_AS_RUN)
    if plain["deliveries_held_back"] or plain["rejections_let_through"]:
        raise SystemExit(f"{name}: the plain red team's rule as run changes a decision")


def cells(row: dict) -> tuple[str, str]:
    """(lost, added): the added deliveries carry the counted labels among them."""
    added = row["deliveries_held_back"]
    among = "/".join(str(row["held_back_by_label"][label]) for label in COUNTED)
    return str(row["rejections_let_through"]), f"{added} ({among})" if added else "0"


def table(datas: list[dict], names: list[str], sources: list[str], label: str, redteam: bool) -> str:
    for data, name in zip(datas, names, strict=True):
        check(data, name)
    caption = CAPTION
    if redteam:
        caption += CAPTION_REDTEAM.format(redteam=" and ".join(f"{findings(d, FINDINGS_AS_RUN)['releases_rejected']}/{d['releases']['redteam']}" for d in datas))
    heads = [f"\\multicolumn{{2}}{{c}}{{{n} ({variant(d, {})['releases_rejected']}/{d['releases']['compare']})}}" for d, n in zip(datas, names, strict=True)]
    rules = [f"\\cmidrule({'lr' if i < len(datas) - 1 else 'l'}){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(datas))]
    out = [
        f"% Generated by paper/sensitivity_table.py from {' and '.join(sources)}; do not edit.",
        "\\begin{table}[t]",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3pt}",
        f"  \\begin{{tabular}}{{@{{}}l {' '.join('rr' for _ in datas)}@{{}}}}",
        "    \\toprule",
        "    & " + " & ".join(heads) + " \\\\",
        "    " + " ".join(rules),
        "    Threshold & " + " & ".join(f"Lost & {ADDED_HEAD}" for _ in datas) + " \\\\",
        "    \\midrule",
    ]
    for row_label, change in ROWS:
        out.append(f"    {row_label} & " + " & ".join(" & ".join(cells(variant(d, change))) for d in datas) + " \\\\")
    if redteam:
        out.append("    \\midrule")
        row_label, want = REDTEAM_ROW
        out.append(f"    {row_label} & " + " & ".join(" & ".join(cells(findings(d, want))) for d in datas) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def counts_shown(datas: list[dict], names: list[str], redteam: bool) -> str:
    """The counts behind the table, for checking it against the files."""
    lines = []
    for data, name in zip(datas, names, strict=True):
        lines.append(f"  {name}: as run {variant(data, {})['releases_rejected']} of {data['releases']['compare']} releases rejected")
        for _, change in ROWS:
            row = variant(data, change)
            lines.append(f"    {str(change):<22} lost {row['rejections_let_through']:>3} | added {row['deliveries_held_back']:>3} {row['held_back_by_label']}")
        if redteam:
            row = findings(data, REDTEAM_ROW[1])
            lines.append(f"    {'redteam ' + str(REDTEAM_ROW[1]['share']):<22} lost {row['rejections_let_through']:>3} | added {row['deliveries_held_back']:>3} {row['held_back_by_label']} (as run {findings(data, FINDINGS_AS_RUN)['releases_rejected']} of {data['releases']['redteam']})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensitivity", type=Path, nargs="+", default=None, help="one file of experiment/sensitivity.py, or one for each backend (default: results/esci-sensitivity.json)")
    ap.add_argument("--names", nargs="+", default=None, help="the backends' names, in the order of the files (default for one file: ESCI)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables" / "tab_sensitivity.tex")
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--no-redteam", action="store_true", help="leave out the row of the plain red team's rule")
    args = ap.parse_args()
    paths = args.sensitivity or [ROOT / "results" / "esci-sensitivity.json"]
    names = args.names or ([FIRST_BACKEND] if len(paths) == 1 else None)
    if names is None or len(names) != len(paths):
        raise SystemExit("--names: one name for each file")
    datas = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    args.out.write_text(table(datas, names, [p.name for p in paths], args.label, not args.no_redteam), encoding="utf-8")
    print(f"wrote {args.out} from {' and '.join(p.name for p in paths)}")
    print(counts_shown(datas, names, not args.no_redteam))


if __name__ == "__main__":
    main()
