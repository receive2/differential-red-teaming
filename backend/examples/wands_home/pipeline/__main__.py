# SPDX-License-Identifier: Apache-2.0

"""The pipeline's command line, run with ``examples/`` on the path:

    PYTHONPATH=examples python -m wands_home.pipeline extract --source-dir <WANDS dataset dir>
    PYTHONPATH=examples python -m wands_home.pipeline build
    PYTHONPATH=examples python -m wands_home.pipeline eval-search
    PYTHONPATH=examples python -m wands_home.pipeline all --source-dir <WANDS dataset dir>

The source directory holds ``product.csv``, ``query.csv`` and ``label.csv`` of the WANDS
dataset. ``extract`` and ``build`` need the standard library only; ``eval-search`` needs the
repository's own packages.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .catalog import ANCHOR, build_catalog
from .select import SelectionParams

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DEFAULTS = SelectionParams()


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-queries", type=int, default=DEFAULTS.n_queries)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--min-exact", type=int, default=DEFAULTS.min_exact)
    parser.add_argument("--min-irrelevant", type=int, default=DEFAULTS.min_irrelevant)
    parser.add_argument("--min-judged", type=int, default=DEFAULTS.min_judged)
    parser.add_argument("--max-judged", type=int, default=DEFAULTS.max_judged)


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--anchor",
        type=date.fromisoformat,
        default=ANCHOR,
        help="the day the demo orders are dated from (default: %(default)s)",
    )


def params_from(args: argparse.Namespace) -> SelectionParams:
    return SelectionParams(
        n_queries=args.n_queries,
        seed=args.seed,
        min_exact=args.min_exact,
        min_irrelevant=args.min_irrelevant,
        min_judged=args.min_judged,
        max_judged=args.max_judged,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wands_home.pipeline",
        description=(__doc__ or "").partition("\n")[0],
        epilog=(__doc__ or "").partition("\n")[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="the three WANDS files -> data/raw/")
    extract.add_argument("--source-dir", type=Path, required=True, help="the three WANDS files")
    extract.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    add_selection_arguments(extract)

    build = commands.add_parser("build", help="data/raw/ -> catalog.sqlite, gold, orders, manifest")
    add_build_arguments(build)
    build.add_argument("--seed", type=int, default=DEFAULTS.seed)

    evaluate = commands.add_parser("eval-search", help="recall of the Exact products at k")
    evaluate.add_argument("--data-dir", type=Path, default=DATA_DIR)
    evaluate.add_argument("-k", type=int, default=8)

    everything = commands.add_parser("all", help="extract, build and eval-search in one go")
    everything.add_argument("--source-dir", type=Path, required=True)
    add_build_arguments(everything)
    everything.add_argument("-k", type=int, default=8)
    add_selection_arguments(everything)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("extract", "all"):
        from .extract import extract

        summary = extract(args.source_dir, args.raw_dir, params_from(args))
        print("extracted:", json.dumps(summary["counts"]))
        print("excluded:", json.dumps(summary["excluded"]))
    if args.command in ("build", "all"):
        manifest = build_catalog(args.raw_dir, args.data_dir, seed=args.seed, anchor=args.anchor)
        counts = manifest["counts"]
        print("built:", json.dumps({k: v for k, v in counts.items() if k != "departments"}))
        print("departments:", json.dumps(counts["departments"]))
        print("records:", manifest["records_sha256"])
    if args.command in ("eval-search", "all"):
        from .search_eval import format_report
        from .search_eval import run as run_search_eval

        print(format_report(run_search_eval(args.data_dir, args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
