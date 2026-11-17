# SPDX-License-Identifier: Apache-2.0

"""The pipeline's command line, run with ``examples/`` on the path:

    PYTHONPATH=examples python -m esci_retail.pipeline extract --source-dir <esci parquet dir>
    PYTHONPATH=examples python -m esci_retail.pipeline build
    PYTHONPATH=examples python -m esci_retail.pipeline eval-search
    PYTHONPATH=examples python -m esci_retail.pipeline all --source-dir <esci parquet dir>

``extract`` needs pyarrow; the other two need only the repository's own packages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import build_catalog
from .search_eval import format_report
from .search_eval import run as run_search_eval
from .select import SelectionParams

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DEFAULTS = SelectionParams()


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--locale", default=DEFAULTS.locale)
    parser.add_argument("--version", default=DEFAULTS.version, choices=("small", "large"))
    parser.add_argument("--n-queries", type=int, default=DEFAULTS.n_queries)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--min-exact", type=int, default=DEFAULTS.min_exact)
    parser.add_argument("--min-irrelevant", type=int, default=DEFAULTS.min_irrelevant)
    parser.add_argument("--min-judged", type=int, default=DEFAULTS.min_judged)
    parser.add_argument("--max-judged", type=int, default=DEFAULTS.max_judged)


def params_from(args: argparse.Namespace) -> SelectionParams:
    return SelectionParams(
        locale=args.locale,
        version=args.version,
        n_queries=args.n_queries,
        seed=args.seed,
        min_exact=args.min_exact,
        min_irrelevant=args.min_irrelevant,
        min_judged=args.min_judged,
        max_judged=args.max_judged,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m esci_retail.pipeline",
        description=(__doc__ or "").partition("\n")[0],
        epilog=(__doc__ or "").partition("\n")[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="parquet files -> data/raw/ (needs pyarrow)")
    extract.add_argument("--source-dir", type=Path, required=True, help="the two parquet files")
    extract.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    add_selection_arguments(extract)

    build = commands.add_parser("build", help="data/raw/ -> catalog.sqlite, gold, orders, manifest")
    build.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    build.add_argument("--data-dir", type=Path, default=DATA_DIR)
    build.add_argument("--seed", type=int, default=DEFAULTS.seed)

    evaluate = commands.add_parser("eval-search", help="recall of the Exact products at k")
    evaluate.add_argument("--data-dir", type=Path, default=DATA_DIR)
    evaluate.add_argument("-k", type=int, default=8)

    everything = commands.add_parser("all", help="extract, build and eval-search in one go")
    everything.add_argument("--source-dir", type=Path, required=True)
    everything.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    everything.add_argument("--data-dir", type=Path, default=DATA_DIR)
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
        manifest = build_catalog(args.raw_dir, args.data_dir, seed=args.seed)
        print(
            "built:", json.dumps({k: v for k, v in manifest["counts"].items() if k != "categories"})
        )
        print("categories:", json.dumps(manifest["counts"]["categories"]))
    if args.command in ("eval-search", "all"):
        print(format_report(run_search_eval(args.data_dir, args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
