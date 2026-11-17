# SPDX-License-Identifier: Apache-2.0

"""Stage one: the three WANDS files to the raw subset in ``data/raw/``.

WANDS ships tab-separated text (``product.csv``, ``query.csv``, ``label.csv``), read here with
the ``csv`` module. The source directory is downloaded data: it is only ever read, by path,
and never imported from or executed. A product id becomes the catalog's id here (``B0WF000123``
for WANDS product 123) and nothing else changes: names, descriptions and features are written
as the source has them, and the build stage cleans them.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .select import Selection, SelectionParams, finalize, select_queries

PRODUCTS_FILE = "product.csv"
QUERIES_FILE = "query.csv"
LABELS_FILE = "label.csv"
SOURCE_FILES = (PRODUCTS_FILE, QUERIES_FILE, LABELS_FILE)

PRODUCT_COLUMNS = (
    "product_name",
    "product_class",
    "category hierarchy",
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
)

# A product's features are one field of a few thousand characters; the csv module's default
# limit is not far above the longest.
_FIELD_LIMIT = 1 << 24


def product_key(wands_id: str | int) -> str:
    """The catalog's id for a WANDS product id: ``B0WF`` and the id in six digits. WANDS
    numbers its products; the catalog's ids have the ten-character shape of the ESCI
    example's, which the assistant is deployed to recognise in a customer's message."""
    return f"B0WF{int(wands_id):06d}"


def read_table(path: Path) -> Iterator[dict[str, str]]:
    """The rows of one WANDS file, keyed by its header."""
    csv.field_size_limit(_FIELD_LIMIT)
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        for row in reader:
            yield dict(zip(header, row, strict=True))


def read_judgments(source_dir: Path) -> list[dict[str, Any]]:
    """Every judgment of ``label.csv`` in the file's order, with its query's text."""
    queries = {row["query_id"]: row["query"] for row in read_table(source_dir / QUERIES_FILE)}
    return [
        {
            "query_id": int(row["query_id"]),
            "query": queries[row["query_id"]],
            "product_id": product_key(row["product_id"]),
            "label": row["label"],
        }
        for row in read_table(source_dir / LABELS_FILE)
        if row["query_id"] in queries
    ]


def read_products(source_dir: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """The rows of ``product.csv`` whose id is wanted; an empty field is None."""
    products: dict[str, dict[str, Any]] = {}
    for row in read_table(source_dir / PRODUCTS_FILE):
        product_id = product_key(row["product_id"])
        if product_id in wanted:
            products[product_id] = {"product_id": product_id} | {
                column: row[column] or None for column in PRODUCT_COLUMNS
            }
    return products


def select_from_rows(
    judgments: Iterable[Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    params: SelectionParams,
) -> tuple[Selection, dict[str, dict[str, Any]]]:
    """The selection over in-memory rows: the queries first, then the products they need,
    then the product filters. ``products`` may hold more rows than the selection needs."""
    selection = select_queries(judgments, params)
    wanted = {pid: dict(products[pid]) for pid in selection.product_ids if pid in products}
    selection = finalize(selection, wanted)
    return selection, {pid: wanted[pid] for pid in sorted(selection.product_ids)}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_raw(
    out_dir: Path,
    selection: Selection,
    products: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, str],
) -> dict[str, Any]:
    """``products.jsonl``, ``examples.jsonl`` and ``selection.json`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "products.jsonl", [products[pid] for pid in sorted(products)])
    write_jsonl(out_dir / "examples.jsonl", selection.judgments)
    summary = selection.summary() | {"sources": dict(sources)}
    (out_dir / "selection.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def extract(source_dir: Path, out_dir: Path, params: SelectionParams) -> dict[str, Any]:
    """The whole stage: select queries, fetch their products, filter, write the raw subset."""
    selection = select_queries(read_judgments(source_dir), params)
    products = read_products(source_dir, selection.product_ids)
    selection = finalize(selection, products)
    chosen = {pid: products[pid] for pid in sorted(selection.product_ids)}
    sources = {name: sha256_of(source_dir / name) for name in SOURCE_FILES}
    return write_raw(out_dir, selection, chosen, sources)
