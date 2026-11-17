# SPDX-License-Identifier: Apache-2.0

"""Stage one: the two Shopping Queries Dataset parquet files to the raw subset in
``data/raw/``. Only this module reads parquet (pyarrow, imported where it is used), so the
rest of the pipeline and the tests run without it. The source directory is downloaded
data: it is only ever read, by path, and never imported from or executed."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .select import Selection, SelectionParams, finalize, select_queries

EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"
PRODUCT_COLUMNS = (
    "product_id",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
    "product_locale",
)
JUDGMENT_COLUMNS = (
    "query_id",
    "query",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
    "large_version",
    "split",
)


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


def read_judgments(source_dir: Path, params: SelectionParams) -> list[dict[str, Any]]:
    """Every judgment of the locale in the chosen version, as dicts."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(source_dir / EXAMPLES_FILE, columns=list(JUDGMENT_COLUMNS))
    mask = pc.and_(
        pc.equal(table["product_locale"], params.locale),
        pc.equal(table[f"{params.version}_version"], 1),
    )
    return table.filter(mask).to_pylist()


def read_products(source_dir: Path, locale: str, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """The listings in ``wanted`` for the locale, streamed in batches so the 1.1 GB file is
    never held whole."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted_ids = pa.array(sorted(wanted))
    found: dict[str, dict[str, Any]] = {}
    reader = pq.ParquetFile(source_dir / PRODUCTS_FILE)
    for batch in reader.iter_batches(batch_size=65_536, columns=list(PRODUCT_COLUMNS)):
        table = pa.Table.from_batches([batch])
        mask = pc.and_(
            pc.equal(table["product_locale"], locale),
            pc.is_in(table["product_id"], value_set=wanted_ids),
        )
        for row in table.filter(mask).to_pylist():
            found[row["product_id"]] = row
    return found


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
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
    selection = select_queries(read_judgments(source_dir, params), params)
    products = read_products(source_dir, params.locale, selection.product_ids)
    selection = finalize(selection, products)
    chosen = {pid: products[pid] for pid in sorted(selection.product_ids)}
    sources = {name: sha256_of(source_dir / name) for name in (EXAMPLES_FILE, PRODUCTS_FILE)}
    return write_raw(out_dir, selection, chosen, sources)
