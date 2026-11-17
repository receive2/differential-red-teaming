# SPDX-License-Identifier: Apache-2.0

"""The catalog pipeline, in two stages so the heavy one runs once:

1. ``extract`` (needs pyarrow): the two ESCI parquet files -> ``data/raw/`` (the chosen
   queries' judgments, their products, and ``selection.json``).
2. ``build`` (standard library): ``data/raw/`` -> ``data/catalog.sqlite`` (records plus two
   FTS5 indexes), ``queries_gold.json``, ``orders.json``, ``manifest.json``.

``eval-search`` reports how well the store's search surfaces the Exact-labelled products.
Everything is deterministic for a (parameters, seed) pair: ``python -m esci_retail.pipeline``.
"""
