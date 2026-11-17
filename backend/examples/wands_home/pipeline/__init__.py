# SPDX-License-Identifier: Apache-2.0

"""The catalog pipeline, in two stages as the ESCI example's:

1. ``extract``: the three WANDS files -> ``data/raw/`` (the chosen queries' judgments, their
   products as the source has them, and ``selection.json``).
2. ``build``: ``data/raw/`` -> ``data/catalog.sqlite`` (records plus two FTS5 indexes),
   ``queries_gold.json``, ``orders.json``, ``manifest.json``.

``eval-search`` reports how well the backend's search surfaces the Exact-labelled products.
Both stages are standard library only and deterministic for a (parameters, seed, anchor)
triple: ``python -m wands_home.pipeline``.
"""
