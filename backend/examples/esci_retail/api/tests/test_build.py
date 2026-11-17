# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from dataclasses import replace

from esci_retail.pipeline.catalog import (
    GENERAL,
    bullets_of,
    categorize,
    clean_text,
    synthesize,
)
from esci_retail.pipeline.extract import select_from_rows
from esci_retail.pipeline.select import exclusion_reason, select_queries
from shopping_agent import ProductDetails

from .conftest import SAMPLE_PARAMS, build_sample, sample_rows


def test_clean_text_strips_markup_and_decodes_entities():
    assert (
        clean_text("<b>Two-person tent</b> for trips.<br>Freestanding &amp; light  <p>x</p>")
        == "Two-person tent for trips. Freestanding & light x"
    )
    assert clean_text(None) == "" and clean_text("   ") == ""


def test_bullets_split_on_newlines_and_drop_repeats():
    assert bullets_of("Pack of 1\nSoft fabric\npack of 1\n\nok\nSoft fabric") == [
        "Pack of 1",
        "Soft fabric",
    ]
    assert bullets_of(None) == []


def test_categorize_prefers_phrases_and_the_title():
    assert categorize("Northwind 2-Person Backpacking Tent") == "sports-outdoors"
    assert categorize("BrewLine 12-Cup Programmable Drip Coffee Maker") == "home-kitchen"
    assert categorize("Women's Cloudrun Running Shoes, Black") == "clothing-shoes"
    assert categorize("PawRest Orthopedic Memory Foam Dog Bed, Large") == "pet-supplies"
    assert categorize("Aurex Wireless Noise Cancelling Headphones") == "electronics"
    assert categorize("Plixio 61 Key Digital Electric Piano Keyboard") == "musical-instruments"
    assert categorize("Mystery Widget", ["works with any tent"]) == "sports-outdoors"
    assert categorize("Mystery Widget") == GENERAL


def test_synthesis_depends_only_on_seed_and_id():
    first = synthesize(7, "B0TEST0001", "sports-outdoors")
    assert first == synthesize(7, "B0TEST0001", "sports-outdoors")
    assert first["price"] != synthesize(8, "B0TEST0001", "sports-outdoors")["price"]
    assert 1.99 <= first["price"] <= 2999 and 3.0 <= first["rating"] <= 5.0
    assert first["review_count"] >= 0 and isinstance(first["in_stock"], bool)
    prices = {synthesize(7, f"B0TEST{n:04d}", "grocery")["price"] for n in range(40)}
    assert len(prices) > 15  # a spread, not a constant


def test_exclusion_reasons():
    assert exclusion_reason("lightweight tent") is None
    assert exclusion_reason("champu para perros blancos") == "non_english"
    assert exclusion_reason("pistol holster right hand") == "excluded_content"
    assert exclusion_reason("電気ケトル") == "non_ascii"
    assert exclusion_reason("glue gun for crafts") is None  # bare "gun" is not excluded
    assert exclusion_reason("unisex rain jacket") is None  # word boundaries
    assert exclusion_reason("") == "empty"


def test_selection_filters_queries_and_products_and_is_deterministic():
    judgments, products = sample_rows()
    selection, chosen = select_from_rows(judgments, products, SAMPLE_PARAMS)
    assert set(selection.queries) == {1, 2, 3, 4, 5, 6}
    assert selection.excluded["query_excluded_content"] == 1  # q7
    assert selection.excluded["query_non_english"] == 1  # q8 (q11 is dropped by the locale filter)
    assert selection.excluded["query_thresholds"] == 1  # q9
    assert selection.excluded["product_excluded_content"] == 1  # the holster under q4
    assert selection.excluded["product_non_ascii"] == 1  # the kettle under q10
    assert selection.excluded["query_thresholds_after_products"] == 1  # q10 lost its Exact
    assert not {"B0TEST0034", "B0TEST0035", "B0TEST0036"} & set(chosen)
    assert len(chosen) == 33
    again, _ = select_from_rows(judgments, products, SAMPLE_PARAMS)
    assert again.judgments == selection.judgments
    fewer = select_queries(judgments, replace(SAMPLE_PARAMS, n_queries=2))
    assert len(fewer.queries) == 2 and set(fewer.queries) <= {1, 2, 3, 4, 5, 6}


def test_build_writes_a_valid_catalog(tmp_path):
    manifest = build_sample(tmp_path)
    counts = manifest["counts"]
    assert (counts["products"], counts["queries"], counts["judgments"]) == (33, 6, 33)
    assert 0 < counts["in_stock"] <= 33
    assert manifest["counts"]["categories"]["pet-supplies"] >= 3
    db = sqlite3.connect(tmp_path / "catalog.sqlite")
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {
        "products",
        "products_fts",
        "products_trigram",
        "queries",
        "judgments",
        "meta",
    } <= tables
    records = [json.loads(row[0]) for row in db.execute("SELECT record FROM products")]
    assert len(records) == 33
    for record in records:
        product = ProductDetails.model_validate(record)
        assert product.price > 0 and product.category and product.short_description
    assert (
        db.execute("SELECT COUNT(*) FROM products_fts WHERE products_fts MATCH 'tent'").fetchone()[
            0
        ]
        >= 3
    )
    gold = json.loads((tmp_path / "queries_gold.json").read_text(encoding="utf-8"))
    assert [q["query_id"] for q in gold["queries"]] == [1, 2, 3, 4, 5, 6]
    assert gold["queries"][1]["E"] == ["B0TEST0007"] and gold["queries"][3]["I"] == [
        "B0TEST0021",
        "B0TEST0023",
    ]
    orders = json.loads((tmp_path / "orders.json").read_text(encoding="utf-8"))
    assert orders["dates_anchored_to"] == "2026-11-06" and len(orders["orders"]) == 5
    ids = {record["product_id"] for record in records}
    assert all(item["product_id"] in ids for order in orders["orders"] for item in order["items"])


def test_two_builds_with_one_seed_are_identical(tmp_path):
    build_sample(tmp_path / "a")
    build_sample(tmp_path / "b")
    rows = lambda path: list(  # noqa: E731
        sqlite3.connect(path / "catalog.sqlite").execute("SELECT product_id, record FROM products")
    )
    assert rows(tmp_path / "a") == rows(tmp_path / "b")
    other = build_sample(tmp_path / "c", seed=8)
    assert other["seed"] == 8 and rows(tmp_path / "c") != rows(tmp_path / "a")
