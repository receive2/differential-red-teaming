# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from dataclasses import replace

from shopping_agent import ProductDetails
from wands_home.pipeline.catalog import (
    FREIGHT_FROM_LB,
    GENERAL,
    SPECS,
    attributes_of,
    category_of,
    category_words,
    department_of,
    features_of,
    read_listing,
    slug,
    specs_of,
    states_dimension,
    synthesize,
    weight_of,
    width_of,
)
from wands_home.pipeline.extract import product_key, select_from_rows
from wands_home.pipeline.select import select_queries, unusable
from wands_home.pipeline.text import (
    clean_description,
    clean_name,
    first_sentence,
    repair,
    sentence_case,
    shorten,
    states_length,
)

from .conftest import SAMPLE_PARAMS, build_sample, sample_rows


def test_text_gets_its_spacing_its_characters_and_its_case_back():
    assert (
        repair("it 's finished on all sides , so it can float in a room .")
        == "it's finished on all sides, so it can float in a room."
    )
    assert repair("dã©cor for the kids ' room") == "décor for the kids' room"
    assert (
        clean_name("arlo 47.2 '' w height adjustable standing desk")
        == "Arlo Height Adjustable Standing Desk"
    )
    assert (
        clean_name("led light bulb , 60 watt equivalent ( set of 4 )")
        == "LED Light Bulb, 60 Watt Equivalent (Set of 4)"
    )
    assert clean_name("3 - light kitchen island pendant") == "3-Light Kitchen Island Pendant"
    assert (
        clean_description(
            "a cotton cover with a mid-century print ; the insert is sold separately . "
            "it â€™s machine washable ."
        )
        == "A cotton cover with a mid-century print; the insert is sold separately. "
        "It's machine washable."
    )
    assert clean_name(None) == "" and clean_description("   ") == ""


def test_the_first_sentence_does_not_end_at_an_abbreviation():
    text = sentence_case(repair("measures 12 in . wide and 30 in . high . assembly is required ."))
    assert text == "Measures 12 in. wide and 30 in. high. Assembly is required."
    assert first_sentence(text) == "Measures 12 in. wide and 30 in. high."
    assert first_sentence("One sentence only") == "One sentence only"
    assert shorten("a desk for a small corner", 14) == "a desk for..."
    assert shorten("a desk for a small corner", 14, ellipsis="") == "a desk for a"
    assert shorten("a desk", 14) == "a desk"


def test_a_statement_of_a_length_leaves_a_name_and_takes_its_sentence_from_a_description():
    assert clean_name("liberty 18 '' wide square pouf ottoman") == "Liberty Square Pouf Ottoman"
    assert (
        clean_name("fluted 4 '' h x 94 1/2 '' w x 3/8 '' d panel moulding")
        == "Fluted Panel Moulding"
    )
    assert clean_name("skofte tv stand for tvs up to 60 ''") == "Skofte TV Stand for TVs"
    assert (
        clean_name("4 ft . h x 8.5 ft . w outdoor wood privacy screen")
        == "Outdoor Wood Privacy Screen"
    )
    assert (
        clean_name("studio set ( 1x 32inch table + 2x chairs )") == "Studio Set (Table + 2x Chairs)"
    )
    assert (
        clean_name("antek 86 '' x 98 '' l-shape reception desk") == "Antek L-Shape Reception Desk"
    )
    # Not lengths: "3 in 1", a wattage, a count of people, a weight.
    assert (
        clean_name("3 in 1 floating headboard , 60 w led") == "3 in 1 Floating Headboard, 60 W LED"
    )
    assert (
        clean_description("seats 4 in comfort and holds 300 lb .")
        == "Seats 4 in comfort and holds 300 lb."
    )
    assert (
        clean_description(
            "a storage bench in black . it measures 18 '' high x 40 '' wide x 16 '' deep . "
            "assembly is required ."
        )
        == "A storage bench in black. Assembly is required."
    )
    assert (
        clean_description("since this design spans just 16.38 '' wide , it fits narrow nooks .")
        == ""
    )
    assert states_length('overall size: 27 5/9 x 11 4/5 x 68 8/9" (lxwxh)') and states_length(
        "47.3x23.6inch"
    )
    assert states_length("48 inch led light fixture") and not states_length("available in 2 colors")


def test_features_are_read_by_key_with_their_distinct_values():
    features = features_of(
        "overallwidth-sidetoside:59.1|color : brown|color : brown|color:white"
        "|material : manufactured wood|no colon here|empty:| :lost"
    )
    assert features == {
        "overallwidth-sidetoside": ("59.1",),
        "color": ("brown", "white"),
        "material": ("manufactured wood",),
    }
    assert features_of(None) == {}


def test_a_record_shows_no_dimension_and_the_catalog_keeps_width_and_weight():
    _judgments, products = sample_rows()
    features = features_of(products["B0WF000004"]["product_features"])
    assert states_dimension("47 in. matte black") and states_dimension('59.1" w x 55" d')
    assert not states_dimension("modern & contemporary") and not states_dimension("2")
    assert attributes_of(features) == {"color": "black", "material": "steel"}
    assert specs_of(features) == {
        "Product Type": "Standing desk",
        "Color": "Black",
        "Frame Material": "Steel",
        "Top Material": "Manufactured wood",
        "Weight Capacity": "154 lb",
        "Power Source": "Corded electric",
    }
    assert (width_of(features), weight_of(features)) == (47.2, 55.0)
    # A product listed in two sizes has no one width; its weight is the largest stated.
    two_sizes = features_of(
        "overallwidth-sidetoside:60|overallwidth-sidetoside:96"
        "|overallproductweight:22|overallproductweight:30"
    )
    assert width_of(two_sizes) is None and weight_of(two_sizes) == 30.0
    assert width_of({}) is None and weight_of({}) is None


def test_specifications_keep_names_and_initialisms_and_drop_what_says_nothing():
    assert specs_of(
        features_of("bulbtype : led|countryoforigin : united states|powersource : plug-in")
    ) == {"Power Source": "Plug-in", "Bulb Type": "LED", "Country of Origin": "United States"}
    assert specs_of(features_of("material : n/a|levelofassembly : none|finish : none")) == {
        "Level of Assembly": "None"
    }
    # A count that has two values has none to show.
    assert specs_of(features_of("numberofdrawers:2|numberofdrawers:4|numberofshelves:3")) == {
        "Number of Shelves": "3"
    }


def test_the_category_is_the_first_class_or_the_end_of_the_path():
    assert slug("Accent Chests / Cabinets") == "accent-chests-cabinets"
    assert slug("Décor & Pillows") == "decor-pillows"
    _judgments, products = sample_rows()
    kids_desk = read_listing(products["B0WF000006"])  # two classes
    assert (category_of(kids_desk), department_of(kids_desk)) == ("kids-desks", "baby-kids")
    assert category_words(kids_desk) == "kids desks baby furniture"
    cushion = read_listing(products["B0WF000033"])  # no class
    assert (category_of(cushion), department_of(cushion)) == ("outdoor-cushions", "outdoor")
    bulbs = read_listing(products["B0WF000028"])  # neither a class nor a path
    assert (category_of(bulbs), department_of(bulbs)) == (GENERAL, None)
    assert category_words(bulbs) == ""


def test_a_listing_keeps_the_source_rating_and_review_count():
    _judgments, products = sample_rows()
    desk = read_listing(products["B0WF000001"])
    assert (desk.title, desk.rating, desk.review_count) == ("Glenbrook L-Shape Desk", 4.5, 210)
    unrated = read_listing(products["B0WF000004"])
    assert unrated.rating is None and unrated.review_count is None and unrated.description == ""


def test_synthesis_depends_only_on_seed_and_id():
    first = synthesize(7, "B0WF000001", "desks", "furniture")
    assert first == synthesize(7, "B0WF000001", "desks", "furniture")
    assert first["price"] != synthesize(8, "B0WF000001", "desks", "furniture")["price"]
    assert set(first) == {"price", "in_stock"} and isinstance(first["in_stock"], bool)
    assert 1.99 <= first["price"] <= 5999
    prices = {synthesize(7, product_key(n), "desks", "furniture")["price"] for n in range(40)}
    assert len(prices) > 15  # a spread, not a constant
    # A class without a price band takes its department's, and failing that the general one.
    assert synthesize(7, "B0WF000001", "no-such-class", "lighting") == synthesize(
        7, "B0WF000001", "another-class", "lighting"
    )
    assert synthesize(7, "B0WF000001", "no-such-class", None) == synthesize(
        7, "B0WF000001", "another-class", "no-such-department"
    )
    assert product_key(123) == "B0WF000123" and product_key("42") == "B0WF000042"


def test_unusable_text():
    assert unusable("l shaped desk") is None
    assert unusable("french country décor") is None  # an accent is not another language
    assert unusable("champu para perros blancos") == "non_english"
    assert unusable("pistol holster right hand") == "excluded_content"
    assert unusable("電気ケトル 1.0l") == "non_ascii"
    assert unusable("") == "empty"


def test_selection_filters_queries_and_products_and_is_deterministic():
    judgments, products = sample_rows()
    selection, chosen = select_from_rows(judgments, products, SAMPLE_PARAMS)
    assert set(selection.queries) == {1, 2, 3, 4, 5, 6} and selection.eligible == 7
    assert dict(selection.excluded) == {
        "judgment_repeated": 1,  # q1 judges the writing desk twice
        "query_excluded_content": 1,  # q7
        "query_non_english": 1,  # q8
        "query_thresholds": 1,  # q9 has two judgments
        "product_excluded_content": 1,  # the holster under q4
        "product_non_ascii": 1,  # the kettle under q10
        "product_missing": 1,  # q6 judges a product the table lacks
        "query_thresholds_after_products": 1,  # q10 lost its Exact
    }
    first = {(row["query_id"], row["product_id"]): row["label"] for row in selection.judgments}
    assert first[(1, "B0WF000003")] == "Partial"  # the first of the two judgments stands
    assert not {"B0WF000023", "B0WF000035", "B0WF000036", "B0WF000037", "B0WF000038"} & set(chosen)
    assert len(chosen) == 33 and len(selection.judgments) == 34
    again, _ = select_from_rows(judgments, products, SAMPLE_PARAMS)
    assert again.judgments == selection.judgments
    fewer = select_queries(judgments, replace(SAMPLE_PARAMS, n_queries=2))
    assert len(fewer.queries) == 2 and set(fewer.queries) <= {1, 2, 3, 4, 5, 6, 10}


def test_build_writes_a_valid_catalog(tmp_path):
    manifest = build_sample(tmp_path)
    counts = manifest["counts"]
    assert (counts["products"], counts["queries"], counts["judgments"]) == (33, 6, 34)
    assert 0 < counts["in_stock"] <= 33
    assert (counts["with_width"], counts["with_weight"], counts["freight"]) == (23, 33, 11)
    assert counts["departments"]["furniture"] == 15 and counts["departments"][GENERAL] == 1
    assert manifest["orders_anchored_to"] == "2026-11-16"
    assert manifest["sources"] == {"fixture": "wands_sample"}
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
    shown = {name for name, _keys, _unit in SPECS}
    for record in records:
        product = ProductDetails.model_validate(record)
        assert product.price > 0 and product.category and product.short_description
        assert set(product.attributes) <= {"color", "material", "style"}
        assert set(product.specs) <= shown
        assert not any(states_dimension(value) for value in product.specs.values())
    # The desk's width and weight are columns of the catalog; its record states neither.
    desk = next(record for record in records if record["product_id"] == "B0WF000001")
    assert "59.1" not in json.dumps(desk) and "85" not in json.dumps(desk)
    assert db.execute(
        "SELECT width_in, weight_lb, category_words FROM products WHERE product_id = 'B0WF000001'"
    ).fetchone() == (59.1, 85.0, "desks furniture office")
    assert db.execute(
        "SELECT width_in, weight_lb FROM products WHERE product_id = 'B0WF000013'"
    ).fetchone() == (None, 22.0)
    assert (
        db.execute("SELECT COUNT(*) FROM products_fts WHERE products_fts MATCH 'desk'").fetchone()[
            0
        ]
        >= 5
    )
    assert (
        json.loads(
            db.execute("SELECT value FROM meta WHERE key = 'orders_anchored_to'").fetchone()[0]
        )
        == "2026-11-16"
    )
    gold = json.loads((tmp_path / "queries_gold.json").read_text(encoding="utf-8"))
    assert [q["query_id"] for q in gold["queries"]] == [1, 2, 3, 4, 5, 6]
    assert gold["queries"][0]["E"] == ["B0WF000001", "B0WF000002"]
    assert gold["queries"][0]["S"] == ["B0WF000003", "B0WF000004", "B0WF000006"]
    assert gold["queries"][0]["C"] == [] and gold["queries"][0]["I"] == ["B0WF000005"]
    assert gold["queries"][4]["S"] == ["B0WF000017", "B0WF000027", "B0WF000028"]
    orders = json.loads((tmp_path / "orders.json").read_text(encoding="utf-8"))
    assert set(orders) == {"_comment", "orders"} and len(orders["orders"]) == 5
    in_stock = {record["product_id"] for record in records if record["in_stock"]}
    assert all(
        item["product_id"] in in_stock for order in orders["orders"] for item in order["items"]
    )
    # The delivered order has one line that shipped by freight and one that did not.
    weights = dict(db.execute("SELECT product_id, weight_lb FROM products"))
    delivered = next(order for order in orders["orders"] if order["status"] == "delivered")
    assert sorted(
        weights[item["product_id"]] >= FREIGHT_FROM_LB for item in delivered["items"]
    ) == [
        False,
        True,
    ]
    assert delivered["estimated_delivery"] == "2026-10-28"


def test_two_builds_with_one_seed_are_identical(tmp_path):
    build_sample(tmp_path / "a")
    build_sample(tmp_path / "b")
    rows = lambda path: list(  # noqa: E731
        sqlite3.connect(path / "catalog.sqlite").execute(
            "SELECT product_id, record, category_words, width_in, weight_lb FROM products"
        )
    )
    assert rows(tmp_path / "a") == rows(tmp_path / "b")
    # No output holds the time it was built: the files the repository tracks are the same.
    for name in ("manifest.json", "orders.json", "queries_gold.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    other = build_sample(tmp_path / "c", seed=8)
    assert other["seed"] == 8 and rows(tmp_path / "c") != rows(tmp_path / "a")
