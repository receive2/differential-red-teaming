# SPDX-License-Identifier: Apache-2.0

"""The backend's variants: a fulfillment service that quotes by ZIP, and one that refuses a ZIP."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from experiment.config import ROOT, ExperimentConfig
from harness.worlds import patch_start, served, standard_window, zone_distance

CFG = ExperimentConfig(name="worlds-test", mode="coupled")
pytestmark = pytest.mark.skipif(not CFG.world_tree.exists() or not CFG.subject_start.exists(), reason="no world on this machine")

PROBE = """
import asyncio, json, sys
from pathlib import Path
from harness.subject import activate
activate(Path(sys.argv[1]), Path(sys.argv[2]))
from harness.runner import World
from shopping_agent.types import ShoppingSessionContext
out = {}
for kind in ("esci", "esci-zip", "esci-refuse"):
    world = World(Path(sys.argv[2]), kind=kind)
    session = ShoppingSessionContext(session_id="s", user_id="demo-user")
    row = {}
    async def ask(**kw):
        try:
            return [o.model_dump() for o in await world.backend.get_fulfillment_options(session, ["B07XJN27NZ"], **kw)]
        except Exception as error:
            return f"{type(error).__name__}: {error}"
    row["none"] = asyncio.run(ask())
    for z in ("94110", "10001", "44137", "00000"):
        row[z] = asyncio.run(ask(zip_code=z))
    out[kind] = row
print(json.dumps(out))
"""


def _probe():
    import json
    import os

    env = {k: v for k, v in os.environ.items() if k in ("HOME", "PATH", "LANG")}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run([sys.executable, "-c", PROBE, str(CFG.subject_start), str(CFG.world_tree)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr[-800:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_three_stores_differ_only_in_what_a_zip_gets():
    out = _probe()
    base = out["esci"]["none"]
    assert [o["method"] for o in base] == ["delivery", "delivery", "pickup"]
    assert out["esci-zip"]["none"] == base and out["esci-refuse"]["none"] == base  # no ZIP: the same answer everywhere
    assert out["esci"]["94110"].startswith("TypeError")  # upstream's service has no such parameter
    assert out["esci-refuse"]["94110"].startswith("NotOffered") and "ZIP" in out["esci-refuse"]["94110"]
    far, mid, home, nowhere = (out["esci-zip"][z] for z in ("94110", "10001", "44137", "00000"))
    assert [o["eta"] for o in far] == ["4-6 business days", "3 business days (express)"] and far[1]["fee"] == 14.99
    assert [o["eta"] for o in mid] == ["3-4 business days", "2 business days (express)"]
    assert all(o["location"] == "ZIP 94110" for o in far)
    assert [o["method"] for o in home] == ["delivery", "delivery", "pickup"]  # pickup only near the backend
    assert nowhere == []  # outside the delivery area: nothing to quote
    assert base[0]["eta"] not in (far[0]["eta"], mid[0]["eta"])  # a ZIP quote is never the saved location's


def test_distance_bands_and_the_delivery_area():
    assert zone_distance("94110", "441") == 5 and zone_distance("10001", "441") == 3
    assert [standard_window(d) for d in (0, 1, 2, 3, 4, 5, 6, 9)] == ["2-3 business days"] * 2 + ["3-4 business days"] * 2 + ["4-6 business days"] * 2 + ["5-8 business days"] * 2
    assert served("94110") and not served("00000") and not served("99950") and not served("9411")


def test_the_patched_start_shows_the_parameter_and_still_passes_its_tests(tmp_path):
    work = tmp_path / "start"
    shutil.copytree(CFG.subject_start, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    patch_start(work)
    assert "zip_code: str | None = None" in (work / "shopping-agent/core/shopping_agent/backend.py").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        patch_start(work)  # not twice
    import os

    env = {k: v for k, v in os.environ.items() if k in ("HOME", "PATH", "LANG")}
    env["PYTHONPATH"] = ":".join(str(p) for p in (work / "shopping-agent/core", work / "shopping-agent/runtime-messages-api", CFG.world_tree / "commerce-common"))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=work, env=env, text=True, capture_output=True)
    assert " passed" in proc.stdout and proc.returncode == 0, proc.stdout[-600:]


def test_the_contract_file_states_the_gap_per_store_and_the_check_gets_the_types():
    from experiment.contract import render_contract, render_types

    silent = render_contract(CFG.world_tree, "esci", explicit=True, features=["F3"])
    assert "there is no method that takes a destination or a ZIP, none that returns a price history" in silent
    refusing = render_contract(CFG.world_tree, "esci-refuse", explicit=True, features=["F3"])
    assert "gives no quote for a ZIP other than the customer's saved location: it refuses such a request" in refusing
    assert "takes a destination or a ZIP" not in refusing and "A request that gives a ``zip_code`` is refused" in refusing
    capable = render_contract(CFG.world_tree, "esci-zip", explicit=False, features=["F3"])
    assert "for the five-digit ``zip_code`` when one is given" in capable and "refused" not in capable
    plain = render_contract(CFG.world_tree, "esci", explicit=False, features=["F3"])
    assert "does not offer" not in plain and "class " not in plain  # the evaluator's file has no type definitions
    types = render_types(CFG.world_tree, "esci")
    assert 'sort: Literal["relevance", "price_asc", "price_desc", "rating"]' in types
    assert "class FulfillmentOption(BaseModel):" in types and "class SearchFilters(BaseModel):" in types


SALES_PROBE = """
import asyncio, json, sys
from pathlib import Path
from harness.subject import activate
activate(Path(sys.argv[1]), Path(sys.argv[2]))
from harness.runner import World
from harness.worlds import units_sold
from shopping_agent.types import SearchFilters, ShoppingSessionContext
out = {}
for kind in ("esci", "esci-sales", "esci-nosales"):
    world = World(Path(sys.argv[2]), kind=kind)
    session = ShoppingSessionContext(session_id="s", user_id="demo-user")
    async def ask(query, sort, limit):
        try:
            found = await world.backend.search_products(session, query, SearchFilters(sort=sort) if sort else None, limit)
            return [[p.product_id, p.title, units_sold(p.product_id)] for p in found]
        except Exception as error:
            return f"{type(error).__name__}: {error}"
    out[kind] = {q: {"relevance": asyncio.run(ask(q, None, 8)), "rating": asyncio.run(ask(q, "rating", 8)), "best": asyncio.run(ask(q, "best_selling", 5)), "all": asyncio.run(ask(q, None, 200))} for q in ("flashlight", "wireless chargers", "best selling flashlights", "zzzz")}
print(json.dumps(out))
"""


def _patched_start(tmp_path, kind):
    work = tmp_path / "start"
    shutil.copytree(CFG.subject_start, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    patch_start(work, kind)
    return work


def test_the_three_stores_differ_only_in_what_a_sales_sort_gets(tmp_path):
    import json
    import os

    work = _patched_start(tmp_path, "esci-sales")
    env = {k: v for k, v in os.environ.items() if k in ("HOME", "PATH", "LANG")}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run([sys.executable, "-c", SALES_PROBE, str(work), str(CFG.world_tree)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr[-800:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    for query in ("flashlight", "wireless chargers"):
        silent, sales, refusing = (out[kind][query] for kind in ("esci", "esci-sales", "esci-nosales"))
        for other in (sales, refusing):  # every other search: the same answer everywhere
            assert all(other[key] == silent[key] for key in ("relevance", "rating", "all"))
        assert silent["best"] == silent["relevance"][:5]  # upstream's search orders an unknown sort by relevance, silently
        assert refusing["best"] == "NotOffered: A ranking by sales"
        best = sales["best"]
        assert len(best) == 5 and [row[2] for row in best] == sorted((row[2] for row in best), reverse=True)
        assert {row[0] for row in best} <= {row[0] for row in silent["all"]}  # matches of the query, no others
        words = ["flashlight"] if query == "flashlight" else ["wireless", "charger"]
        assert all(all(word in row[1].lower() for word in words) for row in best)  # of the kind asked for
        assert [row[0] for row in best] != [row[0] for row in silent["relevance"][:5]] != [row[0] for row in silent["rating"][:5]]
    # Words for the ranking name no kind; a query that names none has no ranking.
    assert out["esci-sales"]["best selling flashlights"]["best"] == out["esci-sales"]["flashlight"]["best"]
    assert out["esci-sales"]["zzzz"]["best"] == []


def test_units_sold_are_fixed_and_the_kind_is_read_from_the_title():
    from harness.worlds import kind_terms, names_the_kind, units_sold

    assert units_sold("B07XJN27NZ") == units_sold("B07XJN27NZ") and units_sold("B07XJN27NZ", seed=1) != units_sold("B07XJN27NZ", seed=2)
    assert all(1 <= units_sold(f"id-{n}") <= 5001 for n in range(200))
    assert names_the_kind("Anker Wireless Chargers Bundle, PowerWave Pad", ["wireless", "charger"])
    assert names_the_kind("LED Flashlight, 2 Pack", ["flashlights"]) and not names_the_kind("Tower Fan with Remote", ["flashlight"])
    assert kind_terms(["best", "selling", "flashlights"]) == ["flashlights"] and not names_the_kind("Best Seller!", ["best", "sellers"])


def test_the_start_of_a_sales_store_shows_the_sort_in_its_type_and_not_in_its_tool(tmp_path):
    import os

    work = _patched_start(tmp_path, "esci-nosales")
    assert '"rating", "best_selling"]' in (work / "shopping-agent/core/shopping_agent/types.py").read_text(encoding="utf-8")
    assert "best_selling" not in (work / "shopping-agent/core/shopping_agent/tools/registry.py").read_text(encoding="utf-8")
    assert "zip_code" not in (work / "shopping-agent/core/shopping_agent/backend.py").read_text(encoding="utf-8")  # one difference a backend
    env = {k: v for k, v in os.environ.items() if k in ("HOME", "PATH", "LANG")}
    env["PYTHONPATH"] = ":".join(str(p) for p in (work / "shopping-agent/core", work / "shopping-agent/runtime-messages-api", CFG.world_tree / "commerce-common"))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=work, env=env, text=True, capture_output=True)
    assert " passed" in proc.stdout and proc.returncode == 0, proc.stdout[-600:]


NEW_PROBE = """
import asyncio, json, sys
from pathlib import Path
from harness.subject import activate
activate(Path(sys.argv[1]), Path(sys.argv[2]))
from harness.runner import World
from shopping_agent.types import SearchFilters, ShoppingSessionContext
wt = Path(sys.argv[2])
session = ShoppingSessionContext(session_id="s", user_id="demo-user")
def run(coro):
    try:
        return asyncio.run(coro)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
out = {}
for kind in ("esci", "esci-zip-ignore", "esci-refuse", "esci-sales-ignore", "esci-onsale-refuse", "esci-onsale",
             "esci-stock-ignore", "esci-stock-refuse", "esci-stock", "esci-adjust-refuse", "esci-adjust"):
    b = World(wt, kind=kind).backend
    row = {}
    def ids(found):
        return found if isinstance(found, str) else [p.product_id for p in found]
    row["plain"] = ids(run(b.search_products(session, "flashlight", None, 5)))
    sale = run(b.search_products(session, "flashlight", SearchFilters(attributes={"offer": "on_sale"}), 5))
    row["sale"] = ids(sale)
    row["sale_marks"] = sale if isinstance(sale, str) else [[p.labels, p.attributes.get("regular_price"), p.price] for p in sale]
    if kind == "esci-sales-ignore":
        row["best"] = ids(run(b.search_products(session, "flashlight", SearchFilters(sort="best_selling"), 5)))
    try:
        d = run(b.get_product_details(session, "B07XJN27NZ"))
        row["read"] = d if isinstance(d, str) else [d.in_stock, d.attributes.get("store")]
        for store in ("Springfield", "the Riverton store", "Shelbyville"):
            d = run(b.get_product_details(session, "B07XJN27NZ", store))
            row["read " + store] = d if isinstance(d, str) else [d.in_stock, d.attributes.get("store"), d.attributes.get("units_at_store")]
    except TypeError as error:
        row["read Springfield"] = f"TypeError: {error}"
    found = run(b.search_policies(session, "price adjustment after purchase"))
    row["policies"] = found if isinstance(found, str) else [[p.policy_id, p.content[:60]] for p in found]
    if kind == "esci-zip-ignore":
        row["quote"] = [o.model_dump() for o in asyncio.run(b.get_fulfillment_options(session, ["B07XJN27NZ"]))]
        row["quote 94110"] = [o.model_dump() for o in asyncio.run(b.get_fulfillment_options(session, ["B07XJN27NZ"], zip_code="94110"))]
    out[kind] = row
print(json.dumps(out))
"""


def test_the_stores_of_the_new_specs_ignore_refuse_or_are_capable(tmp_path):
    import json
    import os

    from harness.worlds import ADJUST_POLICY_ID, sale_of, units_at

    work = _patched_start(tmp_path, "esci-stock")
    patch_start(work, "esci-zip-ignore")  # the patches of different specs touch different places
    patch_start(work, "esci-sales-ignore")
    env = {k: v for k, v in os.environ.items() if k in ("HOME", "PATH", "LANG")}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run([sys.executable, "-c", NEW_PROBE, str(work), str(CFG.world_tree)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr[-800:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    plain = out["esci"]["plain"]
    # F9. Upstream's backend drops the sale filter without a word; one variant refuses it, one answers it.
    assert out["esci"]["sale"] == plain and all(marks[0] == [] and marks[1] is None for marks in out["esci"]["sale_marks"])
    assert out["esci-onsale-refuse"]["sale"] == "NotOffered: A filter for products on sale" and out["esci-onsale-refuse"]["plain"] == plain
    sale = out["esci-onsale"]
    assert sale["plain"] == plain and sale["sale"] != plain and len(sale["sale"]) == 5
    assert all(sale_of(pid) for pid in sale["sale"]) and all("On sale" in marks[0] and float(marks[1]) > marks[2] for marks in sale["sale_marks"])
    # F8 and F3 with the interface of their two variants: the request is taken and ignored.
    assert out["esci-sales-ignore"]["best"] == plain
    quiet = out["esci-zip-ignore"]
    assert quiet["quote 94110"] == quiet["quote"] and len(quiet["quote"]) == 3
    # F10. One flag for the chain, a refusal, or the stock of the backend that is named.
    for kind in ("esci-stock-ignore", "esci-stock-refuse", "esci-stock"):
        assert out[kind]["read"] == [True, None] and out[kind]["plain"] == plain
    assert out["esci-stock-ignore"]["read Springfield"] == [True, None, None] == out["esci-stock-ignore"]["read Shelbyville"]
    assert out["esci-stock-refuse"]["read Springfield"] == "NotOffered: Stock at a named store"
    capable = out["esci-stock"]
    assert capable["read Springfield"] == [False, "ACME Springfield", "0"] and units_at("B07XJN27NZ", "Springfield") == 0
    assert capable["read the Riverton store"] == [True, "ACME Riverton", str(units_at("B07XJN27NZ", "Riverton"))]
    assert capable["read Shelbyville"] == "NotOffered: Stock at a store named Shelbyville"
    assert units_at("B0753Z4PGC", "Riverton") > 0 and units_at("B07XJN27NZ", "Maple Heights") > 0  # the second probe, and the saved store
    assert out["esci"]["read Springfield"].startswith("TypeError")  # upstream's product read has no such parameter
    # F11. The nearest passage, a passage that says no, or one that grants it.
    assert [p[0] for p in out["esci"]["policies"]][0] == "price-match" and ADJUST_POLICY_ID not in [p[0] for p in out["esci"]["policies"]]
    assert out["esci-adjust-refuse"]["policies"][0][0] == ADJUST_POLICY_ID and out["esci-adjust-refuse"]["policies"][0][1].startswith("ACME does not adjust")
    assert out["esci-adjust"]["policies"][0][0] == ADJUST_POLICY_ID and "within 21 days" in out["esci-adjust"]["policies"][0][1]
    assert out["esci-adjust"]["plain"] == plain


def test_the_start_of_a_stock_store_shows_the_parameter_and_passes_its_tests(tmp_path):
    import os

    work = _patched_start(tmp_path, "esci-stock-ignore")
    assert "product_id: str, store: str | None = None" in (work / "shopping-agent/core/shopping_agent/backend.py").read_text(encoding="utf-8")
    assert '"store"' not in (work / "shopping-agent/core/shopping_agent/tools/registry.py").read_text(encoding="utf-8")  # the tool is the developer's to change
    env = {k: v for k, v in os.environ.items() if k in ("HOME", "PATH", "LANG")}
    env["PYTHONPATH"] = ":".join(str(p) for p in (work / "shopping-agent/core", work / "shopping-agent/runtime-messages-api", CFG.world_tree / "commerce-common"))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=work, env=env, text=True, capture_output=True)
    assert " passed" in proc.stdout and proc.returncode == 0, proc.stdout[-600:]


def test_every_variant_is_a_known_world_and_names_its_spec():
    from harness.worlds import BEHAVIOUR, PATCHES, WORLD_KINDS, store_named

    assert set(BEHAVIOUR) | {"esci", "retail", "wands"} == set(WORLD_KINDS)
    for feature in ("F3", "F8", "F10"):  # a patched repository, shared by the spec's three variants
        kinds = [k for k, (f, _) in BEHAVIOUR.items() if f == feature]
        assert sorted(b for _, b in (BEHAVIOUR[k] for k in kinds)) == ["capable", "ignore", "refuse"]
        assert len({PATCHES[k] for k in kinds}) == 1
    for feature in ("F9", "F11"):  # upstream's backend is the ignoring one: no file of the repository differs
        kinds = [k for k, (f, _) in BEHAVIOUR.items() if f == feature]
        assert sorted(b for _, b in (BEHAVIOUR[k] for k in kinds)) == ["capable", "refuse"] and not any(k in PATCHES for k in kinds)
    assert store_named("the Springfield store") == "Springfield" == store_named("ACME Springfield") and store_named("Shelbyville") is None
