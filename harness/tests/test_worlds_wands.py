# SPDX-License-Identifier: Apache-2.0

"""The second backend's variants: for each of its four topics, a service that takes the request
and ignores it and one that is capable of it; and the backend as it is, whose services take no
such input."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import date, timedelta

import pytest

from experiment.config import ROOT, ExperimentConfig
from harness.worlds import (
    ADDED_WITHIN_DAYS,
    BEHAVIOUR,
    PATCHES,
    SHOWROOMS,
    TOPIC_TITLES,
    WORLD_KINDS,
    added_on,
    arrival_terms,
    business_days_after,
    data_dir_of,
    day_named,
    day_words,
    example_of,
    freight_days,
    patch_start,
    safety_search,
    showroom_named,
    standard_days,
    units_at_showroom,
)

CFG = ExperimentConfig(name="worlds-wands-test", mode="coupled")
pytestmark = pytest.mark.skipif(
    not (data_dir_of(CFG.world_tree, "wands") / "catalog.sqlite").exists() or not CFG.subject_start.exists(),
    reason="no WANDS backend on this machine",
)

KINDS = ("wands", "wands-fit-ignore", "wands-fit", "wands-date-ignore", "wands-date", "wands-new-ignore", "wands-new", "wands-stock-ignore", "wands-stock")
TODAY = date(2026, 11, 16)  # the day the catalog's orders are dated from, a Monday
FREIGHT, PARCEL = "B0WF022817", "B0WF005007"  # a dresser of 120 lb, a nightstand of 38 lb

PROBE = """
import asyncio, json, sys
from pathlib import Path
from harness.subject import activate
activate(Path(sys.argv[1]), Path(sys.argv[2]))
from harness.runner import World
from shopping_agent.types import SearchFilters, ShoppingSessionContext
wt = Path(sys.argv[2])
freight, parcel = sys.argv[3], sys.argv[4]
session = ShoppingSessionContext(session_id="s", user_id="demo-user")
def run(make):
    try:
        return asyncio.run(make())
    except Exception as error:
        return f"{type(error).__name__}: {error}"
def rows(found, mark):
    return found if isinstance(found, str) else [[p.product_id, p.attributes.get(mark), p.in_stock] for p in found]
def quotes(found):
    return found if isinstance(found, str) else [[o.method, o.eta, o.fee, o.location] for o in found]
def read(found):
    return found if isinstance(found, str) else [found.in_stock, found.attributes.get("store"), found.attributes.get("units_at_store")]
out = {}
for kind in sys.argv[5:]:
    world = World(wt, kind=kind)
    b = world.backend
    row = {"name": type(b).__name__, "anchor": world.anchor.isoformat()}
    row["pillows"] = rows(run(lambda: b.search_products(session, "throw pillows", None, 8)), "width_inches")
    row["plain"] = rows(run(lambda: b.search_products(session, "nightstands", None, 8)), "width_inches")
    row["all"] = rows(run(lambda: b.search_products(session, "nightstands", None, 300)), "width_inches")
    row["widths"] = {pid: b.width_of(pid) for pid, _, _ in row["all"]}
    for inches in (18, 0):
        row[f"fit {inches}"] = rows(run(lambda: b.search_products(session, "nightstands", SearchFilters(max_width_inches=inches), 8)), "width_inches")
    row["newest"] = rows(run(lambda: b.search_products(session, "newest nightstands", SearchFilters(sort="newest"), 8)), "added_on")
    row["newest, kind alone"] = rows(run(lambda: b.search_products(session, "nightstands", SearchFilters(sort="newest"), 8)), "added_on")
    row["newest, no sort"] = rows(run(lambda: b.search_products(session, "newest nightstands", None, 8)), "added_on")
    row["newest, no kind"] = rows(run(lambda: b.search_products(session, "newest zzzz", SearchFilters(sort="newest"), 8)), "added_on")
    row["filed"] = {pid: [b.product(pid).title, b.product(pid).category] for pid, _, _ in (row["newest"] if isinstance(row["newest"], list) else [])}
    for name, ids in (("freight", [freight]), ("parcel", [parcel])):
        row[name] = quotes(run(lambda: b.get_fulfillment_options(session, ids)))
        for day in ("2026-11-15", "2026-11-16", "2026-11-18", "2026-11-20", "2026-11-23", "2026-12-07", "December 7th", "next Friday"):
            row[f"{name} {day}"] = quotes(run(lambda: b.get_fulfillment_options(session, ids, needed_by=day)))
    row["read"] = read(run(lambda: b.get_product_details(session, parcel)))
    for store in ("Harborview", "the Springfield showroom", "Shelbyville"):
        row["read " + store] = read(run(lambda: b.get_product_details(session, parcel, store)))
    out[kind] = row
print(json.dumps(out))
"""


def _env(*path):
    env = {k: v for k, v in os.environ.items() if k in ("HOME", "PATH", "LANG")}
    env["PYTHONPATH"] = ":".join(str(p) for p in path)
    return env


def _start(tmp_path, *kinds):
    work = tmp_path / "start"
    shutil.copytree(CFG.subject_start, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    for kind in kinds:
        patch_start(work, kind)
    return work


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    # The patches of the four topics touch different places: one repository carries them all.
    work = _start(tmp_path_factory.mktemp("wands"), "wands-fit", "wands-date-ignore", "wands-new", "wands-stock-ignore")
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, str(work), str(CFG.world_tree), FREIGHT, PARCEL, *KINDS],
        cwd=ROOT, env=_env(ROOT), text=True, capture_output=True,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-800:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_every_kind_is_the_same_backend_until_its_own_input_is_given(out):
    base = out["wands"]
    assert base["name"] == "WandsStorefront" and base["anchor"] == "2026-11-16T10:00:00"
    assert len(base["pillows"]) == 8 and all(in_stock for _, _, in_stock in base["pillows"])  # the safety set's search
    assert len(base["plain"]) == 8 and base["freight"] == [["shipping", "1-3 weeks (freight, delivery by appointment)", 49.0, None]]
    assert [quote[0] for quote in base["parcel"]] == ["delivery", "delivery", "pickup"] and base["read"] == [True, None, None]
    for kind in KINDS[1:]:
        row = out[kind]
        assert row["name"] == "WandsStorefront" and row["anchor"] == base["anchor"]  # the variant is not named
        assert all(row[key] == base[key] for key in ("pillows", "plain", "all", "newest, no sort", "freight", "parcel", "read")), kind
    # The backend as it is takes none of the inputs: its services have no such parameter, and
    # its search reads neither the limit nor the sort.
    assert base["freight 2026-12-07"].startswith("TypeError") and base["read Harborview"].startswith("TypeError")
    assert base["fit 18"] == base["plain"] and base["newest"] == base["newest, no sort"]


def test_a_width_limit_is_ignored_or_answered_with_the_matches_that_fit(out):
    base, quiet, capable = out["wands"], out["wands-fit-ignore"], out["wands-fit"]
    assert quiet["fit 18"] == base["plain"] == quiet["fit 0"]  # taken and ignored: no width on any record
    assert all(width is None for _, width, _ in quiet["fit 18"])
    widths = base["widths"]
    fitting = [pid for pid, _, _ in base["all"] if widths[pid] is not None and widths[pid] <= 18]
    assert len(fitting) >= 8 and [pid for pid, _, _ in capable["fit 18"]] == fitting[:8]  # the matches in their order, less the rest
    assert all(mark == f"{widths[pid]:g}" for pid, mark, _ in capable["fit 18"])  # each with its width
    assert any(widths[pid] is None or widths[pid] > 18 for pid, _, _ in base["plain"])  # so the answer is another one
    assert capable["fit 0"] == []


def test_a_date_is_ignored_or_answered_with_the_options_promised_by_it(out):
    base, quiet, capable = out["wands"], out["wands-date-ignore"], out["wands-date"]
    days = ("2026-11-15", "2026-11-16", "2026-11-18", "2026-11-20", "2026-11-23", "2026-12-07", "December 7th", "next Friday")
    for name in ("freight", "parcel"):
        assert all(quiet[f"{name} {day}"] == base[name] for day in days)  # taken and ignored, readable or not
    # A freight order: one option, promised for a day one to three weeks out.
    promised = TODAY + timedelta(days=freight_days(FREIGHT))
    assert TODAY + timedelta(days=7) <= promised <= TODAY + timedelta(days=21)
    dated = [["shipping", f"by {day_words(promised)} (freight, delivery by appointment)", 49.0, None]]
    assert capable["freight 2026-12-07"] == dated == capable["freight December 7th"]
    assert capable["freight 2026-11-20"] == [] == capable["freight 2026-11-15"]  # no option is promised by then
    assert capable["freight next Friday"].startswith("NotOffered: A delivery promise for a date written as next Friday")
    # A parcel order: the options in the order of the usual answer, each with its day.
    standard = business_days_after(TODAY, standard_days(PARCEL))
    express, pickup = ["delivery", "by Wed, Nov 18 (express)", 9.99, None], base["parcel"][2]
    assert pickup[1] == "today by 12 PM" and capable["parcel 2026-11-15"] == [] and capable["parcel 2026-11-16"] == [pickup]
    assert capable["parcel 2026-11-18"] == ([express, pickup] if standard > date(2026, 11, 18) else capable["parcel 2026-12-07"])
    assert capable["parcel 2026-12-07"] == [["delivery", f"by {day_words(standard)} (standard)", 0.0, None], express, pickup]
    for name in ("freight", "parcel"):  # what a request with a date gets is never the usual answer
        assert all(capable[f"{name} {day}"] != base[name] for day in days)


def test_the_newest_sort_is_ignored_or_answered_by_the_day_added(out):
    base, quiet, capable = out["wands"], out["wands-new-ignore"], out["wands-new"]
    assert quiet["newest"] == base["newest, no sort"] and all(mark is None for _, mark, _ in quiet["newest"])
    newest = capable["newest"]
    assert len(newest) == 8 and [pid for pid, _, _ in newest] != [pid for pid, _, _ in base["newest, no sort"]]
    assert all(mark == added_on(pid, TODAY).isoformat() for pid, mark, _ in newest)  # each with its day
    assert [mark for _, mark, _ in newest] == sorted((mark for _, mark, _ in newest), reverse=True)
    assert all("nightstand" in f"{title} {category}".lower() for title, category in capable["filed"].values())  # of the kind
    # Words for the ranking name no kind; a query that names none has no ranking.
    assert capable["newest, kind alone"] == newest and capable["newest, no kind"] == []


def test_a_showroom_is_ignored_or_answered_with_its_stock(out):
    base, quiet, capable = out["wands"], out["wands-stock-ignore"], out["wands-stock"]
    assert all(quiet["read " + store] == base["read"] for store in ("Harborview", "the Springfield showroom", "Shelbyville"))
    for words, showroom in (("Harborview", "Harborview"), ("the Springfield showroom", "Springfield")):
        units = units_at_showroom(PARCEL, showroom)
        assert capable["read " + words] == [units > 0, f"ACME Home {showroom}", str(units)]
    assert capable["read Shelbyville"] == "NotOffered: Stock at a showroom named Shelbyville"


@pytest.mark.parametrize(
    ("kind", "file", "shown", "tool_word"),
    [
        ("wands-fit-ignore", "types.py", "    max_width_inches: float | None = None\n", "max_width_inches"),
        ("wands-date", "backend.py", "        needed_by: str | None = None,\n", "needed_by"),
        ("wands-new-ignore", "types.py", '"rating", "newest"]', "newest"),
        ("wands-stock", "backend.py", "product_id: str, store: str | None = None", '"store"'),
    ],
)
def test_the_start_of_a_variant_shows_the_input_in_its_contract_and_not_in_its_tool(tmp_path, kind, file, shown, tool_word):
    work = _start(tmp_path, kind)
    package = work / "shopping-agent/core/shopping_agent"
    assert shown in (package / file).read_text(encoding="utf-8")
    assert tool_word not in (package / "tools/registry.py").read_text(encoding="utf-8")  # the tool is the developer's to change
    if kind == "wands-stock":
        assert "names an ACME Home showroom" in (package / "backend.py").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        patch_start(work, kind)  # not twice
    path = (work / "shopping-agent/core", work / "shopping-agent/runtime-messages-api", CFG.world_tree / "commerce-common")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=work, env=_env(*path), text=True, capture_output=True)
    assert " passed" in proc.stdout and proc.returncode == 0, proc.stdout[-600:]


def test_the_synthesized_facts_are_fixed_and_within_their_ranges():
    ids = [f"B0WF{n:06d}" for n in range(400)]
    assert {standard_days(pid) for pid in ids} == {3, 4, 5} and {freight_days(pid) for pid in ids} == set(range(7, 22))
    assert freight_days(FREIGHT) == freight_days(FREIGHT) and len({freight_days(FREIGHT, seed=seed) for seed in range(40)}) > 1
    ages = [(TODAY - added_on(pid, TODAY)).days for pid in ids]
    assert min(ages) >= 1 and max(ages) <= ADDED_WITHIN_DAYS and len(set(ages)) > 200
    stock = [units_at_showroom(pid, showroom) for pid in ids[:100] for showroom in SHOWROOMS]
    assert set(stock) == set(range(10)) and 0.35 < stock.count(0) / len(stock) < 0.55
    assert business_days_after(TODAY, 2) == date(2026, 11, 18) and business_days_after(date(2026, 11, 19), 3) == date(2026, 11, 24)
    assert business_days_after(date(2026, 11, 20), 5) == date(2026, 11, 27) and day_words(date(2026, 12, 7)) == "Mon, Dec 7"


def test_dates_and_showrooms_are_read_as_people_write_them():
    monday = date(2026, 11, 30)
    for words in ("2026-11-30", "2026-11-30T09:00:00", "11/30/2026", "11/30/26", "November 30, 2026", "Nov 30 2026", "30 November 2026",
                  "November 30th", "Mon, Nov 30", "Monday, November 30th", "11/30"):  # fmt: skip
        assert day_named(words, TODAY) == monday, words
    assert day_named("January 5", TODAY) == date(2027, 1, 5) and day_named("November 16", TODAY) == TODAY  # the first such day from today on
    assert all(day_named(words, TODAY) is None for words in ("next Friday", "soon", "", "13/45/2026", "the 30th"))
    assert showroom_named("the Springfield showroom") == "Springfield" == showroom_named("ACME Home Springfield")
    assert showroom_named("Cedar Park store") == "Cedar Park" and showroom_named("Shelbyville") is None and showroom_named("") is None
    assert arrival_terms(["newest", "nightstands"]) == ["nightstands"] == arrival_terms(["new", "arrivals", "nightstands"])


def test_every_variant_of_the_second_backend_is_a_known_kind_and_names_its_topic():
    assert set(KINDS) == {kind for kind in WORLD_KINDS if example_of(kind) == "wands_home"}
    assert all(example_of(kind) == "esci_retail" for kind in WORLD_KINDS if kind.startswith("esci")) and example_of("retail") == "retail"
    for feature in ("F21", "F22", "F23", "F24"):  # a patched repository, shared by the topic's two variants
        kinds = [k for k, (f, _) in BEHAVIOUR.items() if f == feature]
        assert sorted(BEHAVIOUR[k][1] for k in kinds) == ["capable", "ignore"] and len({PATCHES[k] for k in kinds}) == 1
        assert feature in TOPIC_TITLES
    assert "wands" not in PATCHES and "wands" not in BEHAVIOUR  # the backend as it is: nothing of the repository differs
    assert safety_search("wands-fit") == "throw pillows" and safety_search("esci-zip") == "wireless chargers" == safety_search("esci")
    assert data_dir_of(CFG.world_tree, "wands-new") == CFG.world_tree / "examples/wands_home/data"
