# SPDX-License-Identifier: Apache-2.0

"""The WANDS example's fixtures: a catalog built from the synthetic, WANDS-shaped sample in
``fixtures/wands_sample/`` (the real build needs the downloaded dataset), the backend over
it, and the app with ``WANDS_HOME_DATA_DIR`` pointing at it."""

import importlib
import os
import shutil
from datetime import date
from pathlib import Path

import pytest

from demo_common.tests.fixtures import *  # noqa: F403
from wands_home.pipeline.catalog import build_catalog
from wands_home.pipeline.extract import read_jsonl, select_from_rows, write_raw
from wands_home.pipeline.select import SelectionParams

TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = TESTS_DIR / "fixtures" / "wands_sample"
EXAMPLE_DATA_DIR = TESTS_DIR.parents[1] / "data"
SAMPLE_PARAMS = SelectionParams(
    n_queries=10, seed=7, min_exact=1, min_irrelevant=1, min_judged=3, max_judged=40
)
SAMPLE_SEED = 7
SAMPLE_ANCHOR = date(2026, 11, 16)
STATIC_FILES = ("users.json", "policies.json", "memory-seed.json")


def sample_rows() -> tuple[list[dict], dict[str, dict]]:
    judgments = read_jsonl(SAMPLE_DIR / "examples.jsonl")
    products = {row["product_id"]: row for row in read_jsonl(SAMPLE_DIR / "products.jsonl")}
    return judgments, products


def build_sample(data_dir: Path, seed: int = SAMPLE_SEED) -> dict:
    """Extract (from the fixture rows) and build into ``data_dir``; returns the manifest."""
    judgments, products = sample_rows()
    selection, chosen = select_from_rows(judgments, products, SAMPLE_PARAMS)
    write_raw(data_dir / "raw", selection, chosen, {"fixture": "wands_sample"})
    manifest = build_catalog(data_dir / "raw", data_dir, seed=seed, anchor=SAMPLE_ANCHOR)
    for name in STATIC_FILES:
        shutil.copy(EXAMPLE_DATA_DIR / name, data_dir / name)
    return manifest


@pytest.fixture(scope="session")
def built_data_dir(tmp_path_factory) -> Path:
    data_dir = tmp_path_factory.mktemp("wands-data")
    build_sample(data_dir)
    return data_dir


@pytest.fixture(scope="session")
def make_storefront(built_data_dir):
    from wands_home.api.wands_backend import WandsStorefront

    return lambda: WandsStorefront(built_data_dir)


@pytest.fixture(scope="session")
def main(built_data_dir):
    os.environ["WANDS_HOME_DATA_DIR"] = str(built_data_dir)
    return importlib.import_module("wands_home.api.main")
