# SPDX-License-Identifier: Apache-2.0

"""The harness tests run against the real subject and world trees beside this repository
(``TESTBED_SUBJECT``, ``TESTBED_WORLD`` override the paths) and a small ESCI world built from
the example's synthetic sample, so no dataset and no credentials are needed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.subject import activate

ROOT = Path(__file__).resolve().parents[2]
SUBJECT = Path(os.environ.get("TESTBED_SUBJECT", ROOT.parent / "commerce-agents-subject")).resolve()
WORLD = Path(os.environ.get("TESTBED_WORLD", ROOT.parent / "commerce-agents")).resolve()

if (SUBJECT / "shopping-agent/core").exists() and (WORLD / "examples/esci_retail").exists():
    activate(SUBJECT, WORLD)
    TREES_PRESENT = True
else:
    TREES_PRESENT = False

needs_trees = pytest.mark.skipif(not TREES_PRESENT, reason="subject or world tree not found")


@pytest.fixture(scope="session")
def subject_tree() -> Path:
    return SUBJECT


@pytest.fixture(scope="session")
def world_tree() -> Path:
    return WORLD


@pytest.fixture(scope="session")
def mini_world_dir(tmp_path_factory) -> Path:
    """A small ESCI world: the example's synthetic sample built into a temp directory."""
    from esci_retail.api.tests.conftest import build_sample

    data_dir = tmp_path_factory.mktemp("mini-world")
    build_sample(data_dir)
    return data_dir
