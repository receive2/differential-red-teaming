# SPDX-License-Identifier: Apache-2.0

"""Put one subject tree and the world tree first on ``sys.path``. The venv's editable
installs point at the world checkout; a subject under test is a different tree, so its
``shopping_agent`` and ``shopping_agent_runtime`` must shadow those. ``commerce_common`` and
the world (``examples/``) always come from the world tree: they are frozen. So do the
merchant-side packages the examples' backends are built on: a world tree other than the
checkout the interpreter was installed from (a frozen copy of it) then brings its own."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .worlds import EXAMPLES

SUBJECT_PACKAGES = ("shopping-agent/core", "shopping-agent/runtime-messages-api")
WORLD_PACKAGES = ("commerce-common", "examples", "merchant-agent/core", "merchant-agent/runtime-messages-api")
_SHADOWED = (
    "shopping_agent",
    "shopping_agent_runtime",
    "commerce_common",
    *EXAMPLES,
    "demo_common",
    "merchant_agent",
    "merchant_agent_runtime",
)


def activate(subject: Path, world: Path) -> None:
    """Prepend the trees to ``sys.path`` and drop any already-imported copies, so the next
    import of a shadowed package comes from the trees given."""
    entries = [str(subject / p) for p in SUBJECT_PACKAGES] + [
        str(world / p) for p in WORLD_PACKAGES
    ]
    for entry in reversed(entries):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    for name in list(sys.modules):
        if name.split(".")[0] in _SHADOWED:
            del sys.modules[name]
    importlib.invalidate_caches()
    verify(subject, world)


def verify(subject: Path, world: Path) -> None:
    """Fail loudly when a shadowed package resolves outside the tree it must come from."""
    import commerce_common
    import shopping_agent
    import shopping_agent_runtime

    checks = (
        (shopping_agent, subject.resolve()),
        (shopping_agent_runtime, subject.resolve()),
        (commerce_common, world.resolve()),
    )
    for module, root in checks:
        location = Path(module.__file__).resolve()
        if root not in location.parents:
            raise RuntimeError(f"{module.__name__} resolved to {location}, expected under {root}")
