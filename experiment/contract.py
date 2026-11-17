# SPDX-License-Identifier: Apache-2.0

"""The storefront's service contract, rendered from the backend implementation: the methods the
assistant's tools call, with their signatures and docstrings as the backend implements them.
The `world` evaluator arm reads it as ``cases/storefront/CONTRACT.md``."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

SALES_GAP = " and the catalog search has no sales data: it sorts by relevance, price or rating only"
# What the explicit list says of the destination, by backend.
NO_DESTINATION = "Any operation or parameter not listed above: there is no method that takes a destination or a ZIP, none"
REFUSED_DESTINATION = (
    "The fulfillment service gives no quote for a ZIP other than the customer's saved location: it refuses such a "
    "request. Any operation or parameter not listed above: there is no method"
)
TYPE_PACKAGES = ("shopping_agent", "commerce_common")


def _service(world_tree: Path, kind: str) -> tuple[Any, Any, list[str]]:
    """The backend implementation, the protocol it implements, and the names of the protocol's methods."""
    import sys

    for entry in (world_tree / "examples", world_tree / "commerce-common"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    from harness.runner import World

    backend = World(world_tree, kind=kind).backend
    base = next((b for b in type(backend).__mro__ if b.__name__ == "StorefrontBackend"), None)
    names = sorted(
        n
        for n, member in inspect.getmembers(base or type(backend))
        if not n.startswith("_") and inspect.isfunction(member)
    )
    return backend, base, names


def service_types(world_tree: Path, kind: str = "esci") -> list[type]:
    """The classes the service's methods take and return, and the classes those are made of,
    in the order they are first met (methods by name)."""
    import re
    import sys
    import typing

    backend, base, names = _service(world_tree, kind)
    namespace = vars(sys.modules[(base or type(backend)).__module__])
    found: dict[str, type] = {}

    def walk(tp: Any) -> None:
        for arg in typing.get_args(tp):
            walk(arg)
        if not inspect.isclass(tp) or tp.__module__.split(".")[0] not in TYPE_PACKAGES or tp.__name__ in found:
            return
        found[tp.__name__] = tp
        for parent in tp.__mro__[1:]:
            walk(parent)
        for field in (getattr(tp, "model_fields", None) or {}).values():
            walk(field.annotation)

    for name in names:
        member = getattr(type(backend), name, None) or getattr(base, name)
        try:
            signature = str(inspect.signature(member))
        except (TypeError, ValueError):
            continue
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", signature):
            walk(namespace.get(word))
    return list(found.values())


def render_types(world_tree: Path, kind: str = "esci") -> str:
    """The definitions of those classes as the backend's code has them. Only the automatic
    check of the `audit` arm is given this (experiment/spec_audit.py); no role reads it."""
    lines = ["## The types the methods take and return", "", "```python"]
    for index, cls in enumerate(service_types(world_tree, kind)):
        if index:
            lines.append("")
        lines.append(inspect.getsource(cls).rstrip())
    lines += ["```", ""]
    return "\n".join(lines)


def render_contract(world_tree: Path, kind: str = "esci", explicit: bool = True, features: list[str] | None = None) -> str:
    """``explicit`` (the `world` arm) adds what the backend does not offer and that a spec may
    assume it; without it the file is the methods and nothing else (`contract`, `mandate`,
    `audit`). For a run that has F8 the explicit list also names the missing sales data;
    the file of a run without F8 is what it always was. In the backend whose
    fulfillment service takes a ZIP and refuses it the list says that."""
    backend, base, names = _service(world_tree, kind)
    lines = [
        "# The storefront's service contract",
        "",
        f"The assistant's tools call these methods of the store's service (`{type(backend).__name__}`). "
        "This is what the service takes and what it returns, as deployed"
        + (
            "; the assistant cannot make the service accept an input or perform an operation that is not here. "
            "A spec is written by product and may assume a capability the service does not have."
            if explicit
            else "."
        ),
        "",
    ]
    for name in names:
        member = getattr(type(backend), name, None) or getattr(base, name)
        try:
            signature = str(inspect.signature(member)).replace("'", "")
        except (TypeError, ValueError):
            signature = "(...)"
        doc = inspect.getdoc(member) or inspect.getdoc(getattr(base, name, None)) or ""
        first = doc.split("\n\n")[0].replace("\n", " ").strip()
        lines.append(f"## `{name}{signature}`")
        lines.append("")
        lines.append(first or "(no description)")
        lines.append("")
    if not explicit:
        return "\n".join(lines)
    lines += [
        "## What the service does not offer",
        "",
        (REFUSED_DESTINATION if kind == "esci-refuse" else NO_DESTINATION)
        + " that returns a price history, none that cancels or changes an order"
        + (SALES_GAP if "F8" in (features or []) else "")
        + ".",
        "",
    ]
    return "\n".join(lines)
