# SPDX-License-Identifier: Apache-2.0

"""The `audit` arm's automatic check. When a spec arrives, one call to the role
model, with no tools, is given the spec and the backend's service contract with the definitions
of its types, and lists what the spec presumes that the contract does not show. The answer
goes into the evaluator's tree as ``storefront/AUDIT.md`` and into the run's events. The
prompt is fixed and names no feature and no backend; nobody who knows the backend writes any of
the file, and no role but the evaluator reads it."""

from __future__ import annotations

import hashlib
from pathlib import Path

SYSTEM = (
    "You compare the spec of one feature of a store's shopping assistant with the service contract of the store the "
    "assistant runs on. You answer in Markdown and say nothing about how the feature should be built."
)
PROMPT = """Below are the spec of one feature and the store's service contract: the methods the assistant's tools call, with what each takes and returns as the store has deployed it, and the definitions of the types those methods use. The assistant can ask the store for nothing that the contract does not show.

List what the spec presumes of the store's service that the contract does not show: an input a method does not take, a value a type does not allow, a field no result carries, an operation no method performs. For each, one bullet: the clause or section of the spec that presumes it, what it presumes, and what the contract shows instead. List only what the contract does not show. When the spec presumes nothing beyond the contract, answer with the one line "Nothing found." At most eight bullets, no introduction and no advice.

## The spec

{spec}

## The service contract

{contract}
"""
HEADER = (
    "# What the specs presume that the service contract does not show\n\n"
    "This file is written automatically. When a spec arrives, one model call is given the spec and the service "
    "contract (`CONTRACT.md`, with the definitions of its types) and is asked what the spec presumes of the store's "
    "service that the contract does not show. Nobody has checked its answers.\n"
)
MAX_TOKENS = 1500


def messages(spec: str, contract: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(spec=spec.strip(), contract=contract.strip())},
    ]


def prompt_sha() -> str:
    return hashlib.sha256((SYSTEM + "\x1f" + PROMPT).encode("utf-8")).hexdigest()[:12]


def write(path: Path, feature: str, round_index: int, answer: str) -> str:
    """Add one spec's section to the file (made with its header on the first arrival) and
    return the file's text."""
    text = path.read_text(encoding="utf-8") if path.exists() else HEADER
    text += f"\n## {feature} (arrived in round {round_index})\n\n{answer.strip() or '(the check returned nothing)'}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
