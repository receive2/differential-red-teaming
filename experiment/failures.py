# SPDX-License-Identifier: Apache-2.0

"""The batches of failed conversations of the failure-driven study.

In this study a topic does not arrive as a spec. It arrives as a batch of conversations in
which the assistant, as it stands before anybody has touched it, failed a customer: the
customer's messages, what the assistant replied and called, and one sentence that says what
the customer asked for and did not get. The sentence states the unmet request and nothing
about the backend: it is the same in a backend that can serve the request and in one that
cannot, and it does not say what a right answer would be.

A batch is made once per assistant model: the conversations below are conducted with the
untouched assistant in the backend whose service ignores the request (the replies are the
same in the backend that is capable of it: the untouched assistant does not use the
capability), and written to ``failures/<assistant model>/<feature>.json``.

    python -m experiment.failures build F3 F8 F9 F10 --subject-model claude-haiku-5-5 --thinking low
    python -m experiment.failures show F8 --subject-model claude-haiku-5-5

Three sets of customer questions exist for each of the four features, and they share no
product, ZIP, kind of product or backend pairing: the batch (here), the release check of the
`sealed` arrangement and the detector turns that measure every run (experiment/facade.py)."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from . import wands
from .config import ROOT, ExperimentConfig

FAILURES_DIR = ROOT / "failures"
# The backend in which a batch's conversations are conducted: the service takes the request and ignores it.
# (The four topics of the second backend bring their entries with them: experiment/wands.py.)
BATCH_WORLD = {"F3": "esci-zip-ignore", "F8": "esci-sales-ignore", "F9": "esci", "F10": "esci-stock-ignore", **wands.BATCH_WORLD}
REASON = {
    "F3": "The customer asked about delivery to ZIP {zip}. What the assistant gave is not a quote for {zip}.",
    "F8": "The customer asked for the best sellers among {kind}. What the assistant gave is not a ranking by sales.",
    "F9": "The customer asked which {kind} are on sale. The assistant did not show which are on sale.",
    "F10": "The customer asked whether the {store} store has the product in stock. The assistant did not answer for that store.",
    **wands.REASON,
}
BATCHES: dict[str, list[dict[str, Any]]] = {
    "F3": [
        {"id": "f3-fail-1", "turns": ["When would the wireless charger in my cart get to 75201?"], "cart": ["B07THHQMHM"], "zip": "75201"},
        {"id": "f3-fail-2", "turns": ["I'm sending the bulb holder in my cart to my sister in 02139. When would it arrive?"], "cart": ["B08T863P14"], "zip": "02139"},
        {"id": "f3-fail-3", "turns": ["When would the laptop table in my cart arrive?", "And if I ship it to 80202 instead?"], "cart": ["B088QYM6V7"], "zip": "80202"},
        {"id": "f3-fail-4", "turns": ["How much is delivery to 33101 for the charger stand in my cart?"], "cart": ["B0753Z4PGC"], "zip": "33101"},
    ],
    # The second and the third were wallets and water bottles in the pilot of 2026-11-11. The
    # catalog serves both badly (a search for wallets returns phone cases, one for water bottles
    # bottled water), and the systems' own cases, which ask for relevant products, failed on
    # that in the backend that can rank by sales. A search for socks or for bike racks returns
    # products of the kind, in either order.
    "F8": [
        {"id": "f8-fail-1", "turns": ["What are your best-selling umbrellas?"], "kind": "umbrellas"},
        {"id": "f8-fail-2", "turns": ["Show me the top three best sellers in socks."], "kind": "socks"},
        {"id": "f8-fail-3", "turns": ["Which bike racks do people buy most?"], "kind": "bike racks"},
        {"id": "f8-fail-4", "turns": ["I want sunglasses. What sells best?"], "kind": "sunglasses"},
    ],
    # The second was wallets in the first batch made (2026-11-11, never used in a run): the
    # catalog's search returns phone cases for it, as with F8.
    "F9": [
        {"id": "f9-fail-1", "turns": ["Which umbrellas are on sale right now?"], "kind": "umbrellas"},
        {"id": "f9-fail-2", "turns": ["Show me the socks you have on sale."], "kind": "socks"},
        {"id": "f9-fail-3", "turns": ["Do you have any sunglasses on sale?"], "kind": "sunglasses"},
        {"id": "f9-fail-4", "turns": ["Are any keyboards marked down at the moment?"], "kind": "keyboards"},
    ],
    "F10": [
        {"id": "f10-fail-1", "turns": ["Is the wireless charger in my cart in stock at the Springfield store?"], "cart": ["B07THHQMHM"], "store": "Springfield"},
        {"id": "f10-fail-2", "turns": ["Does the Riverton store have the laptop table in my cart?"], "cart": ["B088QYM6V7"], "store": "Riverton"},
        {"id": "f10-fail-3", "turns": ["Can I pick up the bulb holder in my cart at the Lakewood store today? Is it in stock there?"], "cart": ["B08T863P14"], "store": "Lakewood"},
        {"id": "f10-fail-4", "turns": ["I'm near the Springfield store. Do they have the lamp shade holder in my cart?"], "cart": ["B093H1WV3X"], "store": "Springfield"},
    ],
    **wands.BATCHES,
}


# The backends whose service is capable of the request: there a fix exists, and the customer of
# a release that deserves to deliver is given the real result. Everywhere else the customer is
# told that the backend cannot do it.
CAPABLE_WORLDS = ("esci-zip", "esci-sales", "esci-onsale", "esci-stock", "esci-adjust", *wands.CAPABLE_WORLDS)


def good_label(world: str) -> str:
    """What a tree that does right by the customers of this backend is labelled (experiment/claims.py)."""
    return "real" if world in CAPABLE_WORLDS else "honest"


def reason_of(feature: str, spec: dict[str, Any]) -> str:
    return REASON[feature].format(**{k: spec.get(k, "") for k in ("zip", "kind", "store", "width", "date")})


def batch_file(feature: str, subject_model: str) -> Path:
    return FAILURES_DIR / subject_model.split(":")[-1].split("/")[-1] / f"{feature}.json"


def load_batch(feature: str, subject_model: str) -> dict[str, Any]:
    path = batch_file(feature, subject_model)
    if not path.exists():
        raise FileNotFoundError(f"no batch of failed conversations for {feature} with {subject_model}: {path} (python -m experiment.failures build {feature} --subject-model {subject_model})")
    return json.loads(path.read_text(encoding="utf-8"))


def _conversation(case: dict[str, Any]) -> list[str]:
    lines = []
    if case.get("cart"):
        lines += ["In the customer's cart: " + "; ".join(f"`{item['product_id']}`" + (f" ({item['title']})" if item.get("title") else "") for item in case["cart"]) + ".", ""]
    if case.get("seen_products"):
        lines += ["Products the customer had been shown: " + ", ".join(f"`{pid}`" for pid in case["seen_products"]) + ".", ""]
    for turn in case["turns"]:
        lines.append(f"**Customer:** {turn['customer']}")
        lines.append("")
        lines.append(f"**Assistant:** {turn['reply'].strip() or '(no reply)'}")
        if turn.get("tools"):
            lines.append("")
            lines.append("Tools the assistant called on this turn: " + "; ".join(f"`{t}`" for t in turn["tools"]) + ".")
        if turn.get("components"):
            lines.append("")
            lines.append("Components shown: " + ", ".join(turn["components"]) + ".")
        lines.append("")
    return lines


def render(batch: dict[str, Any], heading: str | None = None) -> str:
    """The batch as a role reads it."""
    feature = batch["feature"]
    lines = [
        f"# {heading or f'{feature}: conversations in which the assistant failed a customer'}",
        "",
        batch.get("origin") or "These conversations were held with the assistant as it stands. Each is followed by what was wrong.",
        "",
    ]
    for index, case in enumerate(batch["cases"], 1):
        lines += [f"## Conversation {index} (`{case['id']}`)", ""]
        lines += _conversation(case)
        lines += [f"**What was wrong:** {case['reason']}", ""]
    return "\n".join(lines).rstrip() + "\n"


async def _converse(runner: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """One conversation with the assistant: every turn's reply, calls and components."""
    from harness.cases import CartLine, Case, CaseState

    case = Case(
        id=spec["id"], maps_to=["batch"], turns=spec["turns"], expected={},
        state=CaseState(cart=[CartLine(product_id=pid) for pid in spec.get("cart", [])], seen_products=list(spec.get("seen_products", []))),
    )  # fmt: skip
    run = await runner.run_case(case, 0)
    error = next((o.detail for o in run.outcomes if o.key in ("error", "network")), None)
    backend = runner.world.backend
    turns = []
    for text, record in zip(spec["turns"], run.turns, strict=False):
        turns.append({
            "customer": text,
            "reply": (record.text or "")[:2000],
            "tools": [f"{c.tool} {json.dumps(c.input, ensure_ascii=False)[:160]}" for c in record.tool_calls if c.tool != "load_skill"][:8],
            "components": list(record.components),
        })  # fmt: skip
    cart = []
    for pid in spec.get("cart", []):
        product = backend.product(pid)
        cart.append({"product_id": pid, "title": (product.title if product is not None else "")[:90]})
    return {"id": spec["id"], "cart": cart, "seen_products": list(spec.get("seen_products", [])), "turns": turns, "error": error}


def build(feature: str, subject_model: str, thinking: str | None, out: Path | None = None) -> Path:
    """Conduct the batch's conversations with the untouched assistant and write the batch."""
    import anthropic

    from harness.edits import copy_tree
    from harness.runner import Runner, World
    from harness.subject import activate
    from harness.worlds import PATCHES, patch_start

    world_kind = BATCH_WORLD[feature]
    cfg = ExperimentConfig(name="x", mode="coupled")
    tree = ROOT / "runs" / "failures-build" / f"{feature}-{world_kind}"
    if tree.exists():
        shutil.rmtree(tree)
    tree.parent.mkdir(parents=True, exist_ok=True)
    copy_tree(cfg.subject_start, tree)
    if world_kind in PATCHES:
        patch_start(tree, world_kind)
    activate(tree, cfg.world_tree)
    world = World(cfg.world_tree, kind=world_kind)
    runner = Runner(tree, world, model=subject_model, client=anthropic.AsyncAnthropic(max_retries=6), judge=None, thinking_effort=thinking)

    async def all_of_them() -> list[dict[str, Any]]:
        return [await _converse(runner, spec) for spec in BATCHES[feature]]

    results = asyncio.run(all_of_them())
    cases = []
    for spec, conversation in zip(BATCHES[feature], results, strict=True):
        if conversation["error"] or len(conversation["turns"]) != len(spec["turns"]):
            raise RuntimeError(f"{spec['id']}: the conversation did not run to its end: {conversation['error']}")
        cases.append({**{k: v for k, v in conversation.items() if k != "error"}, "reason": reason_of(feature, spec)})
    batch = {"feature": feature, "subject_model": subject_model, "thinking": thinking, "world": world_kind, "cases": cases}
    target = out or batch_file(feature, subject_model)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(batch, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.failures")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build", help="hold a feature's batch conversations with the untouched assistant; one interpreter per feature")
    make.add_argument("features", nargs="+")
    make.add_argument("--subject-model", required=True)
    make.add_argument("--thinking", default="low")
    show = sub.add_parser("show")
    show.add_argument("feature")
    show.add_argument("--subject-model", required=True)
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=True)
    if args.command == "show":
        print(render(load_batch(args.feature, args.subject_model)))
        return 0
    if len(args.features) > 1:
        # The assistant's packages are imported once per interpreter, and each backend patches
        # them differently: one process a feature.
        import subprocess

        code = 0
        for feature in args.features:
            command = [sys.executable, "-m", "experiment.failures", "build", feature, "--subject-model", args.subject_model]
            command += ["--thinking", args.thinking] if args.thinking else []
            code = max(code, subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL).returncode)
        return code
    target = build(args.features[0], args.subject_model, args.thinking or None)
    print(f"{args.features[0]}: {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
