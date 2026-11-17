# SPDX-License-Identifier: Apache-2.0

"""Run evaluation cases against a tree over the ESCI world, in a fresh interpreter. The
roles call it through ``bin/run_eval`` inside their sessions (with a per-session budget); the
loop calls it for the round's evaluation, the hidden gold and the C* adjudication.

    run_eval --ids f1-001 f1-002 | --feature F1 | --all | --gold hidden/F1/gold.json
             [--cases-root DIR] [--tree DIR] [--json OUT] [--quiet]
             [--trials N] [--pass-rule all|majority]
    run_eval --list-queries bike rack        # the world's judged queries (for presented_relevance)
    run_eval --ask-backend METHOD JSON         # one request to the backend's service (the `probe` arm's evaluator)
    run_eval --list-products wireless charger   # products of the catalog, by words of the title
    run_eval --validate --feature F1         # which case files parse as cases; the assistant is not run
    run_eval --ask MESSAGE [MESSAGE ...] [--cart ID ...] [--seen ID ...]   # one conversation with the assistant (the red team)
    run_eval --compare NAME                  # one comparison of the comparing red team (experiment/compare.py)

In the failure-driven study's `isolated` arrangement the evaluator and the developer are
kept apart. The developer's session is told how many of the evaluator's cases pass and which
fail by number, and nothing of what they are. The evaluator's session is never shown the
changed assistant: in the pilot (revision 1) no case was run for it at all; from revision 2
its cases are run on the assistant as it was before any change (``start-tree`` of the run),
and they are held to the cut-apart vocabulary (``v2-apart``, harness/cases.py).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from harness.cases import Case, CaseFile, load_cases
from harness.mini.models import price_of
from harness.netwait import NETWORK_EXIT
from harness.subject import activate
from harness.worlds import data_dir_of

from .cases_io import invalid_files, load_case_dir
from .config import ROOT

SUBJECT_PRICE_KEYS = {"input_tokens": 0, "output_tokens": 1, "cache_read_input_tokens": 2}
CACHE_WRITE_FACTOR = 1.25  # tokens written to the cache are billed a quarter above the input price
# What the evaluator of a cut-apart system is told when it asks for the assistant to be run.
CUT_APART_NOTE = (
    "run_eval: on this team the evaluator's cases are written from the batch alone: the assistant is not run for "
    "the evaluator. `run_eval --validate --feature F<n>` checks your case files; `run_eval --list-products <words>` "
    "and `run_eval --list-queries <words>` look up the store's catalog."
)
# What it is told from revision 2 on, where its cases are run on the assistant as it was.
CUT_APART_START = (
    "run_eval: these cases are run against the assistant as it was before any change. The developer's change is "
    "not shown to you, and the developer does not see your cases."
)
ASK_REPLY = 1500  # characters of a reply, and
ASK_RESULT = 700  # of one tool's result, shown to the red team


def subject_cost(usage: dict[str, int], model: str) -> float:
    """What the assistant's calls cost: input, output, cache reads, and cache writes."""
    price = price_of(model) or (0.0, 0.0, 0.0)
    cost = sum(usage.get(k, 0) * price[i] for k, i in SUBJECT_PRICE_KEYS.items())
    return cost + usage.get("cache_creation_input_tokens", 0) * price[0] * CACHE_WRITE_FACTOR


def list_products(world_tree: Path, world: str, words: list[str], limit: int = 40) -> int:
    """The catalog's products whose title carries every word: id, title, category, price.
    Read from the catalog's own file; neither the assistant nor the backend's service is asked,
    and nothing a service computes (stock, rating, a variant's additions) is printed."""
    import sqlite3

    if world == "retail":
        print("run_eval: --list-products reads the ESCI catalog; this run's store is another one.")
        return 2
    if not words:
        print("run_eval: --list-products takes one or more words of a product's title, for example `--list-products wireless charger`")
        return 2
    path = data_dir_of(world_tree, world) / "catalog.sqlite"
    with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT product_id, record, category, price FROM products").fetchall()
    found = []
    for product_id, record, category, price in rows:
        title = str(json.loads(record).get("title", ""))
        if all(word.lower() in title.lower() for word in words):
            found.append((title.lower(), product_id, title, category, price))
    found.sort()
    print(f"{len(found)} product{'' if len(found) == 1 else 's'} with {' '.join(words)!r} in the title (id | title | category | price):")
    for _, product_id, title, category, price in found[:limit]:
        print(f"  {product_id} | {title[:90]} | {category} | {price:.2f}")
    if len(found) > limit:
        print(f"  ... and {len(found) - limit} more; add words to narrow the list")
    return 0


def _subject_client(dry_run: bool):
    if dry_run:
        from commerce_common.testing import FakeClient, text_message

        return FakeClient(
            [text_message("I can help with that. Let me know what you need.") for _ in range(2000)]
        )
    import anthropic

    return anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=6)


def _judge(dry_run: bool, model: str | None):
    if dry_run or not model:
        return None
    if ":" in model:  # `provider:model`, as the roles' models are named: a judge of any family
        from harness.judge import make_model_judge

        return make_model_judge(model)
    import anthropic

    from harness.judge import make_judge

    return make_judge(
        anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=6), model
    )


ASK_LIMIT = 24  # requests to the backend's service in one session
ASK_STRING = 240  # a longer string of the answer is cut for printing (the digest is of the whole answer)
ASK_CHARS = 7000


def _plain(value: Any) -> Any:
    """What the service returned as plain data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _shortened(value: Any) -> Any:
    if isinstance(value, str) and len(value) > ASK_STRING:
        return value[:ASK_STRING] + "..."
    if isinstance(value, list):
        return [_shortened(v) for v in value]
    if isinstance(value, dict):
        return {k: _shortened(v) for k, v in value.items()}
    return value


def ask_store(run: Path, cfg: dict[str, Any], role: str, round_index: int, method: str, raw: str) -> int:
    """One request to the backend's service, for the evaluator of the `probe` arm. The method
    is called on the backend as the run has it, for the demo customer, and what it returns is
    printed with a digest of the whole answer (two requests with the same digest got exactly
    the same answer). The backend is built on the assistant as it was before any change
    (``backend-tree`` in the run), never on the developer's workspace. Methods that read only;
    nothing is kept from one request to the next. Every request is logged."""
    import asyncio
    import hashlib
    import inspect
    import re
    import time
    import typing

    log = run / "log" / "store_asks.jsonl"
    used = 0
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            used += row.get("role") == role and row.get("round") == round_index
    record: dict[str, Any] = {"t": time.time(), "role": role, "round": round_index, "method": method, "raw": raw[:2000]}

    def done(text: str, code: int, **more: Any) -> int:
        record.update(more)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(text)
        return code

    if used >= ASK_LIMIT:
        return done(f"ask-store: you have put all {ASK_LIMIT} requests allowed this session.", 2, outcome="over-limit")
    if not re.fullmatch(r"(get|search)_[a-z_]+", method):
        return done("ask-store: the method is one that reads from the service: its name begins with `get_` or `search_`.", 2, outcome="not-a-read")
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as error:
        return done(f"ask-store: the arguments are not JSON ({error}).", 2, outcome="bad-json")
    if not isinstance(arguments, dict):
        return done("ask-store: the arguments are one JSON object, a name for each value.", 2, outcome="bad-json")
    tree = run / "store-tree"
    activate(tree, Path(cfg["world_tree"]))
    from shopping_agent import ShoppingSessionContext

    from harness.runner import World

    world = World(Path(cfg["world_tree"]), kind=cfg.get("world", "esci"))
    member = getattr(world.backend, method, None)
    if member is None or not callable(member):
        names = sorted(n for n in dir(world.backend) if re.fullmatch(r"(get|search)_[a-z_]+", n) and callable(getattr(world.backend, n)))
        return done(f"ask-store: the service has no method `{method}`. It has: {', '.join(names)}.", 2, outcome="no-method")
    signature = inspect.signature(member)
    wanted = [n for n in signature.parameters if n not in ("self", "session")]
    unknown = sorted(set(arguments) - set(wanted))
    if unknown:
        return done(f"ask-store: `{method}` takes {', '.join(wanted) or 'no arguments'}; it does not take {', '.join(unknown)}.", 2, outcome="bad-arguments")
    # The types of the parameters, from whichever class of the backend defines the method with
    # annotations that resolve (a variant's override names its types inside a function).
    hints: dict[str, Any] = {}
    for owner in type(world.backend).__mro__:
        function = owner.__dict__.get(method)
        if function is None:
            continue
        try:
            for name, hint in typing.get_type_hints(function).items():
                hints.setdefault(name, hint)
        except Exception:  # noqa: BLE001, S112 - annotations that module cannot resolve
            continue

    def built(name: str, value: Any) -> Any:
        """A JSON object for a parameter whose type is one of the service's records."""
        if not isinstance(value, dict):
            return value
        hint = hints.get(name)
        for candidate in (hint, *typing.get_args(hint)):
            if inspect.isclass(candidate) and hasattr(candidate, "model_validate"):
                return candidate.model_validate(value)
        return value

    record["arguments"] = arguments
    session = ShoppingSessionContext(session_id="ask-store", user_id="demo-user", now=world.anchor)

    try:
        keywords = {name: built(name, value) for name, value in arguments.items()}
    except Exception as error:  # noqa: BLE001 - a value the parameter's type does not take
        return done(f"ask-store: the arguments do not fit `{method}`: {type(error).__name__}: {error}"[:1200], 2, outcome="bad-arguments")

    async def call() -> Any:
        result = member(session, **keywords)
        return await result if inspect.isawaitable(result) else result

    try:
        answer = _plain(asyncio.run(call()))
    except Exception as error:  # noqa: BLE001 - what the service raises is its answer
        kind = type(error).__name__
        text = f"the service did not answer: {kind}: {error}"[:1500]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return done(f"ask-store {method} {json.dumps(arguments, ensure_ascii=False)}\n{text}\nanswer digest: {digest}", 0, outcome="raised", error=kind, digest=digest)
    whole = json.dumps(answer, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(whole.encode("utf-8")).hexdigest()[:12]
    shown = json.dumps(_shortened(answer), ensure_ascii=False, indent=1, default=str)
    if len(shown) > ASK_CHARS:
        shown = shown[:ASK_CHARS] + f"\n... ({len(shown) - ASK_CHARS} more characters; ask for less, with a smaller limit or fewer ids)"
    line = f"answer digest: {digest} (two requests with the same digest got exactly the same answer)"
    return done(
        f"ask-store {method} {json.dumps(arguments, ensure_ascii=False)}\n{line}\n{shown}\n{line}",
        0, outcome="answered", digest=digest, chars=len(whole),
    )


def ask_assistant(
    run: Path, cfg: dict[str, Any], role: str, round_index: int, messages: list[str], cart: list[str], seen: list[str], tree: Path
) -> int:
    """One conversation with the assistant as the run's workspace has it, for the red team:
    the customer's messages are sent in order to the backend's demo customer's session (with the
    given products in the cart or already shown), and each turn is printed: the reply, the
    tools called with what each returned, the components shown. Nothing is graded. Every
    conversation is logged (``log/redteam_asks.jsonl``) and counted against the session's
    allowance."""
    import asyncio
    import time

    counter = run / "log" / "run_eval_calls" / f"{round_index}-{role}-asks.txt"
    used = int(counter.read_text().strip() or 0) if counter.exists() else 0
    limit = int(cfg.get("redteam_asks") or 12)
    if used >= limit:
        print(f"run_eval: you have held all {limit} conversations allowed this session.")
        return 2
    if not 1 <= len(messages) <= 4:
        print("run_eval: a conversation is one to four customer messages, each in quotes.")
        return 2
    try:
        activate(tree, Path(cfg["world_tree"]))
        from harness.cases import CartLine, Case, CaseState
        from harness.runner import Runner, World

        world = World(Path(cfg["world_tree"]), kind=cfg.get("world", "esci"))
    except Exception as error:  # noqa: BLE001 - whatever the assistant's own code raises on import
        if isinstance(error, RuntimeError) and "resolved to" in str(error):
            raise
        print(f"run_eval: the assistant does not start: {type(error).__name__}: {error}"[:400])
        return 0
    unknown = [pid for pid in [*cart, *seen] if world.backend.product(pid) is None]
    if unknown:
        print(f"run_eval: the catalog has no product {', '.join(unknown)} (`run_eval --list-products <words>` finds ids).")
        return 2
    case = Case(
        id=f"ask-r{round_index}-{used + 1}", maps_to=["ask"], turns=messages, expected={},
        state=CaseState(cart=[CartLine(product_id=pid) for pid in cart], seen_products=list(seen)),
    )  # fmt: skip
    runner = Runner(
        tree, world, model=cfg["subject_model"], client=_subject_client(cfg["dry_run"]), judge=None,
        thinking_effort=cfg.get("subject_thinking"),
    )  # fmt: skip
    result = asyncio.run(runner.run_case(case, 0))
    if any(o.key == "network" for o in result.outcomes):
        print("run_eval: the API was unreachable during the conversation; it does not count. Hold it again once the network is back.")
        return NETWORK_EXIT
    counter.parent.mkdir(parents=True, exist_ok=True)
    counter.write_text(str(used + 1))
    print(f"conversation {used + 1} of {limit}" + (f" (in the cart: {', '.join(cart)})" if cart else "") + (f" (already shown: {', '.join(seen)})" if seen else ""))
    turns = []
    for text, record in zip(messages, result.turns, strict=False):
        calls = [f"{c.tool}({json.dumps(c.input, ensure_ascii=False)[:300]})" for c in record.tool_calls]
        # What the backend's services returned. A loaded skill's text is the assistant's own
        # instructions and a component's result is its echo: neither is the red team's to read.
        results = [
            f"{r.tool} [{r.status}]: {r.text[:ASK_RESULT]}"
            for r in record.tool_results
            if r.tool != "load_skill" and not r.tool.startswith("present_")
        ]
        components = [f"{name}: {json.dumps(payload, ensure_ascii=False)[:600]}" for name, payload in record.ui]
        print(f"CUSTOMER: {text}")
        print(f"ASSISTANT: {(record.text or '').strip()[:ASK_REPLY] or '(no text; see the components)'}")
        for line in calls[:10]:
            print(f"   called: {line}")
        for line in results[:8]:
            print(f"   returned: {line}")
        for line in components[:4]:
            print(f"   component: {line}")
        turns.append({"customer": text, "reply": (record.text or "")[:2000], "calls": calls[:10], "results": results[:8], "components": [name for name, _ in record.ui]})
    error = next((o.detail for o in result.outcomes if o.key == "error"), None)
    if error:
        print(f"run_eval: the conversation stopped: {error[:300]}")
    usage: dict[str, int] = {}
    for record in result.turns:
        for key, value in (record.usage or {}).items():
            usage[key] = usage.get(key, 0) + value
    cost = round(subject_cost(usage, cfg["subject_model"]), 5)
    with contextlib.suppress(OSError):
        with (run / "log" / "redteam_asks.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": time.time(), "role": role, "round": round_index, "n": used + 1, "cart": cart, "seen": seen, "turns": turns, "error": error, "cost_usd": cost}, ensure_ascii=False) + "\n")
        with (run / "log" / "subject_costs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"role": role, "round": round_index, "cases": 1, "trials": 1, "cost_usd": cost, "ask": True}) + "\n")
    return 0


def _budget_check(
    run: Path, role: str, round_index: int, budget: dict[str, Any], n_cases: int
) -> tuple[bool, int, str]:
    counter = run / "log/run_eval_calls" / f"{round_index}-{role}.txt"
    used = int(counter.read_text().strip() or 0) if counter.exists() else 0
    if used >= budget["run_eval_calls"]:
        return (
            False,
            n_cases,
            f"run_eval: you have used all {budget['run_eval_calls']} calls allowed this session.",
        )
    counter.write_text(str(used + 1))
    note = ""
    if n_cases > budget["run_eval_cases"]:
        note = f"(only the first {budget['run_eval_cases']} of {n_cases} cases were run: the per-call limit)"
        n_cases = budget["run_eval_cases"]
    return True, n_cases, note


def select_cases(
    args: argparse.Namespace, cases_root: Path, features: list[str], schema: str = "v1"
) -> list[tuple[str, Case]]:
    if args.gold:
        gold = load_cases(Path(args.gold))
        return [(gold.feature, c) for c in gold.cases]
    chosen: list[tuple[str, Case]] = []
    for feature in [*features, "safety"]:
        for case in load_case_dir(cases_root, feature, schema).cases:
            chosen.append((feature, case))
    if args.feature:
        chosen = [(f, c) for f, c in chosen if f == args.feature]
    if args.ids:
        wanted = {part for raw in args.ids for part in raw.split(",") if part}
        chosen = [(f, c) for f, c in chosen if c.id in wanted]
    return chosen


def summarize(run, feature: str) -> dict[str, Any]:
    last = run.turns[-1] if run.turns else None
    return {
        "id": run.case_id,
        "feature": feature,
        "clause": run.clause,
        "passed": run.passed,
        "skipped": run.skipped,
        "failures": [o.detail for o in run.failures()],
        "rubric_score": next((o.score for o in run.outcomes if o.key == "rubric" and o.score is not None), None),
        "reply": (last.text[:600] if last else ""),
        "tools": (
            [f"{c.tool}({json.dumps(c.input, ensure_ascii=False)[:80]})" for c in last.tool_calls]
            if last
            else []
        ),
        "components": (last.components if last else []),
        "cart": (last.cart_ids if last else []),
        "memory": ([f"{f.get('key')}: {f.get('value')}" for f in last.memory] if last else []),
        "usage": (last.usage if last else {}),
        "search_results": [
            r.summary[:400]
            for t in run.turns
            for r in t.tool_results
            if r.tool == "search_products" and r.summary
        ][:4],
        # What each service call of the conversation returned (shown to a role only when the
        # run's configuration says so).
        "service_results": [
            f"{r.tool} [{r.status}]: {r.text[:500]}"
            for t in run.turns
            for r in t.tool_results
            if r.tool != "search_products" and not r.tool.startswith("present_") and r.tool != "load_skill"
        ][:8],
    }


def _did_not_start(case: Any, feature: str, why: str) -> dict[str, Any]:
    """A case's result when the assistant's code could not be imported."""
    return {
        "id": case.id, "feature": feature, "clause": getattr(case, "clause", None), "passed": False, "skipped": None,
        "failures": [why], "rubric_score": None, "reply": "", "tools": [], "components": [], "cart": [], "memory": [],
        "usage": {}, "search_results": [], "service_results": [],
    }


def aggregate(
    per_trial: list[dict[str, Any]], trials: int, pass_rule: str = "all"
) -> list[dict[str, Any]]:
    """One result per case over its trials: ``passed`` when every trial passed (``all``) or
    when more than half did (``majority``), ``pass_rate`` the fraction of trials that did,
    the failure details from the first failing trial (tagged when other trials passed), the
    other fields from that trial or the last one, and the usage summed over the trials."""
    if trials <= 1:
        for r in per_trial:
            r["pass_rate"] = None if r.get("skipped") else (1.0 if r["passed"] else 0.0)
            r["trials"] = 1
        return per_trial
    order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in per_trial:
        if r["id"] not in groups:
            order.append(r["id"])
        groups.setdefault(r["id"], []).append(r)
    out = []
    for cid in order:
        runs = groups[cid]
        done = [r for r in runs if not r.get("skipped")]
        if not done:
            out.append({**runs[0], "pass_rate": None, "trials": len(runs)})
            continue
        failing = [r for r in done if not r["passed"]]
        passed_n = len(done) - len(failing)
        passed = passed_n * 2 > len(done) if pass_rule == "majority" else not failing
        # A passing case shows a passing trial's fields; a failing one its first failure's.
        base = dict(next(r for r in done if r["passed"]) if passed else failing[0])
        base["passed"] = passed
        base["pass_rate"] = round(passed_n / len(done), 3)
        base["trials"] = len(done)
        base["pass_rule"] = pass_rule
        scores = [r.get("rubric_score") for r in done if r.get("rubric_score") is not None]
        base["rubric_scores"] = scores
        if failing and passed_n:
            base["failures"] = [
                f"(failed in {len(failing)} of {len(done)} trials) {f}"
                for f in failing[0]["failures"]
            ]
            if passed:
                base["failures"] = [f"(passed by majority) {f}" for f in base["failures"]]
        usage: dict[str, int] = {}
        for r in done:
            for key, value in r["usage"].items():
                usage[key] = usage.get(key, 0) + value
        base["usage"] = usage
        out.append(base)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_eval")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", default="harness")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--feature")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gold")
    parser.add_argument("--cases-root", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--trials", type=int, default=1, help="runs per case; results are aggregated per case"
    )
    parser.add_argument(
        "--pass-rule", choices=("all", "majority"), default="all",
        help="over several trials a case passes when every trial passed, or when most did",
    )
    parser.add_argument(
        "--list-queries", nargs="*", metavar="WORD",
        help="list the world's judged queries (those containing every WORD) and exit; no cases run",
    )
    parser.add_argument(
        "--ask-store", nargs=2, metavar=("METHOD", "JSON"),
        help="put one request to the store's service and print its answer; no cases run (the `probe` arm's evaluator)",
    )
    parser.add_argument(
        "--list-products", nargs="*", metavar="WORD",
        help="list the catalog's products whose title contains every WORD (id, title, category, price) and exit; no cases run",
    )
    parser.add_argument("--validate", action="store_true", help="say which of the chosen case files parse as cases and exit; the assistant is not run")
    parser.add_argument("--ask", nargs="+", metavar="MESSAGE", help="hold one conversation with the assistant and print each turn; nothing is graded (the red team)")
    parser.add_argument("--cart", nargs="*", default=[], metavar="ID", help="with --ask: products in the customer's cart")
    parser.add_argument("--seen", nargs="*", default=[], metavar="ID", help="with --ask: products the customer has been shown")
    parser.add_argument("--compare", metavar="NAME", help="hold the comparison `comparisons/NAME.json` of the comparing red team and print what it shows")
    parser.add_argument("--compare-before", type=Path, help=argparse.SUPPRESS)  # the child process of a comparison (experiment/compare.py)
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=True)
    cfg = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    schema = cfg.get("case_schema", "v1")
    if args.ask_store is not None:
        return ask_store(args.run, cfg, args.role, args.round, *args.ask_store)
    if args.list_queries is not None:
        from harness.graders import configure_world, judged_queries

        configure_world(data_dir_of(Path(cfg["world_tree"]), cfg.get("world") or "esci"))
        rows = judged_queries(args.list_queries)
        print(f"{len(rows)} judged quer{'y' if len(rows) == 1 else 'ies'}" + (f" containing {args.list_queries}" if args.list_queries else "") + " (id | text | E/S/C/I counts):")
        for qid, text, counts in rows[:80]:
            print(f"  {qid} | {text} | E {counts['E']} S {counts['S']} C {counts['C']} I {counts['I']}")
        if len(rows) > 80:
            print(f"  ... and {len(rows) - 80} more; add words to narrow the list")
        return 0
    if args.list_products is not None:
        return list_products(Path(cfg["world_tree"]), cfg.get("world", "esci"), args.list_products)
    if args.ask:
        return ask_assistant(args.run, cfg, args.role, args.round, args.ask, args.cart, args.seen, args.tree or args.run / "workspace")
    if args.compare_before is not None:
        from .compare import before_main

        return before_main(args.run, cfg, args.compare_before)
    if args.compare:
        from .compare import compare

        return compare(args.run, cfg, args.role, args.round, args.compare)
    cut_apart = cfg.get("source") == "failures" and cfg.get("arrangement") == "isolated"
    revision = int(cfg.get("revision") or 1)
    on_start = cut_apart and args.role == "V" and not args.validate and revision >= 2
    if cut_apart and args.role == "V" and not args.validate and not on_start:
        # The evaluator of a cut-apart system never sees the assistant: no case is run for it.
        print(CUT_APART_NOTE)
        return 2
    if not (args.ids or args.feature or args.all or args.gold):
        print("run_eval: say which cases: --ids <id> ..., --feature F<n>, or --all")
        return 2
    # The cut-apart evaluator's cases run on the assistant as it was before any change: it
    # works before the developer does, and is never shown the developer's change.
    tree = args.tree or (args.run / "start-tree" if on_start else args.run / "workspace")
    cases_root = args.cases_root or args.run / "cases"
    if cut_apart and revision >= 2 and schema == "v2" and not args.cases_root and not args.gold:
        schema = "v2-apart"  # the evaluation tree's cases; the developer's own smoke cases keep the whole vocabulary
    if on_start:
        print(CUT_APART_START)
    # The developer of a cut-apart system is told how many of the evaluator's cases pass and
    # which fail by number; its own smoke cases (--cases-root) are shown to it in full.
    redact = cut_apart and args.role == "F" and not args.cases_root
    chosen = select_cases(args, cases_root, cfg["features"], schema)
    if not args.gold:
        invalid = invalid_files(cases_root, schema)
        if redact:
            if invalid:
                print(f"run_eval: {len(invalid)} case file{'' if len(invalid) == 1 else 's'} of the evaluation do not parse and count for nothing.")
        else:
            for rel, why in invalid.items():
                print(f"INVALID {rel}: not a valid case, skipped ({why[:200]})")
    if args.validate:
        by_feature: dict[str, list[str]] = {}
        for feature, case in chosen:
            by_feature.setdefault(feature, []).append(case.id)
        for feature, ids in by_feature.items():
            short = f" (a topic needs at least {cfg['min_cases']})" if feature != "safety" and len(ids) < int(cfg.get("min_cases", 0)) else ""
            print(f"run_eval --validate: {feature}: {len(ids)} valid case{'' if len(ids) == 1 else 's'}{short}: {', '.join(ids)}")
        if not chosen:
            print("run_eval --validate: no valid case matched. Case files live under F<n>/ in the evaluation tree; ids are the `id` field.")
        print("run_eval --validate: the files were read; the assistant was not run.")
        return 0 if chosen else 1
    if not chosen:
        print(
            "run_eval: no cases matched. Case files live under F<n>/ in the evaluation tree; ids are the `id` field."
        )
        return 1
    if args.role in ("F", "V"):
        budget = cfg["f_budget"] if args.role == "F" else cfg["v_budget"]
        allowed, n, note = _budget_check(args.run, args.role, args.round, budget, len(chosen))
        if not allowed:
            print(note)
            return 2
        chosen = chosen[:n]
    else:
        note = ""
    trials = max(1, args.trials)
    try:
        activate(tree, Path(cfg["world_tree"]))
        # The backend implementation is built on the assistant's own packages (its protocol and its
        # types): a tree that changed them so that the backend no longer fits does not start.
        from harness.runner import Runner, World, run_file_sync

        world = World(Path(cfg["world_tree"]), kind=cfg.get("world", "esci"))
    except Exception as error:  # noqa: BLE001 - whatever the assistant's own code raises on import
        if isinstance(error, RuntimeError) and "resolved to" in str(error):
            raise  # a package found outside its tree is the harness's fault, not the assistant's
        # The assistant does not start (a syntax error, a missing name). That is a result, not
        # a failure of the evaluation: every case fails, and whoever asked is told why.
        why = f"the assistant does not start: {type(error).__name__}: {error}"[:400]
        results = aggregate([_did_not_start(case, feature, why) for feature, case in chosen for _ in range(trials)], trials, args.pass_rule)
        payload = {
            "round": args.round, "role": args.role, "tree": str(tree), "trials": trials, "pass_rule": args.pass_rule,
            "results": results, "subject_usage": {}, "judge_usage": {"input": 0, "output": 0, "calls": 0}, "cost_usd": 0.0,
            "did_not_start": why,
        }
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        if not args.quiet:
            print(f"run_eval: {why}")
            print(f"run_eval: {len(results)} cases, 0 pass, {len(results)} fail (no case can run until the assistant starts) {note}")
        return 0
    judge = _judge(cfg["dry_run"], cfg.get("judge_model"))
    runner = Runner(
        tree,
        world,
        model=cfg["subject_model"],
        client=_subject_client(cfg["dry_run"]),
        judge=judge,
        thinking_effort=cfg.get("subject_thinking"),
    )
    report = run_file_sync(runner, CaseFile(feature="mixed", cases=[c for _, c in chosen]), trials)
    unreachable = sorted(
        {r.case_id for r in report.runs if any(o.key == "network" for o in r.outcomes)}
    )
    if unreachable:
        print(
            f"run_eval: the API was unreachable during {len(unreachable)} case run(s) ({', '.join(unreachable[:4])}"
            f"{', ...' if len(unreachable) > 4 else ''}); nothing was scored. Run it again once the network is back."
        )
        return NETWORK_EXIT
    by_id = {c.id: f for f, c in chosen}
    results = aggregate(
        [summarize(run, by_id.get(run.case_id, "?")) for run in report.runs], trials, args.pass_rule
    )
    totals: dict[str, int] = {}
    for item in results:
        for key, value in item["usage"].items():
            totals[key] = totals.get(key, 0) + value
    assistant_cost = subject_cost(totals, cfg["subject_model"])
    judge_usage = (
        getattr(judge, "usage", {"input": 0, "output": 0, "calls": 0})
        if judge
        else {"input": 0, "output": 0, "calls": 0}
    )
    jprice = price_of(cfg.get("judge_model") or "") or (0.0, 0.0, 0.0)
    # A judge on one of the harness's own model adapters carries its cost from the ledger.
    judge_cost = getattr(judge, "cost", None) if judge else 0.0
    if judge_cost is None:
        judge_cost = judge_usage["input"] * jprice[0] + judge_usage["output"] * jprice[1]
    payload = {
        "round": args.round,
        "role": args.role,
        "tree": str(tree),
        "trials": trials,
        "pass_rule": args.pass_rule,
        "results": results,
        "subject_usage": totals,
        "judge_usage": judge_usage,
        "cost_usd": round(assistant_cost + judge_cost, 4),
        "subject_cost_usd": round(assistant_cost, 4),
        "judge_cost_usd": round(judge_cost, 4),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    # Every evaluation's subject and judge cost, the roles' own run_eval calls included, so a
    # run's total can be summed after the fact.
    try:
        ledger = args.run / "log" / "subject_costs.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"role": args.role, "round": args.round, "cases": len(chosen), "trials": trials, "cost_usd": payload.get("cost_usd")}) + "\n")
    except OSError:
        pass
    if not args.quiet and redact:
        # Numbers only: the order of the files, which is the same every time.
        passed = sum(r["passed"] for r in results)
        print(f"run_eval: {len(results)} cases, {passed} pass, {len(results) - passed} fail {note}")
        for number, r in enumerate(results, 1):
            print(f"case {number}: {'SKIP' if r['skipped'] else 'PASS' if r['passed'] else 'FAIL'}")
        print("run_eval: the evaluation is kept apart from you; its cases and what the assistant replied to them are not shown.")
    elif not args.quiet:
        passed = sum(r["passed"] for r in results)
        print(f"run_eval: {len(results)} cases, {passed} pass, {len(results) - passed} fail {note}")
        for r in results:
            if r["skipped"]:
                print(f"SKIP {r['id']}: {r['skipped']}")
                continue
            print(
                f"{'PASS' if r['passed'] else 'FAIL'} {r['id']}"
                + (f" ({r['pass_rate']:.0%} of {trials} trials)" if trials > 1 else "")
                + ("" if r["passed"] else " : " + "; ".join(r["failures"])[:400])
            )
            if not r["passed"] or args.ids:
                print(f"   reply: {r['reply'][:300]!r}")
                if r["tools"]:
                    print(f"   tools: {' -> '.join(r['tools'][:8])}")
                if r["components"]:
                    print(f"   components: {r['components']}")
                if r["cart"]:
                    print(f"   cart: {r['cart']}")
                for s in r["search_results"][:2]:
                    print(f"   search result: {s[:300]!r}")
                if cfg.get("service_results"):
                    for s in r.get("service_results", [])[:6]:
                        print(f"   service result: {s[:500]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
