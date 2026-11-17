# SPDX-License-Identifier: Apache-2.0

"""Run case files against a subject tree over the ESCI world. ``activate`` must have put the
trees on the path before this module's functions import the agent packages (they import
lazily for that reason)."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .cases import Case, CaseFile
from .graders import Judge, Outcome, configure_world, grade_turn
from .netwait import online, wait_for_network
from .record import TurnRecord, record_from_events
from .worlds import WORLD_KINDS, data_dir_of, deployment

# A network or server hiccup (TLS failure, dropped connection, 5xx or 429 after the client's
# own retries) is rerun instead of being scored as the subject's failure.
TRANSIENT_RETRIES = 2
TRANSIENT_BACKOFF_SECONDS = 5.0


def is_transient(error: BaseException) -> bool:
    import ssl

    kinds: list[type[BaseException]] = [ssl.SSLError, ConnectionError, TimeoutError]
    try:
        import httpx

        kinds.append(httpx.TransportError)
    except ImportError:  # pragma: no cover
        pass
    try:
        import anthropic

        kinds += [
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
            anthropic.RateLimitError,
        ]
    except ImportError:  # pragma: no cover
        pass
    if isinstance(error, tuple(kinds)):
        return True
    # The provider refusing the account (no credit left) is an outage of another kind.
    from .netwait import account_error

    return account_error(error)


@dataclass
class CaseRun:
    case_id: str
    clause: str | None
    trial: int
    passed: bool
    outcomes: list[Outcome]
    turns: list[TurnRecord]
    skipped: str | None = None

    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.passed]


@dataclass
class RunReport:
    feature: str
    subject: str
    model: str
    trials: int
    runs: list[CaseRun] = field(default_factory=list)

    def pass_rate(self, case_id: str) -> float:
        done = [r for r in self.runs if r.case_id == case_id and not r.skipped]
        return sum(r.passed for r in done) / len(done) if done else 0.0

    def summary(self) -> dict[str, Any]:
        ids = sorted({r.case_id for r in self.runs})
        return {
            "feature": self.feature,
            "subject": self.subject,
            "model": self.model,
            "trials": self.trials,
            "cases": len(ids),
            "passing_every_trial": sum(self.pass_rate(i) == 1.0 for i in ids),
            "failing_every_trial": sum(self.pass_rate(i) == 0.0 for i in ids),
            "per_case": {i: self.pass_rate(i) for i in ids},
        }

    def to_json(self) -> str:
        return json.dumps(
            {"summary": self.summary(), "runs": [asdict(r) for r in self.runs]},
            indent=1,
            default=str,
        )


class World:
    """The frozen backend the subject talks to: a storefront backend from the world tree's
    examples (``esci``: the ESCI catalog with its judged queries; ``retail``: upstream's
    fixture backend with the same service contract), its demo users and orders, and the clock
    the session reports."""

    KINDS = WORLD_KINDS  # the ESCI backend, its variants (harness/worlds.py), and upstream's fixture backend

    def __init__(self, world_tree: Path, data_dir: Path | None = None, kind: str = "esci") -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown world {kind!r}; one of {self.KINDS}")
        self.tree = world_tree
        self.kind = kind
        if kind != "retail":
            # The backend of the kind and the config its example deploys the assistant with
            # (harness/worlds.py).
            self.data_dir = data_dir or data_dir_of(world_tree, kind)
            self.backend, self.config = deployment(kind, self.data_dir)
            manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))
            anchor = manifest.get("orders_anchored_to") or manifest["built_on"][:10]
            self.anchor = datetime.fromisoformat(f"{anchor}T10:00:00")
        else:
            from retail.api.agent_config import build_shopping_config
            from retail.api.mock_retail import MockRetail

            self.data_dir = data_dir or world_tree / "examples/retail/data"
            self.backend = MockRetail(self.data_dir)
            self.config = build_shopping_config()
            # The fixture backend re-dates its orders to the day it loads; the clock follows.
            self.anchor = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)


class Runner:
    def __init__(
        self,
        subject_tree: Path,
        world: World,
        *,
        model: str,
        client: Any = None,
        judge: Judge | None = None,
        max_tool_iterations: int | None = None,
        thinking_effort: str | None = None,
    ) -> None:
        self.subject_tree = subject_tree
        self.world = world
        self.model = model
        self.client = client
        self.judge = judge
        self.max_tool_iterations = max_tool_iterations
        # The deployment config asks for adaptive thinking (a Sonnet 5 field); the subject
        # runs without extended thinking unless the harness says otherwise.
        self.thinking_effort = thinking_effort
        configure_world(world.data_dir)  # the ESCI relevance labels for presented_relevance

    def make_agent(self, memory_store: Any) -> Any:
        from shopping_agent_runtime import ShoppingAgent

        updates: dict[str, Any] = {"model": self.model, "thinking_effort": self.thinking_effort}
        if self.max_tool_iterations is not None:
            updates["max_tool_iterations"] = self.max_tool_iterations
        config = self.world.config.model_copy(update=updates)
        kwargs: dict[str, Any] = {
            "backend": self.world.backend,
            "skills_dir": self.subject_tree / "shopping-agent/skills",
            "config": config,
            "memory_store": memory_store,
        }
        if self.client is not None:
            kwargs["client"] = self.client
        return ShoppingAgent(**kwargs)

    async def run_case(self, case: Case, trial: int = 0) -> CaseRun:
        if case.skip:
            return CaseRun(case.id, case.clause, trial, False, [], [], skipped=case.skip)
        for attempt in range(TRANSIENT_RETRIES + 1):
            run, transient = await self._run_case_once(case, trial, attempt)
            if not transient or attempt == TRANSIENT_RETRIES:
                return run
            print(f"retrying {case.id} trial {trial}: {run.outcomes[-1].detail}", file=sys.stderr)
            if not online():
                wait_for_network(max_wait=300)
            await asyncio.sleep(TRANSIENT_BACKOFF_SECONDS * (attempt + 1))
        return run  # pragma: no cover

    async def _run_case_once(self, case: Case, trial: int, attempt: int) -> tuple[CaseRun, bool]:
        from commerce_common.memory import InMemoryMemoryStore
        from commerce_common.types import MemoryCategory, MemoryFact
        from shopping_agent import ShoppingSessionContext, ShoppingSessionState

        memory_store = InMemoryMemoryStore()
        try:
            agent = self.make_agent(memory_store)
        except Exception as error:  # noqa: BLE001 - the assistant's own code fails before a turn
            # An assistant that cannot be built (a missing name in its tool registry, say) is a
            # failed case like a crashed turn, not a failed evaluation.
            detail = f"the assistant does not start: {type(error).__name__}: {error}"[:400]
            return CaseRun(case.id, case.clause, trial, False, [Outcome("error", False, detail)], []), False
        session_id = f"{case.id}-{trial}-{uuid.uuid4().hex[:6]}"
        now = datetime.fromisoformat(case.now) if case.now else self.world.anchor
        session = ShoppingSessionContext(session_id=session_id, user_id=case.user_id, now=now)
        state = ShoppingSessionState()
        backend = self.world.backend
        # Preconditions: provenance, cart lines, memory facts, straight into the backends.
        try:
            seen = [p for pid in case.state.seen_products if (p := backend.product(pid))]
            state.remember_products(seen)
            for line in case.state.cart:
                product = backend.product(line.product_id)
                if product is not None:
                    state.remember_products([product])
                    await backend.add_to_cart(session, line.product_id, line.quantity)
            if case.state.memory:
                await memory_store.upsert_facts(
                    case.user_id,
                    [
                        MemoryFact(
                            key=m.key,
                            value=m.value,
                            category=MemoryCategory(m.category),
                            updated_at=datetime.fromisoformat(m.updated_at) if m.updated_at else now,
                        )
                        for m in case.state.memory
                    ],
                )
        except Exception as error:  # noqa: BLE001 - a case whose state the backend cannot take
            # A case is its author's: one whose preconditions cannot be set up is a failed case
            # that says why, not a failed evaluation.
            detail = f"the case's preconditions cannot be set up: {type(error).__name__}: {error}"[:400]
            return CaseRun(case.id, case.clause, trial, False, [Outcome("error", False, detail)], []), False
        messages: list[dict[str, Any]] = []
        turns: list[TurnRecord] = []
        outcomes: list[Outcome] = []
        transient = False
        try:
            for index, text in enumerate(case.turns):
                cart_before = [item.product_id for item in (await backend.get_cart(session)).items]
                memory_before = [
                    f.model_dump(mode="json") for f in await memory_store.get_facts(case.user_id)
                ]
                messages.append({"role": "user", "content": text})
                events = [event async for event in agent.stream_turn(messages, session, state)]
                if hasattr(agent, "update_memory"):
                    await agent.update_memory(messages, session)
                record = record_from_events(events)
                record.previous = list(turns)
                record.cart_before, record.memory_before = cart_before, memory_before
                cart = await backend.get_cart(session)
                record.cart_ids = [item.product_id for item in cart.items]
                record.cart_lines = len(cart.items)
                record.memory = [
                    f.model_dump(mode="json") for f in await memory_store.get_facts(case.user_id)
                ]
                turns.append(record)
                expected = case.expected_for(index, last=index == len(case.turns) - 1)
                outcomes.extend(
                    grade_turn(expected, record, messages, self.judge, case.reference_response)
                )
        except Exception as error:  # a crashed turn is a failed case, not a failed run
            transient = is_transient(error)
            detail = f"{type(error).__name__}: {error}"
            if transient:
                detail += f" (attempt {attempt + 1} of {TRANSIENT_RETRIES + 1})"
            # "network": the failure is the connection's, not the subject's; run_eval does not
            # score a report that carries one.
            outcomes.append(Outcome("network" if transient else "error", False, detail))
        finally:
            backend.reset_session(session_id)
        run = CaseRun(case.id, case.clause, trial, all(o.passed for o in outcomes), outcomes, turns)
        return run, transient

    async def run_file(
        self, cases: CaseFile, trials: int = 1, only: set[str] | None = None
    ) -> RunReport:
        report = RunReport(cases.feature, str(self.subject_tree), self.model, trials)
        for case in cases.cases:
            if only and case.id not in only:
                continue
            for trial in range(trials):
                run = await self.run_case(case, trial)
                report.runs.append(run)
                if any(o.key == "network" for o in run.outcomes):
                    # The network is gone: the rest of the file would only record the outage.
                    return report
        return report


def run_file_sync(
    runner: Runner, cases: CaseFile, trials: int = 1, only: set[str] | None = None
) -> RunReport:
    return asyncio.run(runner.run_file(cases, trials, only))
