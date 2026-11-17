# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]


class RoleBudget(BaseModel):
    # A session is told its command budget and nothing of its step limit
    # (harness/mini/templates.py): out of commands it is asked to submit and still can; at the
    # step limit it is ended without a report. Every role's step limit therefore lies 20 above
    # its command budget; no session of an earlier experiment (2026-11-12) that handed in a report
    # made more than 6 model calls without a command. In that experiment the limits lay at or
    # below the budgets (developer 30 for 40, evaluator 30 for 30, monitor 20 for 30, red team
    # 30 for 40, 40 with the differential checker), and 694 sessions, 646 of them the
    # developer's, were ended at the limit. The defaults are the developer's: 60 commands,
    # where that experiment told it 40. At 30 model calls 52% of the developer sessions of the
    # DeepSeek role model and 4% of the GPT-5.6 one's were ended; the budget is meant to bind
    # for neither.
    steps: int = 80  # model calls per session
    commands: int = 60  # shell commands per session
    run_eval_calls: int = 2
    run_eval_cases: int = 20
    max_tokens: int = 3000
    wall_seconds: int = 1500


ARRANGEMENTS = ("solo", "pair", "isolated", "monitor", "redteam", "compare", "sealed")


class ExperimentConfig(BaseModel):
    name: str
    mode: Literal["coupled", "isolated"]
    # The failure-driven study (experiment/failures.py). ``source`` "failures": a topic arrives
    # as a batch of conversations in which the assistant failed a customer, not as a spec; the
    # system writes the spec of the fix itself. ``arrangement`` says who is on the system:
    #   solo      one agent writes the spec, the cases and the change (no evaluator)
    #   pair      a developer and an evaluator who see each other's work (the baseline)
    #   isolated  the same two, cut apart both ways: the evaluator works once, from the batch
    #             alone, and never sees the assistant; the developer sees neither the spec nor
    #             the cases, only how many pass (from revision 4 the spec is the developer's
    #             and the evaluator reads it)
    #   monitor   the pair, and a release check that reads the evidence before a release
    #   redteam   the pair, and a release check that questions the changed assistant before a
    #             release; what it finds comes back as a further batch
    #   compare   the pair, and a red team that also conducts comparisons: one request asked with
    #             one detail changed, and asked of the assistant as it was before the change;
    #             the tool says whether the backend returned the same, and a rule decides
    #             (experiment/compare.py)
    #   sealed    the pair, and a release check of the backend's own that the system never sees
    #             (the reference: it needs conversations written by someone who knows the backend)
    source: Literal["spec", "failures"] = "spec"
    arrangement: Literal["solo", "pair", "isolated", "monitor", "redteam", "compare", "sealed"] | None = None
    # The revision of the failure-driven study's protocol a run is made with. 1: the pilot of
    # 2026-11-11 (its runs carry no such field). 2: a monitor whose report has no verdict is
    # asked for it once; the cut-apart evaluator may run its cases on the assistant as it was
    # before any change, and its cases may fix neither a tool's inputs nor a component's inner
    # shape; the further batch names the products in the cart; the `compare` arrangement.
    # 3: the comparing red team's second rule (a detail that never reached the backend,
    # experiment/compare.py); nothing else differs from 2.
    # 4: where the system has an evaluator, the developer writes the spec of the fix and the
    # evaluator writes the cases from the batch and that spec, which it cannot change (until
    # then the evaluator wrote both). While a topic has no spec a round runs developer (the
    # spec, in a session of its own in which the assistant is not changed), evaluator (the
    # cases), developer (the change). Cut apart, the evaluator reads the spec and the developer
    # still sees no case. The quantity-cap case of the safety set no longer asks the reply to
    # report the cap (experiment/workspace.py).
    revision: int = 4
    seed: int = 1
    role_model: str = "openai:gpt-5.6-luna"  # F, V, O
    # "disabled" turns extended thinking off for an Anthropic role model. A Claude 5 model thinks
    # before each reply; the thinking is not kept between a session's steps.
    role_thinking: str | None = None
    # Another model for the evaluator alone; the developer and the orchestrator
    # stay on ``role_model``.
    v_model: str | None = None
    subject_model: str = "claude-haiku-4-5"  # the ACME assistant under evaluation
    judge_model: str | None = "claude-haiku-4-5"
    subject_thinking: str | None = None  # adaptive thinking effort for the subject; None disables it
    features: list[str] = Field(default_factory=lambda: ["F1", "F2", "F3"])
    first_arrival_round: int = 1
    arrival_spacing: int = 2
    arrivals: dict[str, int] | None = None  # explicit schedule; overrides the spacing
    # In the isolated mode the evaluation side (V and O_V) is told neither the pass counts
    # nor, unless this is set, the round budget and the gate's threshold.
    show_clock_to_evaluation: bool = False
    countdown: int = 4
    total_rounds: int = 8
    tau: float = 0.8
    epsilon: int = 1
    min_cases: int = 6
    f_budget: RoleBudget = Field(default_factory=RoleBudget)
    v_budget: RoleBudget = Field(
        default_factory=lambda: RoleBudget(steps=50, commands=30, run_eval_calls=3)
    )
    # The release checks of the failure-driven study: the monitor reads, the red team conducts
    # conversations of its own with the changed assistant (``redteam_asks`` a session, of which
    # it has to conduct ``redteam_min_asks`` for its verdict to count) and is called at every gate
    # the system's cases pass, as the other release checks are (``redteam_sessions`` is not read:
    # the round budget bounds the sessions); a release is rejected when more than ``redteam_share``
    # of its conversations are confirmed as findings.
    m_budget: RoleBudget = Field(default_factory=lambda: RoleBudget(steps=50, commands=30, run_eval_calls=0))
    r_budget: RoleBudget = Field(default_factory=lambda: RoleBudget(steps=60, commands=40, run_eval_calls=0))
    redteam_asks: int = 12
    redteam_min_asks: int = 6
    redteam_sessions: int = 3
    redteam_share: float = 0.2
    # The comparing red team (`compare`): how many comparisons it may conduct in a session, and
    # how many it has to conduct for its session to count.
    compare_max: int = 5
    compare_min: int = 3
    # Another model for the monitor or the red team; the role model when absent.
    third_model: str | None = None
    # The judges that label what a candidate tells the customers of the release check (the
    # `sealed` arrangement): the first one decides.
    label_judges: list[str] = Field(default_factory=lambda: ["anthropic:claude-sonnet-5", "openai:gpt-5.6-luna"])
    cost_cap_usd: float = 120.0
    eval_trials: int = 1  # trials per case at the visible gate (a case passes when every trial does)
    # Trials for the measurements outside the loop: the hidden gold and the adjudication of
    # V's edits. Over more than one trial a case passes by majority.
    gold_trials: int = 1
    adjudication_trials: int = 1
    adjudicate: bool = True  # classify V's edits on the references every round
    # "v1": any check the trace supports; "v2": the outcome vocabulary (harness/cases.py).
    case_schema: Literal["v1", "v2"] = "v1"
    # A model that audits each case V adds or changes against the clauses it maps to
    # (entailed / stricter / looser / unrelated); a checker, not an input to any role.
    audit_model: str | None = None
    dry_run: bool = False
    # Which backend the subject talks to (harness.runner.World.KINDS): the ESCI catalog, or
    # upstream's fixture backend with the same service contract (the second world).
    # "esci-zip" and "esci-refuse" are the ESCI backend with a fulfillment service that takes a
    # ZIP and quotes for it, or takes it and refuses it; the further variants are listed in
    # harness/worlds.py (WORLD_KINDS).
    world: str = "esci"
    # The evaluator arm: "spec" (V works from the specs, as in iterations 1 and 2), "world"
    # (V also reads the storefront's service contract, `cases/storefront/CONTRACT.md`),
    # "none" (no evaluator: the developer writes the cases the gate runs).
    # "contract": V reads the contract file and is told only what it is; "mandate": the same
    # and one sentence saying a spec may assume what the backend lacks (experiment/roles.py).
    # "world" is the explicit arm: its note describes the facades and says to fail them.
    # "probe": V reads the contract file, can put requests to the backend's service itself
    # (`run_eval --ask-backend`), and is told to ask again with another value of each input a
    # spec names; no feature and no gap is named to it.
    evaluator: Literal["spec", "world", "none", "contract", "mandate", "audit", "probe"] = "spec"
    # run_eval shows a role the backend's results for each service call of the turn, not only
    # the search results.
    service_results: bool = False
    # A variant of the feature's spec from specs/variants/<spec name>.<variant>.md, published
    # under the spec's usual name.
    spec_variant: str | None = None
    # When the hidden gold runs: every round, or only in the round the run ends.
    gold_rounds: Literal["every", "final", "never"] = "every"
    # Stop after the round in which no feature is active and none is still to arrive.
    end_when_resolved: bool = False
    # Role shells run under the macOS sandbox, and run_eval through the harness's proxy
    # (experiment/sandbox.py). Off only where `sandbox-exec` does not exist.
    sandbox: bool = True
    subject_start: Path = ROOT.parent / "commerce-agents-subject"
    world_tree: Path = ROOT.parent / "commerce-agents"
    reference_tree: Path = ROOT / "runs/trees/reference"
    # Further reference implementations: a case is over-strict only when no reference passes it.
    reference_trees: list[Path] = Field(default_factory=list)
    runs_dir: Path = ROOT / "runs"

    @field_validator("world")
    @classmethod
    def _known_world(cls, value: str) -> str:
        from harness.worlds import WORLD_KINDS

        if value not in WORLD_KINDS:
            raise ValueError(f"unknown world {value!r}; one of {WORLD_KINDS}")
        return value

    @property
    def failure_driven(self) -> bool:
        return self.source == "failures"

    @property
    def developer_writes_cases(self) -> bool:
        """No evaluator on the system: the `none` arm of the spec-driven study, `solo` here."""
        return self.evaluator == "none"

    @property
    def developer_spec(self) -> bool:
        """Does the developer write the spec of the fix on a system that has an evaluator (from
        revision 4)? Without an evaluator (`solo`) the developer always wrote it."""
        return self.failure_driven and self.revision >= 4 and not self.developer_writes_cases

    @property
    def cut_apart(self) -> bool:
        """The `isolated` arrangement of the failure-driven study."""
        return self.failure_driven and self.arrangement == "isolated"

    @property
    def red_team(self) -> bool:
        """An arrangement with a red team: the plain one, or the one that conducts comparisons."""
        return self.failure_driven and self.arrangement in ("redteam", "compare")

    @property
    def keeps_start(self) -> bool:
        """Does the run keep a copy of the assistant as it was before any change (``start-tree``)?
        The cut-apart evaluator's cases are run on it, and the comparing red team asks it."""
        return self.failure_driven and self.revision >= 2 and (self.cut_apart or self.arrangement == "compare")

    @property
    def evaluation_schema(self) -> str:
        """The vocabulary the evaluation tree's cases are held to. Cut apart, a case may not
        fix what only the developer's change will name (``v2-apart``, harness/cases.py)."""
        return "v2-apart" if self.cut_apart and self.case_schema == "v2" and self.revision >= 2 else self.case_schema

    def references(self) -> list[Path]:
        trees = [self.reference_tree, *self.reference_trees]
        return [t for t in trees if t.exists()]

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.name

    def arrival_round(self, feature: str) -> int:
        if self.arrivals and feature in self.arrivals:
            return self.arrivals[feature]
        return self.first_arrival_round + self.features.index(feature) * self.arrival_spacing
