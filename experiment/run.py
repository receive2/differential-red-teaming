# SPDX-License-Identifier: Apache-2.0

"""Launch one run.

A run of the paper's experiments is one line of a job list (``paper/jobs/R-*.jobs``,
``W-*.jobs``), for example

    python -m experiment.run --name R-0509 --arrangement compare --seed 38 --features F3 --arrivals F3=1 \
        --world esci-zip --role-model openai:gpt-5.6-luna --source failures --rounds 5 --countdown 5 \
        --trials 2 --schema v2 --gold-rounds never --no-adjudication --end-when-resolved \
        --subject-model claude-haiku-5-5 --subject-thinking low --judge-model openai:gpt-5.6-luna \
        --service-results --f-eval-calls 2 --v-eval-calls 4 --cost-cap 4 --world-tree runs/world-esci-e57457b

and an offline run with scripted agents, which calls no model:

    python -m experiment.run --name dry --source failures --arrangement pair --dry-run --rounds 2 \
        --features F1 --subject-model dry --min-cases 1 --end-when-resolved --no-adjudication --gold-rounds never
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from harness.mini.models import LEDGER
from harness.worlds import WORLD_KINDS

from .config import ARRANGEMENTS, ROOT, ExperimentConfig
from .loop import Experiment, scripted_factories
from .report import write_report
from .workspace import prepare_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.run")
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=("coupled", "isolated"), default=None, help="the spec-driven study's mode; a failure-driven run takes it from --arrangement")
    parser.add_argument("--source", choices=("spec", "failures"), default="spec", help="what a topic arrives as: a spec, or a batch of conversations in which the assistant failed a customer (experiment/failures.py)")
    parser.add_argument("--arrangement", choices=ARRANGEMENTS, default=None, help="the team of a failure-driven run: solo, pair, isolated (cut apart both ways), monitor, redteam, compare (a red team that holds comparisons), sealed")
    parser.add_argument("--revision", type=int, default=None, help="the protocol of the failure-driven study, 1 to 4 as experiment/config.py describes (the latest when absent)")
    parser.add_argument("--world-tree", type=Path, default=None, help="the checkout the backend and the shared library are taken from (experiment/config.py's when absent)")
    parser.add_argument("--third-model", default=None, help="another model for the monitor or the red team (provider:model)")
    parser.add_argument("--label-judges", nargs="+", default=None, help="the judges of the sealed arrangement's release check (provider:model; the first decides)")
    parser.add_argument("--m-steps", type=int, default=None, help="model calls a monitor session may make (experiment/config.py when absent)")
    parser.add_argument("--r-steps", type=int, default=None, help="model calls a red-team session may make (experiment/config.py when absent)")
    parser.add_argument("--redteam-asks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--features", nargs="+", default=["F1", "F2", "F3"])
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--countdown", type=int, default=4)
    parser.add_argument("--spacing", type=int, default=2)
    parser.add_argument(
        "--arrivals", nargs="*", default=None, metavar="F=ROUND",
        help="explicit arrival rounds, e.g. F1=1 F2=2 F3=3; overrides --spacing",
    )
    parser.add_argument("--trials", type=int, default=1, help="trials per case at the gate")
    parser.add_argument("--gold-trials", type=int, default=1, help="trials per hidden gold case (majority passes)")
    parser.add_argument("--adjudication-trials", type=int, default=1, help="trials per case when V's edits run on the references")
    parser.add_argument("--schema", choices=("v1", "v2"), default="v1", help="the case vocabulary V writes in (v2: outcomes only)")
    parser.add_argument("--reference-trees", nargs="*", default=[], metavar="DIR", help="further reference implementations for the adjudication")
    parser.add_argument("--audit-model", default=None, help="provider:model that audits V's edits against the spec (an instrument); off when absent")
    parser.add_argument("--f-eval-calls", type=int, default=2)
    parser.add_argument("--v-eval-calls", type=int, default=3)
    parser.add_argument(
        "--show-clock-to-evaluation", action="store_true",
        help="isolated mode: tell V and O_V the countdown and the gate's threshold",
    )
    parser.add_argument(
        "--role-model", default=os.environ.get("TESTBED_ROLE_MODEL", "openai:gpt-5.6-luna")
    )
    parser.add_argument(
        "--subject-model", default=os.environ.get("TESTBED_SUBJECT_MODEL", "claude-haiku-4-5")
    )
    parser.add_argument(
        "--judge-model", default=os.environ.get("TESTBED_JUDGE_MODEL", "claude-haiku-4-5")
    )
    parser.add_argument(
        "--subject-thinking",
        default=os.environ.get("TESTBED_SUBJECT_THINKING") or None,
        help="adaptive thinking effort for the subject model (low|medium|high); off when absent",
    )
    parser.add_argument("--cost-cap", type=float, default=120.0)
    parser.add_argument("--no-adjudication", action="store_true", help="skip the per-round classification of V's edits on the references")
    parser.add_argument("--service-results", action="store_true", help="run_eval shows the roles the store's result of each service call")
    parser.add_argument("--evaluator", choices=("spec", "world", "none", "contract", "mandate", "audit", "probe"), default="spec", help="the evaluator arm: from the spec; with the storefront's service contract; with an automatic comparison of the spec and the contract; with requests of its own to the store's service (probe); or no evaluator (the developer writes the cases)")
    parser.add_argument("--v-model", default=None, help="another model for the evaluator alone (provider:model)")
    parser.add_argument("--spec-variant", default=None, help="publish specs/variants/<spec>.<variant>.md instead of the spec (a wording control)")
    parser.add_argument("--gold-rounds", choices=("every", "final", "never"), default="every", help="when the hidden gold runs")
    parser.add_argument("--end-when-resolved", action="store_true", help="stop once no feature is active and none is still to arrive")
    parser.add_argument("--world", choices=WORLD_KINDS, default="esci")
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--min-cases", type=int, default=6)
    parser.add_argument("--f-steps", type=int, default=None, help="model calls a developer session may make (experiment/config.py when absent)")
    parser.add_argument("--v-steps", type=int, default=None, help="model calls an evaluator session may make (experiment/config.py when absent)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.source == "failures":
        if not args.arrangement:
            parser.error("--source failures needs --arrangement")
        # The arrangement fixes who is on the system: no evaluator (`solo`), the two cut apart
        # (`isolated`), or the pair as in Mode 1, with or without a release check.
        args.evaluator = "none" if args.arrangement == "solo" else "spec"
        args.mode = "isolated" if args.arrangement == "isolated" else "coupled"
    elif args.arrangement:
        parser.error("--arrangement is for --source failures")
    if not args.mode:
        parser.error("--mode is required")
    load_dotenv(ROOT / ".env", override=True)
    cfg = ExperimentConfig(
        name=args.name,
        mode=args.mode,
        source=args.source,
        arrangement=args.arrangement,
        third_model=args.third_model,
        redteam_asks=args.redteam_asks,
        seed=args.seed,
        features=args.features,
        total_rounds=args.rounds,
        countdown=args.countdown,
        arrival_spacing=args.spacing,
        role_model=args.role_model,
        subject_model=args.subject_model,
        judge_model=None if args.judge_model in ("", "none") else args.judge_model,
        subject_thinking=args.subject_thinking,
        cost_cap_usd=args.cost_cap,
        dry_run=args.dry_run,
        arrivals=(
            {k: int(v) for k, v in (item.split("=") for item in args.arrivals)}
            if args.arrivals
            else None
        ),
        eval_trials=args.trials,
        gold_trials=args.gold_trials,
        adjudication_trials=args.adjudication_trials,
        case_schema=args.schema,
        reference_trees=[Path(p).resolve() for p in args.reference_trees],
        audit_model=args.audit_model,
        show_clock_to_evaluation=args.show_clock_to_evaluation,
        evaluator=args.evaluator,
        adjudicate=not args.no_adjudication,
        gold_rounds=args.gold_rounds,
        end_when_resolved=args.end_when_resolved,
        world=args.world,
        tau=args.tau,
        min_cases=args.min_cases,
        service_results=args.service_results,
        v_model=args.v_model,
        spec_variant=args.spec_variant,
    )
    if args.revision is not None:
        cfg.revision = args.revision
    if args.world_tree is not None:
        cfg.world_tree = args.world_tree.resolve()
    for budget, steps in ((cfg.f_budget, args.f_steps), (cfg.v_budget, args.v_steps), (cfg.m_budget, args.m_steps), (cfg.r_budget, args.r_steps)):
        if steps is not None:
            budget.steps = steps
    if args.label_judges:
        cfg.label_judges = list(args.label_judges)
    cfg.f_budget.run_eval_calls, cfg.v_budget.run_eval_calls = args.f_eval_calls, args.v_eval_calls
    LEDGER.cap = cfg.cost_cap_usd
    paths = prepare_run(cfg)
    experiment = Experiment(cfg, paths, scripted_factories() if args.dry_run else None)
    try:
        experiment.run()
    finally:
        report = write_report(paths.run)
        print(f"report: {report}")
        print(f"roles ${LEDGER.total:.2f} + subject/judge ${experiment.subject_cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
