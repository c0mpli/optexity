import logging
import time
import uuid
from copy import deepcopy
from pathlib import Path

from optexity.memory_layer.distill.compiler import distill
from optexity.memory_layer.improve.loop import improve
from optexity.memory_layer.verify.session import make_build_session, missing_parameters
from optexity.memory_layer.verify.verdicts import apply_verdicts
from optexity.memory_layer.verify.walk import verify_automation
from optexity.schema.automation import Automation
from optexity.schema.memory_layer import LoopResult, Trace, VerificationReport

logger = logging.getLogger(__name__)


def compile_for_run(history_path: Path, url: str | None) -> tuple[Automation, Trace]:
    """Distil, and refuse a run that would measure an error page.

    An empty password makes every later node measure the login form it never got
    past, turning one missing value into a page of fabricated verdicts.
    """
    automation, trace = distill(history_path, url)
    unset = missing_parameters(automation)
    if unset:
        raise SystemExit(
            "refusing to verify: no value for "
            + ", ".join(unset)
            + ". Supply them (they were redacted at capture) and re-run."
        )
    return automation, trace


async def run_verification(
    history_path: Path, url: str | None, headless: bool, port: int
) -> tuple[Automation, Trace, VerificationReport, float]:
    automation, trace = compile_for_run(history_path, url)

    build_session = make_build_session(headless, port)
    task, memory, browser, teardown = await build_session(
        f"verify_{uuid.uuid4()}", automation
    )
    try:
        # run_action_node substitutes in place; the caller keeps the original.
        started = time.monotonic()
        report = await verify_automation(
            trace, deepcopy(automation), task, memory, browser
        )
        seconds = time.monotonic() - started
    finally:
        await teardown()

    apply_verdicts(automation, report)
    return automation, trace, report, seconds


async def run_improvement(
    history_path: Path, url: str | None, headless: bool, port: int, rounds: int
) -> tuple[Automation, Trace, LoopResult]:
    automation, trace = compile_for_run(history_path, url)
    return await improve(
        trace, automation, make_build_session(headless, port), max_rounds=rounds
    )


def print_report(report, trace, seconds: float) -> None:
    print(f"\n{report.url}")
    print(f"  verified {report.verified_count}/{len(report.verdicts)} nodes")
    if report.stopped_at is not None:
        print(f"  STOPPED at node {report.stopped_at}: {report.stopped_because}")
    print()
    for verdict in report.verdicts:
        print(f"  [{verdict.status:>11}] {verdict.action:<14} {verdict.reason or ''}")
        if verdict.command:
            print(f"                {verdict.command}")
    print(f"\n  final url: {report.final_url}")
    for key, value in report.signals.model_dump().items():
        print(f"  {key}: {value}")

    # Both clocks exclude browser startup. The walk's also covers probing, which
    # a plain replay does not pay, so this reads as an upper bound on the cost.
    print(f"\n  {'':<12}{'agentic':>10}{'verified':>10}")
    for label, before, after in (
        ("steps", len(trace.rows), len(report.verdicts)),
        (
            "llm tokens",
            trace.agentic_tokens or "unrecorded",
            report.signals.llm_tokens,
        ),
        ("seconds", round(trace.agentic_seconds, 1), round(seconds, 1)),
    ):
        print(f"  {label:<12}{before:>10}{after:>10}")
    print()
