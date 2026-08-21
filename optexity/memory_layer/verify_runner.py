import argparse
import asyncio
import logging
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path

from optexity.memory_layer.distill import distill
from optexity.memory_layer.session import make_build_session, missing_parameters
from optexity.memory_layer.verdicts import apply_verdicts
from optexity.memory_layer.verify import verify_walk

logger = logging.getLogger(__name__)


async def run(history_path: Path, url: str | None, headless: bool, port: int) -> dict:
    automation, trace = distill(history_path, url)

    unset = missing_parameters(automation)
    if unset:
        raise SystemExit(
            "refusing to verify: no value for "
            + ", ".join(unset)
            + ". Supply them (they were redacted at capture) and re-run."
        )

    build_session = make_build_session(headless, port)
    task, memory, browser, teardown = await build_session(
        f"verify_{uuid.uuid4()}", automation
    )
    try:
        # run_action_node substitutes in place; the caller keeps the original.
        started = time.monotonic()
        report = await verify_walk(trace, deepcopy(automation), task, memory, browser)
        seconds = time.monotonic() - started
    finally:
        await teardown()

    apply_verdicts(automation, report)
    return {
        "trace": trace,
        "automation": automation,
        "report": report,
        "seconds": seconds,
    }


def _print_report(report, trace, seconds: float) -> None:
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
    for key, value in report.signals.items():
        print(f"  {key}: {value}")

    # Both clocks exclude browser startup. The walk's also covers probing, which
    # a plain replay does not pay, so this reads as an upper bound on the cost.
    print(f"\n  {'':<12}{'agentic':>10}{'verified':>10}")
    for label, before, after in (
        ("steps", len(trace.rows), len(report.verdicts)),
        (
            "llm tokens",
            trace.agentic_tokens or "unrecorded",
            report.signals["llm_tokens"],
        ),
        ("seconds", round(trace.agentic_seconds, 1), round(seconds, 1)),
    ):
        print(f"  {label:<12}{before:>10}{after:>10}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m optexity.memory_layer.verify_runner",
        description="Measure a distilled automation's locators against the live page.",
    )
    parser.add_argument("history", type=Path, help="path to agent_history.json")
    parser.add_argument("-o", "--out", type=Path, help="write the verified automation")
    parser.add_argument("--trace-out", type=Path, help="write the annotated trace")
    parser.add_argument("--url")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--port", type=int, default=9222)
    args = parser.parse_args(argv)

    # optexity/__init__ already called basicConfig, so set levels directly.
    logging.getLogger("optexity").setLevel(logging.INFO)
    logging.getLogger("browser_use").setLevel(logging.WARNING)
    if not args.history.exists():
        parser.error(f"no such file: {args.history}")

    result = asyncio.run(run(args.history, args.url, args.headless, args.port))
    _print_report(result["report"], result["trace"], result["seconds"])

    if args.out:
        args.out.write_text(
            result["automation"].model_dump_json(indent=2, exclude_none=True)
        )
    if args.trace_out:
        args.trace_out.write_text(result["trace"].model_dump_json(indent=2))
    return 0 if result["report"].complete else 1


if __name__ == "__main__":
    sys.exit(main())
