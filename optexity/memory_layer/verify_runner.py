import argparse
import asyncio
import logging
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from optexity.inference.infra.actual_browser import ActualBrowser
from optexity.inference.infra.browser import Browser
from optexity.inference.models import normalize_model
from optexity.memory_layer.distill import distill
from optexity.memory_layer.verify import apply_verdicts, verify_walk
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


def build_task(automation, endpoint_name: str = "verify") -> Task:
    """run_action_node posts a trajectory every fifth step, so without
    upload_artifacts a verification pass fills the production store."""
    parameters = automation.parameters
    return Task(
        task_id=str(uuid.uuid4()),
        user_id="local",
        recording_id="local",
        company_id="local",
        endpoint_name=endpoint_name,
        automation=automation,
        input_parameters={k: list(v) for k, v in parameters.input_parameters.items()},
        secure_parameters={k: list(v) for k, v in parameters.secure_parameters.items()},
        unique_parameter_names=[],
        created_at=datetime.now(timezone.utc),
        status="running",
        api_key="local",
        max_retries=0,
        upload_artifacts=False,
    )


def missing_parameters(automation) -> list[str]:
    """An empty password makes every later node measure an error page, turning
    one missing input into a page of fabricated verdicts."""
    return [
        name
        for name, values in automation.parameters.input_parameters.items()
        if not values or not str(values[0]).strip()
    ]


async def run(history_path: Path, url: str | None, headless: bool, port: int) -> dict:
    automation, trace = distill(history_path, url)

    unset = missing_parameters(automation)
    if unset:
        raise SystemExit(
            "refusing to verify: no value for "
            + ", ".join(unset)
            + ". Supply them (they were redacted at capture) and re-run."
        )

    unique_child_arn = f"verify_{uuid.uuid4()}"
    task = build_task(automation)
    memory = Memory(unique_child_arn=unique_child_arn)
    memory.update_system_info()
    memory.automation_state.step_index = -1
    memory.automation_state.try_index = 0

    actual_browser = ActualBrowser(
        channel=automation.browser_channel,
        unique_child_arn=unique_child_arn,
        port=port,
        headless=headless,
        allow_cookies=automation.allow_cookies,
    )

    browser = None
    try:
        await actual_browser.start()
        if actual_browser.cdp_url is None:
            raise RuntimeError("browser started but exposed no CDP url")
        browser = Browser(
            memory=memory,
            cdp_url=str(actual_browser.cdp_url),
            llm_model=normalize_model(task.llm_provider, task.llm_model_name),
        )
        await browser.start()
        await browser.go_to_url("about:blank")
        await browser.go_to_url(automation.url, retry_count=3)

        # run_action_node substitutes in place; the caller keeps the original.
        started = time.monotonic()
        report = await verify_walk(trace, deepcopy(automation), task, memory, browser)
        seconds = time.monotonic() - started
    finally:
        if browser is not None:
            try:
                await asyncio.wait_for(browser.stop(), timeout=30)
            except Exception as e:
                logger.warning(f"error stopping browser: {e}")
        try:
            await actual_browser.stop()
        except Exception as e:
            logger.warning(f"error stopping actual browser: {e}")

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
        print(
            f"  [{verdict.status:>11}] {verdict.action:<14} " f"{verdict.reason or ''}"
        )
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
        ("llm tokens", trace.agentic_tokens, report.signals["llm_tokens"]),
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
