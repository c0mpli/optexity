import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from uvicorn import run

logger = logging.getLogger(__name__)

env_path = os.getenv("ENV_PATH")
if not env_path:
    logger.warning("ENV_PATH is not set, using default values")
else:
    load_dotenv(env_path)


def install_browsers() -> None:
    """Install Playwright + Patchright browsers."""
    try:
        subprocess.run(
            ["playwright", "install", "--with-deps", "chromium", "chrome"],
            check=True,
        )
        subprocess.run(
            ["patchright", "install", "chromium", "chrome"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install browsers", file=sys.stderr)
        sys.exit(e.returncode)


def run_inference(args: argparse.Namespace) -> None:
    from optexity.inference.child_process import get_app_with_endpoints

    app = get_app_with_endpoints(
        is_aws=args.is_aws, child_id=args.child_process_id, port=args.port
    )
    run(
        app,
        host=args.host,
        port=args.port,
    )


def run_distill(args: argparse.Namespace) -> None:
    from optexity.memory_layer.distill.compiler import distill, print_summary

    automation, trace = distill(args.history, args.url)
    args.out.write_text(automation.model_dump_json(indent=2, exclude_none=True))
    if args.trace_out:
        args.trace_out.write_text(trace.model_dump_json(indent=2))
    print_summary(trace, automation, args.history, args.out)


def run_verify(args: argparse.Namespace) -> None:
    import asyncio

    from optexity.memory_layer.improve.loop import format_table
    from optexity.memory_layer.run_verification import (
        print_report,
        run_improvement,
        run_verification,
    )

    if args.rounds > 1:
        automation, trace, loop_result = asyncio.run(
            run_improvement(
                args.history, args.url, args.headless, args.port, args.rounds
            )
        )
        print(format_table(loop_result, trace))
        succeeded = loop_result.converged
    else:
        automation, trace, report, seconds = asyncio.run(
            run_verification(args.history, args.url, args.headless, args.port)
        )
        print_report(report, trace, seconds)
        succeeded = report.complete
    if args.out:
        args.out.write_text(automation.model_dump_json(indent=2, exclude_none=True))
    if not succeeded:
        sys.exit(1)


def run_enrich(args: argparse.Namespace) -> None:
    import asyncio

    from optexity.inference.models.chat_litellm import build_agent_llm
    from optexity.memory_layer.distill.compiler import distill
    from optexity.memory_layer.improve.enrich import enrich, print_enrichment

    automation, trace = distill(args.history, args.url)
    enriched = asyncio.run(
        enrich(automation, trace, build_agent_llm(args.model), args.objective)
    )
    args.out.write_text(enriched.model_dump_json(indent=2, exclude_none=True))
    print_enrichment(automation, enriched, args.history, args.out)


def run_heal(args: argparse.Namespace) -> None:
    from optexity.memory_layer.improve.heal import format_report, heal
    from optexity.schema.automation import Automation

    automation = Automation.model_validate_json(args.automation.read_text())
    report = heal(automation, args.logs)
    print(f"\n{args.automation}")
    print(format_report(report))
    print()
    if args.out:
        args.out.write_text(automation.model_dump_json(indent=2, exclude_none=True))
    if report.rescued:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="optexity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------------------------
    # install-browsers
    # ---------------------------
    install_cmd = subparsers.add_parser(
        "install_browsers",
        help="Install required browsers for Optexity",
        aliases=["install-browsers"],
    )
    install_cmd.set_defaults(func=lambda _: install_browsers())

    # ---------------------------
    # inference
    # ---------------------------
    inference_cmd = subparsers.add_parser(
        "inference", help="Run Optexity inference server"
    )
    inference_cmd.add_argument("--host", default="0.0.0.0")
    inference_cmd.add_argument("--port", type=int, default=9000)
    inference_cmd.add_argument(
        "--child_process_id", "--child-process-id", type=int, default=0
    )
    inference_cmd.add_argument(
        "--is_aws", "--is-aws", action="store_true", default=False
    )

    inference_cmd.set_defaults(func=run_inference)

    # ---------------------------
    # distill
    # ---------------------------
    distill_cmd = subparsers.add_parser(
        "distill", help="Compile a captured agentic run into an Automation"
    )
    distill_cmd.add_argument("history", type=Path, help="path to agent_history.json")
    distill_cmd.add_argument("-o", "--out", type=Path, required=True)
    distill_cmd.add_argument("--url", help="defaults to the recorded url")
    distill_cmd.add_argument("--trace-out", type=Path, help="also write the trace")
    distill_cmd.set_defaults(func=run_distill)

    # ---------------------------
    # verify
    # ---------------------------
    verify_cmd = subparsers.add_parser(
        "verify", help="Measure a distilled automation against the live page"
    )
    verify_cmd.add_argument("history", type=Path, help="path to agent_history.json")
    verify_cmd.add_argument("-o", "--out", type=Path)
    verify_cmd.add_argument("--url", help="defaults to the recorded url")
    verify_cmd.add_argument("--headless", action="store_true")
    verify_cmd.add_argument("--port", type=int, default=9222)
    verify_cmd.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="replay this many times, learning from the agentic steps each round",
    )
    verify_cmd.set_defaults(func=run_verify)

    # ---------------------------
    # enrich
    # ---------------------------
    enrich_cmd = subparsers.add_parser(
        "enrich", help="Let an LLM improve a compiled automation"
    )
    enrich_cmd.add_argument("history", type=Path, help="path to agent_history.json")
    enrich_cmd.add_argument("-o", "--out", type=Path, required=True)
    enrich_cmd.add_argument("--url", help="defaults to the recorded url")
    enrich_cmd.add_argument("--objective", default="")
    enrich_cmd.add_argument("--model")
    enrich_cmd.set_defaults(func=run_enrich)

    # ---------------------------
    # heal
    # ---------------------------
    heal_cmd = subparsers.add_parser(
        "heal", help="Fold a finished run's LLM fallbacks back into its automation"
    )
    heal_cmd.add_argument("automation", type=Path, help="the automation that was run")
    heal_cmd.add_argument("logs", type=Path, help="that run's logs directory")
    heal_cmd.add_argument("-o", "--out", type=Path)
    heal_cmd.set_defaults(func=run_heal)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
