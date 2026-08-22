import logging
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from copy import deepcopy

from optexity.memory_layer.agent_history import load_trace
from optexity.memory_layer.capture import AGENT_HISTORY_FILENAME, total_llm_tokens
from optexity.memory_layer.distill.classify import classify
from optexity.memory_layer.distill.compile import trace_to_automation
from optexity.memory_layer.verify.verdicts import apply_verdicts
from optexity.memory_layer.verify.walk import verify_automation
from optexity.schema.automation import Automation
from optexity.schema.memory_layer import (
    Classification,
    LoopResult,
    RoundResult,
    Trace,
    VerdictStatus,
    VerificationReport,
)
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3


def resolve_agentic_rows(
    trace: Trace, report: VerificationReport, task: Task
) -> tuple[Trace, int]:
    """Replace each agentic row with what its own run turned out to be.

    Returns the rebuilt trace and how many rows became deterministic. Rows whose
    fresh history yields nothing usable are left agentic, so the path never
    develops a hole. The capture hook writes agent_history.json for a one-step
    agent exactly as for a whole-task one, so each agentic node leaves fresh
    evidence to distil.
    """
    rows = []
    resolved = 0

    for position, (row, verdict) in enumerate(
        zip(trace.compiled_rows(), report.verdicts, strict=False)
    ):
        if verdict.status != VerdictStatus.AGENTIC:
            rows.append(row)
            continue

        # run_action_node increments step_index once per node, so the node's
        # position is the step directory its agentic run wrote to.
        history = task.logs_directory / f"step_{position}" / AGENT_HISTORY_FILENAME
        if not history.exists():
            rows.append(row)
            continue

        fresh = load_trace(history)
        classify(fresh, row.url_before)
        replacements = [
            candidate
            for candidate in fresh.compiled_rows()
            if candidate.classification == Classification.DETERMINISTIC
        ]
        if not replacements:
            rows.append(row)
            continue

        logger.info(
            f"step {row.step}: {len(replacements)} deterministic row(s) "
            "from its own run"
        )
        resolved += 1
        rows.extend(replacements)

    return Trace(url=trace.url, usage=trace.usage, rows=rows), resolved


async def improve(
    trace: Trace,
    automation: Automation,
    build_session: Callable[..., Awaitable],
    max_rounds: int = MAX_ROUNDS,
) -> tuple[Automation, Trace, LoopResult]:
    """Replay, learn from the agentic steps, recompile. Repeat.

    ``build_session(label, automation)`` is an async callable returning a fresh
    (task, memory, browser, teardown) for one round. It takes the automation
    because each round compiles a new one, and the task carries it. Every round
    needs its own browser: parameter placeholders are substituted into nodes in
    place, and a warm profile would let a login verify against a session that
    was already authenticated.
    """
    result = LoopResult()
    best = (automation, trace)

    for number in range(1, max_rounds + 1):
        started = time.monotonic()
        label = f"round{number}_{uuid.uuid4()}"
        task, memory, browser, teardown = await build_session(label, automation)
        try:
            report = await verify_automation(
                trace, deepcopy(automation), task, memory, browser
            )
        finally:
            await teardown()

        # The walk measures a copy, because run_action_node substitutes parameter
        # placeholders in place. Without this the round's findings -- the probed
        # commands, expect_download -- die with the copy and the loop returns the
        # automation it started with.
        apply_verdicts(automation, report)

        counts = Counter(verdict.status for verdict in report.verdicts)

        round_result = RoundResult(
            number=number,
            nodes=len(report.verdicts),
            verified=counts[VerdictStatus.VERIFIED],
            agentic=counts[VerdictStatus.AGENTIC],
            unresolved=counts[VerdictStatus.DEMOTED]
            + counts[VerdictStatus.UNMEASURED]
            + counts[VerdictStatus.NOT_REACHED],
            llm_tokens=await total_llm_tokens(task, memory),
            seconds=round(time.monotonic() - started, 1),
        )
        result.rounds.append(round_result)
        logger.info(
            f"round {number}: {round_result.verified} verified, "
            f"{round_result.agentic} agentic, {round_result.llm_tokens} tokens"
        )

        regressed = (
            len(result.rounds) > 1
            and round_result.verified < result.rounds[-2].verified
        )
        if regressed:
            result.stopped_because = (
                f"round {number} verified fewer nodes than round {number - 1}; "
                "keeping the previous automation"
            )
            return (*best, result)

        best = (automation, trace)

        # Only agentic rows can be re-learned, and the walk stops on a row that is
        # not one, so another round would stop in the same place.
        if not report.complete:
            result.stopped_because = f"round {number} stopped: {report.stopped_because}"
            return (*best, result)

        # Every step deterministic and measured. On agentic alone, a pass that
        # demoted or never reached its nodes claimed convergence directly above a
        # table reading zero verified.
        if round_result.agentic == 0 and round_result.unresolved == 0:
            result.converged = True
            result.stopped_because = "every step is deterministic"
            return (*best, result)

        trace, resolved = resolve_agentic_rows(trace, report, task)
        if not resolved:
            result.stopped_because = (
                f"{round_result.agentic} step(s) did not resolve on a second look; "
                "they stay agentic"
            )
            return (*best, result)

        automation = trace_to_automation(trace, trace.url)

    result.stopped_because = f"reached the {max_rounds}-round cap"
    return (*best, result)


def format_table(result: LoopResult, trace: Trace) -> str:
    """The before/after the assignment asks for: the agentic run, then each round."""
    columns = (
        "round",
        "nodes",
        "verified",
        "agentic",
        "unresolved",
        "llm tokens",
        "seconds",
    )
    widths = (5, 5, 8, 7, 10, 10, 7)

    def row(cells) -> str:
        return "  " + "   ".join(
            f"{cell:>{width}}" for cell, width in zip(cells, widths)
        )

    lines = [
        row(columns),
        "  " + "-" * (sum(widths) + 3 * (len(widths) - 1)),
        # agentic_tokens is None when the capture recorded no usage -- say so
        # rather than printing a zero the run never earned.
        row(
            (
                "agent",
                len(trace.rows),
                "-",
                len(trace.rows),
                "-",
                trace.agentic_tokens if trace.agentic_tokens is not None else "?",
                f"{trace.agentic_seconds:.1f}",
            )
        ),
    ]
    lines.extend(
        row(
            (
                r.number,
                r.nodes,
                r.verified,
                r.agentic,
                r.unresolved,
                r.llm_tokens if r.llm_tokens is not None else "?",
                r.seconds,
            )
        )
        for r in result.rounds
    )
    return "\n".join(lines + ["", f"  {result.stopped_because}"])
