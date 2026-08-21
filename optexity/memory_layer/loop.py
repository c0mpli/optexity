"""Re-run a compiled automation until nothing is left for the agent to do.

A first pass rarely compiles every step. Some elements have nothing worth
committing to on the page the agent happened to be looking at, and those rows
compile to a narrow agentic node instead. That node still uses the model on
every run, so it is where the remaining cost lives.

The loop closes on it. Each round replays the automation, and every agentic node
that runs writes its own ``agent_history.json`` exactly as the original run did
-- the capture hook does not care whether it was invoked from a whole-task agent
or a one-step one. So a round produces fresh evidence about precisely the steps
that were still undecided, and that evidence is distilled and spliced back in.

A step that could not be pinned down against one page state gets another attempt
against a different one, and the automation gets more deterministic as rounds
go by. When a step never resolves, it stays agentic -- an honest outcome, and a
stable one, rather than a failure to report.

The loop stops early rather than optimistically: if a round verifies fewer nodes
than the last, the changes are making things worse and the previous automation
is the one to keep.
"""

import logging
import time
import uuid
from copy import deepcopy

from pydantic import BaseModel, Field

from optexity.memory_layer.capture import summarize_token_usage
from optexity.memory_layer.distill import classify, trace_to_automation
from optexity.memory_layer.trace import Classification, Trace, load_trace
from optexity.memory_layer.verify import VerdictStatus, verify_walk
from optexity.schema.automation import Automation

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3


class RoundResult(BaseModel):
    number: int
    nodes: int
    verified: int
    agentic: int
    unresolved: int
    llm_tokens: int = 0
    seconds: float = 0.0

    @property
    def deterministic_share(self) -> float:
        return self.verified / self.nodes if self.nodes else 0.0


class LoopResult(BaseModel):
    rounds: list[RoundResult] = Field(default_factory=list)
    converged: bool = False
    stopped_because: str = ""

    @property
    def improved(self) -> bool:
        return len(self.rounds) > 1 and self.rounds[-1].agentic < self.rounds[0].agentic


async def _tokens(task, memory) -> int:
    """Both halves of a round's LLM cost.

    memory.token_usage counts only optexity's own calls -- the index fallback,
    error handling, select prediction. handle_agentic_task is not among its
    writers, so browser-use's spend exists solely in the capture files.
    """
    agentic = await summarize_token_usage(task.logs_directory)
    return agentic["total_tokens"] + memory.token_usage.total_tokens


def resplice(trace: Trace, report, task) -> tuple[Trace, int]:
    """Replace each agentic row with what its own run turned out to be.

    Returns the rebuilt trace and how many rows became deterministic. Rows whose
    fresh history yields nothing usable are left agentic, so the path never
    develops a hole.
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
        history = task.logs_directory / f"step_{position}" / "agent_history.json"
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
    build_run,
    max_rounds: int = MAX_ROUNDS,
) -> tuple[Automation, Trace, LoopResult]:
    """Replay, learn from the agentic steps, recompile. Repeat.

    ``build_run(label, automation)`` is an async callable returning a fresh
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
        task, memory, browser, teardown = await build_run(label, automation)
        try:
            report = await verify_walk(
                trace, deepcopy(automation), task, memory, browser
            )
        finally:
            await teardown()

        counts = {status: 0 for status in VerdictStatus}
        for verdict in report.verdicts:
            counts[verdict.status] += 1

        round_result = RoundResult(
            number=number,
            nodes=len(report.verdicts),
            verified=counts[VerdictStatus.VERIFIED],
            agentic=counts[VerdictStatus.AGENTIC],
            unresolved=counts[VerdictStatus.DEMOTED]
            + counts[VerdictStatus.UNMEASURED]
            + counts[VerdictStatus.NOT_REACHED],
            llm_tokens=await _tokens(task, memory),
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

        if round_result.agentic == 0:
            result.converged = True
            result.stopped_because = "every step is deterministic"
            return (*best, result)

        trace, resolved = resplice(trace, report, task)
        if not resolved:
            result.stopped_because = (
                f"{round_result.agentic} step(s) did not resolve on a second look; "
                "they stay agentic"
            )
            return (*best, result)

        automation = trace_to_automation(trace, trace.url)

    result.stopped_because = f"reached the {max_rounds}-round cap"
    return (*best, result)


def format_table(result: LoopResult) -> str:
    """The before/after the assignment asks for, one row per round."""
    lines = [
        "  round   nodes   verified   agentic   unresolved   llm tokens   seconds",
        "  " + "-" * 68,
    ]
    for round_result in result.rounds:
        lines.append(
            f"  {round_result.number:>5}   {round_result.nodes:>5}   "
            f"{round_result.verified:>8}   {round_result.agentic:>7}   "
            f"{round_result.unresolved:>10}   {round_result.llm_tokens:>10}   "
            f"{round_result.seconds:>7}"
        )
    lines.append("")
    lines.append(f"  {result.stopped_because}")
    return "\n".join(lines)
