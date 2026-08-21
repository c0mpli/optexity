import json
import logging
from collections import Counter
from pathlib import Path

import aiofiles
from browser_use import Agent

from optexity.schema.memory import Memory
from optexity.schema.memory_layer import CapturedUsage
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

AGENT_HISTORY_FILENAME = "agent_history.json"
SUMMED_USAGE_FIELDS = (
    "entry_count",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
)


def save_agent_history(agent: Agent, task: Task, step_index: int) -> None:
    """Never raises: a capture failure must not fail an otherwise successful node."""
    try:
        agent.save_history(
            task.logs_directory / f"step_{step_index}" / AGENT_HISTORY_FILENAME
        )
    except Exception as e:
        logger.warning(f"Could not save agent history: {e}")


async def summarize_token_usage(logs_directory: str | Path) -> CapturedUsage:
    token_usage_totals = Counter(
        dict.fromkeys(("agentic_nodes", *SUMMED_USAGE_FIELDS), 0)
    )

    for agent_history_path in sorted(
        Path(logs_directory).glob(f"step_*/{AGENT_HISTORY_FILENAME}")
    ):
        try:
            async with aiofiles.open(agent_history_path) as f:
                node_token_usage = json.loads(await f.read()).get("usage") or {}
        except Exception as e:
            logger.warning(f"Could not read {agent_history_path}: {e}")
            continue

        token_usage_totals["agentic_nodes"] += 1
        token_usage_totals.update(
            {field: node_token_usage.get(field) or 0 for field in SUMMED_USAGE_FIELDS}
        )

    return CapturedUsage(**token_usage_totals)


async def total_llm_tokens(task: Task, memory: Memory) -> int:
    """Both halves of a run's LLM cost.

    memory.token_usage counts only optexity's own calls -- the index fallback,
    error handling, select prediction. handle_agentic_task is not among its
    writers, so browser-use's spend exists solely in the capture files.
    """
    captured = await summarize_token_usage(task.logs_directory)
    return captured.total_tokens + memory.token_usage.total_tokens
