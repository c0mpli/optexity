import json
import logging
from pathlib import Path

from browser_use import Agent

from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

AGENT_HISTORY_FILENAME = "agent_history.json"


def save_agent_history(agent: Agent, task: Task, step_index: int) -> None:
    """Never raises: a capture failure must not fail an otherwise successful node."""
    try:
        agent.save_history(
            task.logs_directory / f"step_{step_index}" / AGENT_HISTORY_FILENAME
        )
    except Exception as e:
        logger.warning(f"Could not save agent history: {e}")


async def total_llm_tokens(task: Task, memory: Memory) -> int | None:
    """Both halves of a run's LLM cost, or None if an agent ran unmetered.

    memory.token_usage counts only optexity's own calls -- the index fallback,
    error handling, select prediction. handle_agentic_task is not among its
    writers, so browser-use's spend exists solely in the capture files.

    Some browser-use versions save a history with no usage block. A run that
    drove an agent and then reports zero is making a claim rather than a
    measurement, and tokens are the number this layer is judged on.
    """
    captured = 0
    agent_ran = False
    for path in Path(task.logs_directory).glob(f"step_*/{AGENT_HISTORY_FILENAME}"):
        agent_ran = True
        try:
            captured += (json.loads(path.read_text()).get("usage") or {}).get(
                "total_tokens"
            ) or 0
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read {path}: {e}")
    total = captured + memory.token_usage.total_tokens
    return None if agent_ran and total == 0 else total
