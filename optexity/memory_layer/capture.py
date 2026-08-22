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


async def total_llm_tokens(task: Task, memory: Memory) -> int:
    """Both halves of a run's LLM cost.

    memory.token_usage counts only optexity's own calls -- the index fallback,
    error handling, select prediction. handle_agentic_task is not among its
    writers, so browser-use's spend exists solely in the capture files.
    """
    captured = 0
    for path in Path(task.logs_directory).glob(f"step_*/{AGENT_HISTORY_FILENAME}"):
        try:
            captured += (json.loads(path.read_text()).get("usage") or {}).get(
                "total_tokens"
            ) or 0
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read {path}: {e}")
    return captured + memory.token_usage.total_tokens
