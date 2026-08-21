import json
import logging
from pathlib import Path

from browser_use.llm.messages import UserMessage
from pydantic import ValidationError

from optexity.schema.automation import Automation
from optexity.schema.memory_layer import AutomationPatch, placeholder

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "docs"
    / "action-types"
    / "interaction-action.mdx"
)


PROMPT = """You are improving a browser automation compiled from a recording of an AI
agent completing this objective:

    {objective}

The automation below is already valid and its selectors have been measured against the
live page. Do not try to improve the selectors — you cannot see the page, and they are
verified. Improve only the parts that need understanding of what the steps mean.

Return JSON matching this schema, and nothing else:
{patch_schema}

Guidance:
- rename_parameters: map each current parameter name to a clearer one. Keep them
  lowercase snake_case identifiers. Only rename where it genuinely reads better.
- constant_parameters: list any parameter whose recorded value is configuration
  rather than per-run input (a fixed country, a fixed currency). These stop being
  parameters and are inlined. If in doubt, leave it out — a wrongly frozen value is
  much worse than an extra parameter.
- nodes: for each node give a short prompt_instructions describing the element in
  the page's own words ("The 'Full Name' text input on the contact form"). For nodes
  that are agentic_task, also give a clearer task string; those are the steps the
  compiler could not pin down, so a precise instruction matters most there.

Reference for what these fields mean:
{docs}

The compiled automation:
{automation}

The recorded steps, with what was measured about each:
{trace}
"""


def _trace_digest(trace) -> str:
    """The evidence per compiled row, small enough to sit in a prompt."""
    rows = []
    for index, row in enumerate(trace.compiled_rows()):
        element = row.element
        rows.append(
            {
                "index": index,
                "action": row.action,
                "url": row.url_before,
                "typed": row.params.get("text"),
                "element": (
                    {
                        "tag": element.tag_name,
                        "attributes": element.attributes,
                        "accessible_name": element.accessible_name,
                    }
                    if element
                    else None
                ),
                "why": row.reason,
            }
        )
    return json.dumps(rows, indent=2, default=str)


def apply_patch(automation: Automation, patch: AutomationPatch) -> Automation:
    """Apply a patch and re-validate. Raises if the result is not a valid Automation."""
    payload = automation.model_dump(mode="json", exclude_none=True)
    parameters = payload.setdefault("parameters", {}).setdefault("input_parameters", {})

    renames, inlined = [], {}
    for old_name, new_name in patch.rename_parameters.items():
        if old_name in parameters and new_name not in parameters:
            parameters[new_name] = parameters.pop(old_name)
            renames.append((f"{{{old_name}[", f"{{{new_name}["))
    for name in patch.constant_parameters:
        current = patch.rename_parameters.get(name, name)
        if current in parameters:
            values = parameters.pop(current)
            inlined[placeholder(current)] = str(values[0]) if values else ""

    for node in payload.get("nodes", []):
        for action in (node.get("interaction_action") or {}).values():
            value = action.get("input_text") if isinstance(action, dict) else None
            if isinstance(value, str):
                for old, new in renames:
                    value = value.replace(old, new)
                action["input_text"] = inlined.get(value, value)

    for node_patch in patch.nodes:
        if not 0 <= node_patch.index < len(payload.get("nodes", [])):
            continue
        interaction = payload["nodes"][node_patch.index].get("interaction_action") or {}
        agentic = interaction.get("agentic_task")
        if agentic is not None and node_patch.agentic_task:
            agentic["task"] = node_patch.agentic_task
        if node_patch.prompt_instructions:
            for action in interaction.values():
                if isinstance(action, dict) and "command" in action:
                    action["prompt_instructions"] = node_patch.prompt_instructions

    return Automation.model_validate(payload)


async def enrich(automation: Automation, trace, llm, objective: str = "") -> Automation:
    """Ask the model for a patch, apply it, and keep it only if it validates.

    Retries with the validation error fed back, which is what makes pydantic the
    contract rather than a formality. Returns the original automation unchanged
    if every attempt fails — a compiled automation that works is worth more than
    an enriched one that does not.
    """
    docs = DOCS.read_text()[:6000] if DOCS.exists() else "(docs unavailable)"
    prompt = PROMPT.format(
        objective=objective or "(not recorded)",
        patch_schema=json.dumps(AutomationPatch.model_json_schema(), indent=2),
        docs=docs,
        automation=automation.model_dump_json(indent=2, exclude_none=True),
        trace=_trace_digest(trace),
    )

    error: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        message = prompt
        if error is not None:
            correction = "Return corrected JSON only."
            message = (
                f"{prompt}\n\nYour last reply failed validation:\n"
                f"{error}\n{correction}"
            )
        try:
            patch = AutomationPatch.model_validate_json(await _ask(llm, message))
            enriched = apply_patch(automation, patch)
            logger.info(f"llm patch applied on attempt {attempt}")
            return enriched
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            error = str(e)[:800]
            logger.warning(f"llm patch attempt {attempt} rejected: {error[:200]}")

    logger.warning("no usable patch; keeping the compiled automation")
    return automation


async def _ask(llm, message: str) -> str:
    response = await llm.ainvoke([UserMessage(content=message)])
    return _strip_fence(response.completion)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text.strip()


def print_enrichment(automation, enriched, history_path, out_path) -> None:
    print(f"\n{history_path}  ->  {out_path}")
    print(f"  parameters : {list(automation.parameters.input_parameters)}")
    print(f"            -> {list(enriched.parameters.input_parameters)}")
    print()
    for index, (old_node, new_node) in enumerate(
        zip(automation.nodes, enriched.nodes, strict=True)
    ):
        old_task = getattr(old_node.interaction_action.agentic_task, "task", None)
        new_task = getattr(new_node.interaction_action.agentic_task, "task", None)
        if old_task != new_task:
            print(f"  node {index} task : {old_task!r}\n              -> {new_task!r}")
    print()
