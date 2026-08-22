import re
from typing import Any

from optexity.schema.automation import Automation
from optexity.schema.memory_layer import Classification, Trace, TraceRow, placeholder

# Element actions and the InteractionAction field each compiles to. Everything
# else about them -- command, skip_prompt, max_tries -- is identical.
ACTION_FIELD = {
    "click": "click_element",
    "input": "input_text",
    "select_dropdown": "select_option",
    "upload_file": "upload_file",
}


# save_history redacts secure parameters to <secret>KEY</secret>
# (browser-use agent/views.py:330); recording that would bake in a placeholder.
SECRET_PLACEHOLDER = re.compile(r"^<secret>([^<>]+)</secret>$")


# Runtime defaults are tuned for LLM-resolved nodes: 10 tries floors every miss
# at 10s, and end_sleep_time's 5.0 dominates a distilled run.
MAX_TRIES = 2


SLEEP_AFTER_NAVIGATION = 3.0


SLEEP_AFTER_INTERACTION = 0.5


# Narrow enough to do one step, small enough that a wrong task string costs
# seconds rather than a re-exploration of the page.
AGENTIC_FALLBACK_MAX_STEPS = 3


# The parameter name is the automation's public API, so prefer what a human
# wrote; capped because a prose label makes a poor identifier.
PARAMETER_NAME_ATTRIBUTES = ("aria-label", "placeholder", "name", "id")


MAX_PARAMETER_NAME_CHARS = 40


def _parameter_name_for(row: TraceRow, already_used: set[str]) -> str:
    source = row.element.label(PARAMETER_NAME_ATTRIBUTES) if row.element else ""
    if len(source) > MAX_PARAMETER_NAME_CHARS:
        source = ""
    # replace_variables needs a valid identifier, and form fields are often
    # ordered: 04fullname -> fullname.
    slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    slug = re.sub(r"^\d+_?", "", slug) or "input"

    name, suffix = slug, 2
    while name in already_used:
        name, suffix = f"{slug}_{suffix}", suffix + 1
    return name


def _declare_parameter(
    row: TraceRow,
    value: str,
    parameters: dict[str, list[str]],
    already_used: set[str],
) -> str:
    """Record a recorded value as an input parameter, returning its placeholder."""
    redacted = SECRET_PLACEHOLDER.match(value)
    name = redacted.group(1) if redacted else _parameter_name_for(row, already_used)
    already_used.add(name)
    # Declared, never written down. Not secure_parameters: that holds vault
    # references, and a recording has none to emit.
    element = row.element
    is_password = bool(element) and element.attributes.get("type") == "password"
    parameters[name] = [""] if redacted or is_password else [value]
    return placeholder(name)


def _node(row: TraceRow, interaction: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "action_node",
        "interaction_action": interaction,
        "end_sleep_time": (
            SLEEP_AFTER_NAVIGATION if row.changed_url else SLEEP_AFTER_INTERACTION
        ),
    }


def _action_node_for(
    row: TraceRow, parameters: dict[str, list[str]], already_used: set[str]
) -> dict[str, Any]:
    interaction: dict[str, Any] = {}

    if row.action in ACTION_FIELD:
        best = row.best_candidate
        # Single locator only: an or_() bundle is unsafe until measured, which
        # this pure compiler cannot do. verify.choose_command adds one when it can.
        action: dict[str, Any] = {
            "command": best.command if best else None,
            "skip_prompt": True,
        }
        if row.action == "input":
            action["input_text"] = _declare_parameter(
                row, str(row.params.get("text", "")), parameters, already_used
            )
            action["press_enter"] = bool(row.params.get("press_enter"))
        elif row.action == "select_dropdown":
            action["select_values"] = [str(row.params.get("text", ""))]
        elif row.action == "upload_file":
            # A recorded path belongs to the capturing machine, so it is a parameter.
            action["file_path"] = _declare_parameter(
                row, str(row.params.get("path", "")), parameters, already_used
            )
        interaction = {ACTION_FIELD[row.action]: action, "max_tries": MAX_TRIES}
    elif row.action == "navigate":
        interaction = {"go_to_url": {"url": row.params.get("url")}}
    elif row.action == "go_back":
        interaction = {"go_back": {}}

    return _node(row, interaction)


def _instruction_for(row: TraceRow) -> str:
    """A one-line instruction an agent can follow for a row we could not pin down."""
    element = row.element
    named = ""
    if element:
        label = element.label(PARAMETER_NAME_ATTRIBUTES)
        named = f" labelled {label!r}" if label else ""
        named += f" ({element.tag_name})" if element.tag_name else ""

    if row.action == "input":
        return f"Type {str(row.params.get('text', ''))!r} into the field{named}."
    if row.action == "click":
        return f"Click the element{named}."
    if row.action == "select_dropdown":
        return f"Select {str(row.params.get('text', ''))!r} in the dropdown{named}."
    if row.action == "upload_file":
        return f"Upload the file to the input{named}."
    return f"Perform the recorded {row.action} step{named}."


def _agentic_node_for(row: TraceRow) -> dict[str, Any]:
    """A narrow agentic node for a row with no locator worth committing to.

    Scoped to one step and given a small step budget: this is the escape hatch
    for a single element, not a licence to re-explore the page. It is also the
    unit the loop retries — each round gives the row another chance to compile
    to something deterministic.
    """
    return _node(
        row,
        {
            "agentic_task": {
                "task": _instruction_for(row),
                "max_steps": AGENTIC_FALLBACK_MAX_STEPS,
                "backend": "browser_use",
            }
        },
    )


def compile_nodes(
    trace: Trace, parameters: dict[str, list[str]], already_used: set[str]
) -> list[dict[str, Any]]:
    """One node per compiled row, declaring its parameters into ``parameters``."""
    return [
        (
            _action_node_for(row, parameters, already_used)
            if row.classification == Classification.DETERMINISTIC
            else _agentic_node_for(row)
        )
        for row in trace.compiled_rows()
    ]


def trace_to_automation(trace: Trace, url: str | None = None) -> Automation:
    automation_url = url or trace.url
    if not automation_url:
        raise ValueError("no url: pass --url or capture a trace that records one")

    parameters: dict[str, list[str]] = {}
    already_used: set[str] = set()
    nodes = compile_nodes(trace, parameters, already_used)

    return Automation.model_validate(
        {
            "url": automation_url,
            "parameters": {
                "input_parameters": parameters,
                "generated_parameters": {},
            },
            "nodes": nodes,
        }
    )
