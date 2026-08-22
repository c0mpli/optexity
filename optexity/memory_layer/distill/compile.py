import re
from typing import Any

from optexity.memory_layer.distill.classify import ELEMENT_ACTIONS
from optexity.schema.automation import Automation
from optexity.schema.memory_layer import Trace, TraceRow

# save_history redacts secure parameters to <secret>KEY</secret>
# (browser-use agent/views.py:330); recording that would bake in a placeholder.
SECRET_PLACEHOLDER = re.compile(r"^<secret>([^<>]+)</secret>$")

# Runtime defaults are tuned for LLM-resolved nodes: 10 tries floors every miss
# at 10s, and end_sleep_time's 5.0 dominates a distilled run.
MAX_TRIES = 2
SLEEP_AFTER_NAVIGATION = 3.0
SLEEP_AFTER_INTERACTION = 0.5

# The parameter name is the automation's public API, so prefer what a human
# wrote; capped because a prose label makes a poor identifier.
PARAMETER_NAME_ATTRIBUTES = ("aria-label", "placeholder", "name", "id")
MAX_PARAMETER_NAME_CHARS = 40


def _parameter_name_for(row: TraceRow, already_used: set[str]) -> str:
    name_sources = []
    if row.element:
        accessible_name = (row.element.accessible_name or "").strip()
        if 0 < len(accessible_name) <= MAX_PARAMETER_NAME_CHARS:
            name_sources.append(accessible_name)
        name_sources += [
            row.element.attributes[attribute]
            for attribute in PARAMETER_NAME_ATTRIBUTES
            if row.element.attributes.get(attribute)
        ]

    # replace_variables needs a valid identifier, and form fields are often
    # ordered: 04fullname -> fullname.
    slug = ""
    for source in name_sources:
        slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
        slug = re.sub(r"^\d+_?", "", slug)
        if slug:
            break
    slug = slug or "input"

    name, suffix = slug, 2
    while name in already_used:
        name, suffix = f"{slug}_{suffix}", suffix + 1
    already_used.add(name)
    return name


def _action_node_for(
    row: TraceRow, parameters: dict[str, list[str]], already_used: set[str]
) -> dict[str, Any]:
    interaction: dict[str, Any] = {}
    best = row.best_candidate
    # Single locator only: an or_() bundle is unsafe until measured, which this
    # pure compiler cannot do. verify.choose_command adds one when it can.
    command = best.command if best else None

    if row.action == "input":
        recorded_text = str(row.params.get("text", ""))
        redacted = SECRET_PLACEHOLDER.match(recorded_text)
        parameter_name = (
            redacted.group(1) if redacted else _parameter_name_for(row, already_used)
        )
        if redacted:
            # Redacted at capture: declare the parameter, leave it empty.
            already_used.add(parameter_name)
            parameters[parameter_name] = [""]
        else:
            parameters[parameter_name] = [recorded_text]
        interaction["input_text"] = {
            "command": command,
            "input_text": f"{{{parameter_name}[0]}}",
            "press_enter": bool(row.params.get("press_enter")),
            "skip_prompt": True,
        }
    elif row.action == "click":
        interaction["click_element"] = {"command": command, "skip_prompt": True}
    elif row.action == "select_dropdown":
        interaction["select_option"] = {
            "command": command,
            "select_values": [str(row.params.get("text", ""))],
            "skip_prompt": True,
        }
    elif row.action == "upload_file":
        # A recorded path belongs to the capturing machine, so it is a parameter.
        parameter_name = _parameter_name_for(row, already_used)
        parameters[parameter_name] = [str(row.params.get("path", ""))]
        interaction["upload_file"] = {
            "command": command,
            "file_path": f"{{{parameter_name}[0]}}",
            "skip_prompt": True,
        }
    elif row.action == "navigate":
        interaction["go_to_url"] = {"url": row.params.get("url")}
    elif row.action == "go_back":
        interaction["go_back"] = {}

    if row.action in ELEMENT_ACTIONS:
        interaction["max_tries"] = MAX_TRIES

    return {
        "type": "action_node",
        "interaction_action": interaction,
        "end_sleep_time": (
            SLEEP_AFTER_NAVIGATION if row.changed_url else SLEEP_AFTER_INTERACTION
        ),
    }


def trace_to_automation(trace: Trace, url: str | None = None) -> Automation:
    automation_url = url or trace.url
    if not automation_url:
        raise ValueError("no url: pass --url or capture a trace that records one")

    parameters: dict[str, list[str]] = {}
    already_used: set[str] = set()
    nodes = [
        _action_node_for(row, parameters, already_used)
        for row in trace.deterministic_rows()
    ]

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
