import re
from pathlib import Path
from typing import Any

from optexity.memory_layer.agent_history import load_trace
from optexity.memory_layer.distill.candidates import (
    MINIMUM_STABILITY_SCORE,
    build_candidates,
)
from optexity.schema.automation import Automation
from optexity.schema.memory_layer import Trace, TraceRow

DETERMINISTIC_ACTIONS = {
    "click",
    "input",
    "navigate",
    "go_back",
    "select_dropdown",
    "upload_file",
}
ELEMENT_ACTIONS = {"click", "input", "select_dropdown", "upload_file"}

READ_ONLY_ACTIONS = {
    "screenshot",
    "evaluate",
    "done",
    "wait",
    "find_text",
    "dropdown_options",
    "read_file",
}

# browser-use emits a submit as a separate send_keys, so typing then Enter would
# distil to typing only. key_press cannot express it: handle_keypress takes
# Enter/Tab/Space, KEY_NAMES has none.
ENTER_KEYS = {"Enter", "Return", "\n"}

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
CLASSIFICATION_MARKS = {
    "deterministic": "keep",
    "redundant": "drop",
    "non_deterministic": "llm",
}


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


def classify(trace: Trace, automation_url: str | None) -> None:
    seen_first_action = False

    for row in trace.rows:
        if not row.executed:
            row.classification = "redundant"
            row.reason = "action never executed"
            continue
        if row.error:
            row.classification = "redundant"
            row.reason = f"action errored ({row.error[:120]})"
            continue
        if row.action == "scroll":
            row.classification = "redundant"
            row.reason = "the command path already scrolls the element into view"
            continue
        if row.action not in DETERMINISTIC_ACTIONS:
            read_only = row.action in READ_ONLY_ACTIONS
            row.classification = "redundant" if read_only else "non_deterministic"
            row.reason = (
                f"'{row.action}' only observes the page"
                if read_only
                else f"no deterministic equivalent for '{row.action}'"
            )
            continue
        if (
            row.action == "navigate"
            and not seen_first_action
            and automation_url
            and (row.params.get("url") or "").rstrip("/") == automation_url.rstrip("/")
        ):
            row.classification = "redundant"
            row.reason = "the run already begins on the automation url"
            continue

        if row.action in ELEMENT_ACTIONS:
            if row.element is None:
                row.classification = "non_deterministic"
                row.reason = "no element was recorded for this action"
                continue
            if row.element.is_in_subframe:
                row.classification = "non_deterministic"
                row.reason = f"element is inside frame {row.element.frame_id}"
                continue

            row.candidates = build_candidates(row.element)
            best = row.best_candidate
            if best is None or best.stability_score < MINIMUM_STABILITY_SCORE:
                row.classification = "non_deterministic"
                row.reason = (
                    f"best locator scores {best.stability_score if best else 0} "
                    f"< {MINIMUM_STABILITY_SCORE}"
                )
                continue
            matches = "unprobed" if best.is_unprobed else best.match_count
            row.reason = (
                f"locator {best.kind} score={best.stability_score} matches={matches}"
            )

        row.classification = "deterministic"
        row.reason = row.reason or f"deterministic {row.action}"
        seen_first_action = True

    _fold_enter_into_preceding_input(trace)
    _mark_superseded_interactions(trace)


def _fold_enter_into_preceding_input(trace: Trace) -> None:
    """Left alone, a search that types then submits distils to one that types."""
    previous_kept: TraceRow | None = None
    for row in trace.rows:
        if row.action == "send_keys" and previous_kept is not None:
            keys = str(row.params.get("keys", ""))
            if keys in ENTER_KEYS and previous_kept.action == "input":
                previous_kept.params["press_enter"] = True
                row.classification = "redundant"
                row.reason = f"folded into step {previous_kept.step} as press_enter"
                continue
        if row.classification == "deterministic":
            previous_kept = row


def _mark_superseded_interactions(trace: Trace) -> None:
    """Of a run of identical consecutive actions, only the last had any effect.

    fill replaces contents; navigating to one url twice is idempotent. Clicks are
    neither -- two on a checkbox cancel out, two on a stepper count twice -- so a
    repeated click is left alone. Element identity comes from element_hash, which
    cannot find an element on a page but can tell two recorded rows apart."""
    deterministic = trace.deterministic_rows()
    for row, following in zip(deterministic, deterministic[1:], strict=False):
        if row.action != following.action:
            continue
        if row.action == "navigate":
            if row.params.get("url") != following.params.get("url"):
                continue
        elif row.action == "input":
            if row.element is None:
                continue
            if not row.element.is_same_element_as(following.element):
                continue
            if row.params.get("press_enter"):
                continue
        else:
            continue
        row.classification = "redundant"
        row.reason = f"superseded by the same {row.action} at step {following.step}"


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


def distill(history_path: Path, url: str | None = None) -> tuple[Automation, Trace]:
    trace = load_trace(history_path)
    classify(trace, url or trace.url)
    return trace_to_automation(trace, url), trace


def print_summary(
    trace: Trace, automation: Automation, history_path: Path, out_path: Path
) -> None:
    counts = trace.counts_by_classification()
    print(f"\n{history_path}  ->  {out_path}")
    for label in ("total", "deterministic", "redundant", "non_deterministic"):
        print(f"  {label:<18}: {counts.get(label, 0)}")
    print(f"  {'parameters':<18}: {list(automation.parameters.input_parameters)}\n")
    for row in trace.rows:
        mark = CLASSIFICATION_MARKS.get(row.classification, "????")
        print(f"  [{mark:>4}] {row.action:<16} {row.reason}")
    print()
