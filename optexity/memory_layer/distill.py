import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

from optexity.memory_layer.candidates import MINIMUM_STABILITY_SCORE, build_candidates
from optexity.memory_layer.trace import (
    Classification,
    Trace,
    TraceRow,
    load_trace,
    placeholder,
)
from optexity.schema.automation import Automation

# Element actions and the InteractionAction field each compiles to. Everything
# else about them -- command, skip_prompt, max_tries -- is identical.
ACTION_FIELD = {
    "click": "click_element",
    "input": "input_text",
    "select_dropdown": "select_option",
    "upload_file": "upload_file",
}
ELEMENT_ACTIONS = set(ACTION_FIELD)
DETERMINISTIC_ACTIONS = ELEMENT_ACTIONS | {"navigate", "go_back"}

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

# Narrow enough to do one step, small enough that a wrong task string costs
# seconds rather than a re-exploration of the page.
AGENTIC_FALLBACK_MAX_STEPS = 3

# The parameter name is the automation's public API, so prefer what a human
# wrote; capped because a prose label makes a poor identifier.
PARAMETER_NAME_ATTRIBUTES = ("aria-label", "placeholder", "name", "id")
MAX_PARAMETER_NAME_CHARS = 40
CLASSIFICATION_MARKS = {
    Classification.DETERMINISTIC: "keep",
    Classification.REDUNDANT: "drop",
    Classification.NON_DETERMINISTIC: "llm",
}


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
    already_used.add(name)
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
    # Redacted at capture: declare the parameter, leave it empty.
    parameters[name] = [""] if redacted else [value]
    return placeholder(name)


def _mark(row: TraceRow, classification: Classification, reason: str) -> None:
    row.classification, row.reason = classification, reason


def classify(trace: Trace, automation_url: str | None) -> None:
    seen_first_action = False

    for row in trace.rows:
        if not row.executed:
            _mark(row, Classification.REDUNDANT, "action never executed")
            continue
        if row.error:
            _mark(row, Classification.REDUNDANT, f"action errored ({row.error[:120]})")
            continue
        if row.action == "scroll":
            _mark(
                row,
                Classification.REDUNDANT,
                "the command path already scrolls the element into view",
            )
            continue
        if row.action not in DETERMINISTIC_ACTIONS:
            if row.action in READ_ONLY_ACTIONS:
                _mark(
                    row,
                    Classification.REDUNDANT,
                    f"'{row.action}' only observes the page",
                )
            else:
                _mark(
                    row,
                    Classification.NON_DETERMINISTIC,
                    f"no equivalent for '{row.action}'",
                )
            continue
        if (
            row.action == "navigate"
            and not seen_first_action
            and automation_url
            and (row.params.get("url") or "").rstrip("/") == automation_url.rstrip("/")
        ):
            _mark(
                row,
                Classification.REDUNDANT,
                "the run already begins on the automation url",
            )
            continue

        if row.action in ELEMENT_ACTIONS:
            if row.element is None:
                _mark(
                    row,
                    Classification.NON_DETERMINISTIC,
                    "no element was recorded for this action",
                )
                continue
            if row.element.is_in_subframe:
                _mark(
                    row,
                    Classification.NON_DETERMINISTIC,
                    f"element is inside frame {row.element.frame_id}",
                )
                continue

            row.candidates = build_candidates(row.element)
            best = row.best_candidate
            if best is None or best.stability_score < MINIMUM_STABILITY_SCORE:
                row.classification = Classification.NON_DETERMINISTIC
                row.reason = (
                    f"best locator scores {best.stability_score if best else 0} "
                    f"< {MINIMUM_STABILITY_SCORE}"
                )
                continue
            matches = "unprobed" if best.is_unprobed else best.match_count
            row.reason = (
                f"locator {best.kind} score={best.stability_score} matches={matches}"
            )

        row.classification = Classification.DETERMINISTIC
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
                _mark(
                    row,
                    Classification.REDUNDANT,
                    f"folded into step {previous_kept.step} as press_enter",
                )
                continue
        if row.classification == Classification.DETERMINISTIC:
            previous_kept = row


def _mark_superseded_interactions(trace: Trace) -> None:
    """Of a run of identical consecutive actions, only the last had any effect.

    fill replaces contents; navigating to one url twice is idempotent. Element
    identity comes from element_hash, which cannot find an element on a page but
    can tell two recorded rows apart."""
    deterministic = trace.rows_with(Classification.DETERMINISTIC)
    for row, following in zip(deterministic, deterministic[1:], strict=False):
        if row.action != following.action:
            continue
        if row.action == "navigate":
            # Loading one url twice running is idempotent, so only the last of
            # a run has effect. An agent that loses its way emits long runs.
            if row.params.get("url") != following.params.get("url"):
                continue
        elif row.action in {"input", "click"}:
            if row.element is None:
                continue
            if not row.element.is_same_element_as(following.element):
                continue
            if row.params.get("press_enter"):
                continue
        else:
            continue
        row.classification = Classification.REDUNDANT
        row.reason = f"superseded by the same {row.action} at step {following.step}"


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


def _describe(row: TraceRow) -> str:
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
                "task": _describe(row),
                "max_steps": AGENTIC_FALLBACK_MAX_STEPS,
                "backend": "browser_use",
            }
        },
    )


def trace_to_automation(trace: Trace, url: str | None = None) -> Automation:
    automation_url = url or trace.url
    if not automation_url:
        raise ValueError("no url: pass --url or capture a trace that records one")

    parameters: dict[str, list[str]] = {}
    already_used: set[str] = set()
    nodes = [
        (
            _action_node_for(row, parameters, already_used)
            if row.classification == Classification.DETERMINISTIC
            else _agentic_node_for(row)
        )
        for row in trace.compiled_rows()
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


def _print_summary(
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m optexity.memory_layer.distill",
        description="Compile a captured agentic run into a deterministic Automation.",
    )
    parser.add_argument("history", type=Path, help="path to agent_history.json")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--url", help="automation url (defaults to the recorded one)")
    parser.add_argument(
        "--trace-out", type=Path, help="also write the labelled trace here"
    )
    args = parser.parse_args(argv)

    logging.getLogger("optexity").setLevel(logging.INFO)
    if not args.history.exists():
        parser.error(f"no such file: {args.history}")

    automation, trace = distill(args.history, args.url)
    args.out.write_text(automation.model_dump_json(indent=2, exclude_none=True))
    if args.trace_out:
        args.trace_out.write_text(trace.model_dump_json(indent=2))

    _print_summary(trace, automation, args.history, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
