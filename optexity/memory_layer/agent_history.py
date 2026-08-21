import json
from pathlib import Path
from typing import Any

from optexity.memory_layer.trace import Element, Trace, TraceRow


def element_from_record(interacted_element: dict[str, Any]) -> Element:
    return Element(
        tag_name=(interacted_element.get("node_name") or "").lower(),
        attributes=interacted_element.get("attributes") or {},
        xpath=interacted_element.get("x_path") or "",
        accessible_name=interacted_element.get("ax_name") or None,
        element_hash=interacted_element.get("element_hash"),
        stable_hash=interacted_element.get("stable_hash"),
        frame_id=interacted_element.get("frame_id"),
    )


def rows_from_history(history: list[dict[str, Any]]) -> list[TraceRow]:
    rows: list[TraceRow] = []
    url_seen_at_step = [(entry.get("state") or {}).get("url") for entry in history]

    for step_position, step_entry in enumerate(history):
        actions = (step_entry.get("model_output") or {}).get("action") or []
        if not actions:
            continue

        results = step_entry.get("result") or []
        interacted_elements = (step_entry.get("state") or {}).get(
            "interacted_element"
        ) or []
        metadata = step_entry.get("metadata") or {}
        next_position = step_position + 1

        started_at = metadata.get("step_start_time")
        ended_at = metadata.get("step_end_time")

        for action_index, action in enumerate(actions):
            if not action:
                continue
            action_name = next(iter(action))
            # multi_act stops early on a done or errored action, so results can
            # be shorter than actions and zipping would misattribute outcomes.
            result = results[action_index] if action_index < len(results) else None
            interacted_element = (
                interacted_elements[action_index]
                if action_index < len(interacted_elements)
                else None
            )

            rows.append(
                TraceRow(
                    step=metadata.get("step_number", next_position),
                    action_index=action_index,
                    action=action_name,
                    params=action.get(action_name) or {},
                    executed=result is not None,
                    error=(result or {}).get("error"),
                    url_before=url_seen_at_step[step_position],
                    url_after=(
                        url_seen_at_step[next_position]
                        if next_position < len(url_seen_at_step)
                        else url_seen_at_step[step_position]
                    ),
                    step_duration_seconds=(
                        round(ended_at - started_at, 3)
                        if started_at and ended_at
                        else None
                    ),
                    element=(
                        element_from_record(interacted_element)
                        if interacted_element
                        else None
                    ),
                )
            )
    return rows


def load_trace(history_path: str | Path) -> Trace:
    history_record = json.loads(Path(history_path).read_text())
    rows = rows_from_history(history_record.get("history") or [])
    return Trace(
        url=next((row.url_before for row in rows if row.url_before), None),
        usage=history_record.get("usage") or {},
        rows=rows,
    )
