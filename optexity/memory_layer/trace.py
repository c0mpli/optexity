import json
from enum import StrEnum
from operator import attrgetter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Classification(StrEnum):
    """What the distiller concluded about a recorded action."""

    # compiles to a command node
    DETERMINISTIC = "deterministic"
    # had no effect worth reproducing
    REDUNDANT = "redundant"
    # real work, but nothing identifies the element well enough to commit to
    NON_DETERMINISTIC = "non_deterministic"


by_stability = attrgetter("stability_score")


def placeholder(parameter_name: str) -> str:
    """How the runtime's replace_variables spells a parameter reference."""
    return f"{{{parameter_name}[0]}}"


class Element(BaseModel):
    tag_name: str
    attributes: dict[str, str] = Field(default_factory=dict)
    xpath: str = ""
    accessible_name: str | None = None
    element_hash: int | None = None
    stable_hash: int | None = None
    frame_id: str | None = None

    def label(self, attributes: tuple[str, ...]) -> str:
        """The most human-readable handle this element offers, else empty."""
        accessible_name = (self.accessible_name or "").strip()
        return accessible_name or next(
            (
                self.attributes[attribute]
                for attribute in attributes
                if self.attributes.get(attribute)
            ),
            "",
        )

    @property
    def is_in_subframe(self) -> bool:
        return self.frame_id is not None

    def is_same_element_as(self, other: "Element | None") -> bool:
        """Hashes cannot locate an element, but they do tell two rows apart."""
        if other is None or self.frame_id != other.frame_id:
            return False
        if self.stable_hash is not None and other.stable_hash is not None:
            return self.stable_hash == other.stable_hash
        if self.element_hash is not None and other.element_hash is not None:
            return self.element_hash == other.element_hash
        return bool(self.xpath) and self.xpath == other.xpath

    @classmethod
    def from_interacted_element(cls, interacted_element: dict[str, Any]) -> "Element":
        return cls(
            tag_name=(interacted_element.get("node_name") or "").lower(),
            attributes=interacted_element.get("attributes") or {},
            xpath=interacted_element.get("x_path") or "",
            accessible_name=interacted_element.get("ax_name") or None,
            element_hash=interacted_element.get("element_hash"),
            stable_hash=interacted_element.get("stable_hash"),
            frame_id=interacted_element.get("frame_id"),
        )


class Candidate(BaseModel):
    command: str
    kind: str
    stability_score: int
    match_count: int | None = None

    @property
    def matches_exactly_one_element(self) -> bool:
        return self.match_count == 1

    @property
    def is_unprobed(self) -> bool:
        return self.match_count is None


def unique_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Probed at exactly one match, most stable first."""
    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.matches_exactly_one_element
        ),
        key=by_stability,
        reverse=True,
    )


class TraceRow(BaseModel):
    step: int
    action_index: int
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    executed: bool
    error: str | None = None
    url_before: str | None = None
    url_after: str | None = None
    step_duration_seconds: float | None = None
    element: Element | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    classification: Classification | None = None
    reason: str | None = None

    @property
    def changed_url(self) -> bool:
        return bool(self.url_after and self.url_before != self.url_after)

    @property
    def best_candidate(self) -> Candidate | None:
        unique = unique_candidates(self.candidates)
        if unique:
            return unique[0]
        unprobed = [candidate for candidate in self.candidates if candidate.is_unprobed]
        return max(unprobed, key=by_stability) if unprobed else None


class Trace(BaseModel):
    url: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    rows: list[TraceRow] = Field(default_factory=list)

    def rows_with(self, *classifications: Classification) -> list[TraceRow]:
        return [row for row in self.rows if row.classification in classifications]

    def compiled_rows(self) -> list[TraceRow]:
        """Every row that becomes a node, in order.

        Includes the rows we refused to make deterministic: they compile to a
        narrow agentic node rather than vanishing, so the node list stays a
        complete path. A hole would leave a replay silently skipping the step
        and strand every node after it on the wrong page.
        """
        return self.rows_with(
            Classification.DETERMINISTIC, Classification.NON_DETERMINISTIC
        )

    def counts_by_classification(self) -> dict[str, int]:
        counts = {"total": len(self.rows)}
        for row in self.rows:
            key = row.classification or "unclassified"
            counts[key] = counts.get(key, 0) + 1
        return counts


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
                        Element.from_interacted_element(interacted_element)
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
