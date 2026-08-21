import json
from operator import attrgetter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Classification = Literal["deterministic", "redundant", "non_deterministic"]
by_stability = attrgetter("stability_score")


class Element(BaseModel):
    tag_name: str
    attributes: dict[str, str] = Field(default_factory=dict)
    xpath: str = ""
    accessible_name: str | None = None
    element_hash: int | None = None
    stable_hash: int | None = None
    frame_id: str | None = None

    @property
    def is_in_subframe(self) -> bool:
        return self.frame_id is not None

    def is_same_element_as(self, other: "Element | None") -> bool:
        """Hashes cannot locate an element, but they do tell two rows apart."""
        if other is None or self.frame_id != other.frame_id:
            return False
        for hash_attribute in ("stable_hash", "element_hash"):
            mine = getattr(self, hash_attribute)
            theirs = getattr(other, hash_attribute)
            if mine is not None and theirs is not None:
                return mine == theirs
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


def verified_candidates(candidates: list[Candidate]) -> list[Candidate]:
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
        verified = verified_candidates(self.candidates)
        if verified:
            return verified[0]
        unprobed = [candidate for candidate in self.candidates if candidate.is_unprobed]
        return max(unprobed, key=by_stability) if unprobed else None


class Trace(BaseModel):
    url: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    rows: list[TraceRow] = Field(default_factory=list)

    @property
    def agentic_tokens(self) -> int:
        """What the agent spent working the objective out: the cost to beat."""
        return int(self.usage.get("total_tokens") or 0)

    @property
    def agentic_seconds(self) -> float:
        """Only the steps' own durations, so it excludes browser startup."""
        return sum(row.step_duration_seconds or 0.0 for row in self.rows)

    def deterministic_rows(self) -> list[TraceRow]:
        return [row for row in self.rows if row.classification == "deterministic"]

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
