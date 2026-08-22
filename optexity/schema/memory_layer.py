from operator import attrgetter
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

    def deterministic_rows(self) -> list[TraceRow]:
        return [row for row in self.rows if row.classification == "deterministic"]

    def counts_by_classification(self) -> dict[str, int]:
        counts = {"total": len(self.rows)}
        for row in self.rows:
            key = row.classification or "unclassified"
            counts[key] = counts.get(key, 0) + 1
        return counts


class CapturedUsage(BaseModel):
    """browser-use's own accounting block, as save_history writes it.

    Not optexity's TokenUsage: that counts different things under different
    names. agentic_nodes is this layer's own addition when several are summed.
    """

    agentic_nodes: int = 0
    entry_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
