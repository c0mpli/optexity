from enum import StrEnum
from operator import attrgetter
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

    def label(self, attribute_names: tuple[str, ...]) -> str:
        """The most human-readable handle this element offers, else empty."""
        accessible_name = (self.accessible_name or "").strip()
        return accessible_name or next(
            (
                self.attributes[name]
                for name in attribute_names
                if self.attributes.get(name)
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
    def agentic_tokens(self) -> int | None:
        """What the agent spent working the objective out: the cost to beat.

        None when the capture recorded no usage, which is not the same as zero:
        an agentic run that reached the goal certainly spent tokens.
        """
        recorded = (self.usage or {}).get("total_tokens")
        return None if recorded is None else int(recorded)

    @property
    def agentic_seconds(self) -> float:
        """Only the steps' own durations, so it excludes browser startup."""
        return sum(row.step_duration_seconds or 0.0 for row in self.rows)

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


class VerdictStatus(StrEnum):
    """What the walk established about one node."""

    # locator measured at exactly one match, and the node had an observable effect
    VERIFIED = "verified"
    # the step still needs the agent: it compiled to an agentic node
    AGENTIC = "agentic"
    # nothing measurable identified the element, so the row lost its command
    DEMOTED = "demoted"
    # the node ran but showed no evidence it acted; nothing can be concluded
    UNMEASURED = "unmeasured"
    # the pass stopped earlier, so this row was never exercised
    NOT_REACHED = "not_reached"


class NodeVerdict(BaseModel):
    step: int
    action: str
    status: VerdictStatus
    command: str | None = None
    reason: str = ""
    downloaded: str | None = None
    navigated: bool = False


class RunSignals(BaseModel):
    """What the run produced besides the per-node verdicts."""

    # None when an agent ran but its usage was never recorded -- not zero.
    llm_tokens: int | None = 0
    downloaded_files: list[str] = Field(default_factory=list)
    output_data: list[dict[str, Any]] = Field(default_factory=list)


class VerificationReport(BaseModel):
    url: str | None = None
    verdicts: list[NodeVerdict] = Field(default_factory=list)
    stopped_at: int | None = None
    stopped_because: str | None = None
    final_url: str | None = None
    signals: RunSignals = Field(default_factory=RunSignals)

    @property
    def verified_count(self) -> int:
        return sum(
            1 for verdict in self.verdicts if verdict.status == VerdictStatus.VERIFIED
        )

    @property
    def complete(self) -> bool:
        return self.stopped_at is None and bool(self.verdicts)


class NodePatch(BaseModel):
    index: int
    prompt_instructions: str | None = None
    agentic_task: str | None = None


class AutomationPatch(BaseModel):
    """Everything the model is allowed to change. Notably absent: command — with
    nowhere in the patch to put one, the eval(f"page.{command}") surface in
    browser.py stays closed by the shape of the request, not by validation."""

    rename_parameters: dict[str, str] = Field(default_factory=dict)
    constant_parameters: list[str] = Field(default_factory=list)
    nodes: list[NodePatch] = Field(default_factory=list)


class RoundResult(BaseModel):
    number: int
    nodes: int
    verified: int
    agentic: int
    unresolved: int
    # None when the round drove an agent whose usage was never recorded.
    llm_tokens: int | None = 0
    seconds: float = 0.0


class LoopResult(BaseModel):
    rounds: list[RoundResult] = Field(default_factory=list)
    converged: bool = False
    stopped_because: str = ""
