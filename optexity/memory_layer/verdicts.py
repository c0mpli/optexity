from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

LOCATOR_FIELDS = ("click_element", "input_text", "select_option", "upload_file")


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
    status: VerdictStatus  # verified | demoted | unmeasured | not_reached
    command: str | None = None
    reason: str = ""
    downloaded: str | None = None


class VerificationReport(BaseModel):
    url: str | None = None
    verdicts: list[NodeVerdict] = Field(default_factory=list)
    stopped_at: int | None = None
    stopped_because: str | None = None
    final_url: str | None = None
    signals: dict[str, Any] = Field(default_factory=dict)

    @property
    def verified_count(self) -> int:
        return sum(
            1 for verdict in self.verdicts if verdict.status == VerdictStatus.VERIFIED
        )

    @property
    def complete(self) -> bool:
        return self.stopped_at is None and bool(self.verdicts)


def locator_action(node):
    """The node's interaction that carries a command, if it has one."""
    for field in LOCATOR_FIELDS:
        action = getattr(node.interaction_action, field, None)
        if action is not None:
            return action
    return None


def apply_verdicts(automation, report: VerificationReport) -> None:
    """Copy measured commands onto the automation the caller keeps.

    The walk drives a throwaway copy — replace_variables consumes parameter
    placeholders in place — so measured commands have to be written back onto
    the pristine nodes explicitly.
    """
    # expected_downloads gates a wait loop in run_final_downloads_check, so
    # leaving it at 0 means a replay tears the browser down without waiting for
    # the file the walk just proved this automation produces.
    automation.expected_downloads = sum(
        1 for verdict in report.verdicts if verdict.downloaded
    )

    for verdict, node in zip(report.verdicts, automation.nodes, strict=False):
        action = locator_action(node)
        if verdict.command and action is not None:
            action.command = verdict.command
        # Only click carries these; a misaligned verdict must not raise here.
        click = getattr(node.interaction_action, "click_element", None)
        if verdict.downloaded and click is not None:
            click.expect_download = True
            click.download_filename = verdict.downloaded
