import asyncio
import logging
import time
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from optexity.memory_layer.candidates import MINIMUM_STABILITY_SCORE, propose_bundle
from optexity.memory_layer.trace import (
    Classification,
    Trace,
    TraceRow,
    by_stability,
    unique_candidates,
)

logger = logging.getLogger(__name__)

# count() does not auto-wait, so a short timeout reports zero for anything that
# attaches a moment late; a miss costs the full timeout.
PROBE_TIMEOUT_SECONDS = 5.0
PROBE_BUDGET_SECONDS = 20.0
VERIFIED_CANDIDATES_WANTED = 2
LOCATOR_FIELDS = ("click_element", "input_text", "select_option", "upload_file")
# A recorded navigation is waited out in full; one the recording did not capture
# still gets a grace window, since a click that navigates unexpectedly would
# otherwise be judged inert.
NAVIGATION_TIMEOUT_SECONDS = 10.0
NAVIGATION_GRACE_SECONDS = 2.0
NAVIGATING_ACTIONS = {"navigate", "go_back", "click"}
DOWNLOAD_SETTLE_SECONDS = 10.0


def downloads_present(browser) -> set[str]:
    directory = Path(browser.temp_downloads_dir)
    return (
        {entry.name for entry in directory.iterdir()} if directory.is_dir() else set()
    )


async def wait_for_download(browser, before: set[str]) -> str | None:
    """The file a node produced, if it produced one.

    Chrome's download path is set browser-wide, so a file arrives whether or not
    the node declared expect_download -- which is what lets this be measured
    rather than guessed from the href.
    """
    if not downloads_present(browser) - before:
        return None

    deadline = time.monotonic() + DOWNLOAD_SETTLE_SECONDS
    while time.monotonic() < deadline:
        arrived = downloads_present(browser) - before
        done = sorted(n for n in arrived if not n.endswith((".crdownload", ".tmp")))
        if done:
            return done[0]
        await asyncio.sleep(0.1)
    return None


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
    probes_run: int = 0
    probe_seconds: float = 0.0


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


def same_page(left: str | None, right: str | None) -> bool:
    """Session ids live in the query string, so raw urls fail runs that are fine.
    Path is compared exactly: a login-error page and a dashboard differ there."""
    if not left or not right:
        return False
    first, second = urlsplit(left), urlsplit(right)
    return (first.netloc, first.path.rstrip("/")) == (
        second.netloc,
        second.path.rstrip("/"),
    )


async def probe(command: str, browser) -> int | None:
    """None and zero differ: an unparseable command tells us nothing."""
    from optexity.inference.core.run_automation import count_locator_matches

    try:
        return await count_locator_matches(command, PROBE_TIMEOUT_SECONDS, browser)
    except Exception as e:
        # A malformed command raises in eval, before count_locator_matches.
        if "TargetClosed" in type(e).__name__:
            raise
        logger.debug(f"probe failed for {command!r}: {e}")
        return None


async def probe_row(row: TraceRow, browser) -> tuple[int, float]:
    """Bounded twice: every probe costs a second of an authenticated session."""
    started = time.monotonic()
    probes_run = 0
    verified = 0

    for candidate in sorted(row.candidates, key=by_stability, reverse=True):
        if verified >= VERIFIED_CANDIDATES_WANTED:
            break
        if time.monotonic() - started > PROBE_BUDGET_SECONDS:
            break

        candidate.match_count = await probe(candidate.command, browser)
        probes_run += 1
        if candidate.matches_exactly_one_element:
            verified += 1

    return probes_run, round(time.monotonic() - started, 2)


async def choose_command(row: TraceRow, browser) -> tuple[str | None, str]:
    unique = unique_candidates(row.candidates)
    if not unique:
        measured = sum(
            1 for candidate in row.candidates if candidate.match_count is not None
        )
        if not measured:
            return None, "no candidate could be measured"
        return None, f"no candidate matched exactly one element ({measured} probed)"

    best = unique[0]
    # Probing can demote the top candidate and leave only a far weaker one
    # unique; shipping that is the guess this layer exists to avoid.
    if best.stability_score < MINIMUM_STABILITY_SCORE:
        return None, (
            f"no verified candidate above threshold "
            f"(best verified: {best.kind} score={best.stability_score})"
        )

    bundle = propose_bundle(row.candidates)
    if bundle is not None:
        # Only the assembled expression can settle whether the union is unique.
        bundle.match_count = await probe(bundle.command, browser)
        row.candidates.append(bundle)
        if bundle.matches_exactly_one_element:
            return bundle.command, (
                f"{bundle.kind}, both measured and the pair matched 1"
            )
        logger.debug(
            f"bundle matched {bundle.match_count}; shipping the single best locator"
        )

    return best.command, f"{best.kind} score={best.stability_score}, matched 1"


async def wait_for_navigation(browser, from_url: str | None, timeout: float) -> None:
    """click_locator runs with no_wait_after and sleep_for_page_to_load's
    wait_for_load_state returns at once on the already-loaded old page, so a
    navigation the node triggered is still in flight when it returns.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not same_page(await browser.get_current_page_url(), from_url):
            return
        await asyncio.sleep(0.1)


async def acted(
    row: TraceRow, node, browser, before_url: str | None
) -> tuple[bool, str]:
    """Whether the node that just ran actually changed anything.

    Deliberately not keyed on ``memory.browser_states[-1].locator_candidates``:
    that snapshot is written before the action runs, so it proves the command
    resolved a visible element and nothing about whether the click landed. A
    resolved-but-inert node is precisely the failure this check exists to catch,
    so the evidence has to be an observed effect.
    """
    after_url = await browser.get_current_page_url()
    navigated = not same_page(before_url, after_url)

    if row.action in NAVIGATING_ACTIONS:
        if navigated:
            return True, f"navigated to {after_url}"
        if row.action == "click" and not row.changed_url:
            # The recording says this click stayed put, so no navigation is the
            # expected outcome and there is nothing cheap left to observe.
            return True, "clicked; recording shows no navigation for this step"
        return False, f"still on {after_url}"

    if row.action == "input":
        interaction = node.interaction_action
        action = getattr(interaction, "input_text", None)
        command = getattr(action, "command", None)
        if not command:
            return False, "no command to read back"
        try:
            locator = await browser.get_locator_from_command(command)
            value = await locator.input_value()
        except Exception as e:
            return False, f"could not read the field back ({e})"
        expected = str(getattr(action, "input_text", ""))
        if value == expected:
            return True, "field holds the value written"
        return False, f"field holds {value!r}, expected {expected!r}"

    return True, "no effect check for this action"


def locator_action(node):
    """The node's interaction that carries a command, if it has one."""
    interaction = node.interaction_action
    if interaction is None:
        return None
    for field in LOCATOR_FIELDS:
        action = getattr(interaction, field, None)
        if action is not None:
            return action
    return None


def _set_command(node, command: str) -> bool:
    action = locator_action(node)
    if action is None:
        return False
    action.command = command
    return True


def apply_verdicts(automation, report: VerificationReport) -> int:
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

    applied = 0
    for verdict, node in zip(report.verdicts, automation.nodes, strict=False):
        if verdict.command and _set_command(node, verdict.command):
            applied += 1
        # Only click carries these; a misaligned verdict must not raise here.
        click = getattr(node.interaction_action, "click_element", None)
        if verdict.downloaded and click is not None:
            click.expect_download = True
            click.download_filename = verdict.downloaded
    return applied


async def verify_walk(
    trace: Trace, automation, task, memory, browser
) -> VerificationReport:
    """Walk the compiled nodes, measuring each locator against the live page.

    Takes the trace and its compiled automation as a matched pair so the two
    cannot drift: recompiling mid-walk would discard every measurement, and
    re-deriving the pairing after a row is demoted would silently reindex it.
    """
    from optexity.inference.core.run_automation import run_action_node

    report = VerificationReport(url=automation.url)
    rows = trace.compiled_rows()
    if len(rows) != len(automation.nodes):
        report.stopped_at = 0
        report.stopped_because = (
            f"trace and automation disagree: {len(rows)} rows, "
            f"{len(automation.nodes)} nodes"
        )
        return report

    for position, (row, node) in enumerate(zip(rows, automation.nodes, strict=True)):
        verdict = NodeVerdict(
            step=row.step, action=row.action, status=VerdictStatus.NOT_REACHED
        )

        # Mirror run_action_node's own preamble before probing, so the probe and
        # the execution address the same document: handle_new_tabs switches to
        # the newest tab whenever the page count grew.
        await asyncio.sleep(node.before_sleep_time)
        await browser.handle_new_tabs(0)

        live_url = await browser.get_current_page_url()

        agentic = getattr(node.interaction_action, "agentic_task", None) is not None

        if locator_action(node) is not None:
            verdict.probes_run, verdict.probe_seconds = await probe_row(row, browser)
            command, reason = await choose_command(row, browser)
            if command is None or not _set_command(node, command):
                # Distillation deletes the agent's detours, so reaching an
                # element from a different page than the recording is normal.
                if row.url_before and not same_page(live_url, row.url_before):
                    reason = f"{reason}; on {live_url}, recorded at {row.url_before}"
                verdict.status = VerdictStatus.DEMOTED
                row.classification = Classification.NON_DETERMINISTIC
                verdict.reason = row.reason = reason
                report.verdicts.append(verdict)
                report.stopped_at = position
                report.stopped_because = (
                    "no measured locator for this row, so the path breaks here"
                )
                break
            verdict.reason = reason
            verdict.command = command

        before_url = await browser.get_current_page_url()
        downloads_before = downloads_present(browser)
        await run_action_node(node, task, memory, browser)
        if row.action in NAVIGATING_ACTIONS:
            await wait_for_navigation(
                browser,
                before_url,
                (
                    NAVIGATION_TIMEOUT_SECONDS
                    if row.changed_url
                    else NAVIGATION_GRACE_SECONDS
                ),
            )

        if row.action == "click":
            verdict.downloaded = await wait_for_download(browser, downloads_before)

        effective, evidence = await acted(row, node, browser, before_url)
        if not effective:
            verdict.status = VerdictStatus.UNMEASURED
            verdict.reason = f"{verdict.reason}; no observable effect ({evidence})"
            report.verdicts.append(verdict)
            report.stopped_at = position
            report.stopped_because = "node executed without evidence it acted"
            break

        # An agentic node working proves the agent can still do the step, not
        # that anything was made deterministic. Calling it verified would let a
        # run of pure LLM steps read as a fully compiled automation.
        verdict.status = VerdictStatus.AGENTIC if agentic else VerdictStatus.VERIFIED
        if verdict.downloaded:
            evidence = f"{evidence}; downloaded {verdict.downloaded}"
        verdict.reason = f"{verdict.reason}; {evidence}"
        report.verdicts.append(verdict)

    for row in rows[len(report.verdicts) :]:
        report.verdicts.append(
            NodeVerdict(
                step=row.step,
                action=row.action,
                status=VerdictStatus.NOT_REACHED,
                reason="pass stopped before this row",
            )
        )

    report.final_url = await browser.get_current_page_url()
    report.signals = {
        "nodes": len(automation.nodes),
        "verified": report.verified_count,
        "llm_tokens": memory.token_usage.total_tokens,
        "downloaded_files": [
            verdict.downloaded for verdict in report.verdicts if verdict.downloaded
        ],
        "output_data": [
            entry.model_dump(mode="json") for entry in memory.variables.output_data
        ],
    }
    return report
