import asyncio
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from optexity.memory_layer.candidates import MINIMUM_STABILITY_SCORE, propose_bundle
from optexity.memory_layer.capture import total_llm_tokens
from optexity.memory_layer.trace import (
    Trace,
    TraceRow,
    by_stability,
    verified_candidates,
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


def files_in_downloads_dir(browser) -> set[str]:
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
    if not files_in_downloads_dir(browser) - before:
        return None
    deadline = time.monotonic() + DOWNLOAD_SETTLE_SECONDS
    while time.monotonic() < deadline:
        arrived = files_in_downloads_dir(browser) - before
        finished = [
            name for name in arrived if not name.endswith((".crdownload", ".tmp"))
        ]
        if finished:
            return sorted(finished)[0]
        await asyncio.sleep(0.1)
    return None


class NodeVerdict(BaseModel):
    step: int
    action: str
    status: str  # verified | demoted | unmeasured | not_reached
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
        return sum(1 for verdict in self.verdicts if verdict.status == "verified")

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


async def probe_row(row: TraceRow, browser) -> None:
    """Bounded twice: every probe costs a second of an authenticated session."""
    started = time.monotonic()
    verified = 0

    for candidate in sorted(row.candidates, key=by_stability, reverse=True):
        if verified >= VERIFIED_CANDIDATES_WANTED:
            break
        if time.monotonic() - started > PROBE_BUDGET_SECONDS:
            break
        candidate.match_count = await probe(candidate.command, browser)
        if candidate.matches_exactly_one_element:
            verified += 1


async def choose_command(row: TraceRow, browser) -> tuple[str | None, str]:
    verified = verified_candidates(row.candidates)
    if not verified:
        measured = sum(
            1 for candidate in row.candidates if candidate.match_count is not None
        )
        if not measured:
            return None, "no candidate could be measured"
        return None, f"no candidate matched exactly one element ({measured} probed)"

    best = verified[0]
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


async def observe_effect(
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
        action = getattr(node.interaction_action, "input_text", None)
        command = action.command if action else None
        if not command:
            return False, "no command to read back"
        try:
            locator = await browser.get_locator_from_command(command)
            value = await locator.input_value()
        except Exception as e:
            return False, f"could not read the field back ({e})"
        expected = str(action.input_text)
        if value == expected:
            return True, "field holds the value written"
        return False, f"field holds {value!r}, expected {expected!r}"

    return True, "no effect check for this action"


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
    rows = trace.deterministic_rows()
    if len(rows) != len(automation.nodes):
        report.stopped_at = 0
        report.stopped_because = (
            f"trace and automation disagree: {len(rows)} rows, "
            f"{len(automation.nodes)} nodes"
        )
        return report

    for position, (row, node) in enumerate(zip(rows, automation.nodes, strict=True)):
        verdict = NodeVerdict(step=row.step, action=row.action, status="not_reached")

        # Mirror run_action_node's own preamble before probing, so the probe and
        # the execution address the same document: handle_new_tabs switches to
        # the newest tab whenever the page count grew.
        await asyncio.sleep(node.before_sleep_time)
        await browser.handle_new_tabs(0)

        live_url = await browser.get_current_page_url()

        action = locator_action(node)
        if action is not None:
            await probe_row(row, browser)
            command, reason = await choose_command(row, browser)
            if command is None:
                # Distillation deletes the agent's detours, so reaching an
                # element from a different page than the recording is normal.
                if row.url_before and not same_page(live_url, row.url_before):
                    reason = f"{reason}; on {live_url}, recorded at {row.url_before}"
                verdict.status = "demoted"
                row.classification = "non_deterministic"
                verdict.reason = row.reason = reason
                report.verdicts.append(verdict)
                report.stopped_at = position
                report.stopped_because = (
                    "no measured locator for this row, so the path breaks here"
                )
                break
            action.command = command
            verdict.reason = reason
            verdict.command = command

        before_url = await browser.get_current_page_url()
        downloads_before = files_in_downloads_dir(browser)
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

        effective, evidence = await observe_effect(row, node, browser, before_url)
        if not effective:
            verdict.status = "unmeasured"
            verdict.reason = f"{verdict.reason}; no observable effect ({evidence})"
            report.verdicts.append(verdict)
            report.stopped_at = position
            report.stopped_because = "node executed without evidence it acted"
            break

        verdict.status = "verified"
        if verdict.downloaded:
            evidence = f"{evidence}; downloaded {verdict.downloaded}"
        verdict.reason = f"{verdict.reason}; {evidence}"
        report.verdicts.append(verdict)

    for row in rows[len(report.verdicts) :]:
        report.verdicts.append(
            NodeVerdict(
                step=row.step,
                action=row.action,
                status="not_reached",
                reason="pass stopped before this row",
            )
        )

    report.final_url = await browser.get_current_page_url()
    report.signals = {
        "nodes": len(automation.nodes),
        "verified": report.verified_count,
        "llm_tokens": await total_llm_tokens(task, memory),
        "downloaded_files": [
            verdict.downloaded for verdict in report.verdicts if verdict.downloaded
        ],
        "output_data": [
            entry.model_dump(mode="json") for entry in memory.variables.output_data
        ],
    }
    return report
