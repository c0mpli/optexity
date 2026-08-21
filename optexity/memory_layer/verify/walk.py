import asyncio

from optexity.memory_layer.capture import total_llm_tokens
from optexity.memory_layer.verify.effects import (
    NAVIGATING_ACTIONS,
    NAVIGATION_GRACE_SECONDS,
    NAVIGATION_TIMEOUT_SECONDS,
    files_in_downloads_dir,
    observe_effect,
    same_page,
    wait_for_download,
    wait_for_navigation,
)
from optexity.memory_layer.verify.measure import choose_command, probe_row
from optexity.memory_layer.verify.verdicts import locator_action
from optexity.schema.memory_layer import (
    NodeVerdict,
    RunSignals,
    Trace,
    VerificationReport,
)


async def verify_automation(
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
    report.signals = RunSignals(
        llm_tokens=await total_llm_tokens(task, memory),
        downloaded_files=[
            verdict.downloaded for verdict in report.verdicts if verdict.downloaded
        ],
        output_data=[
            entry.model_dump(mode="json") for entry in memory.variables.output_data
        ],
    )
    return report
