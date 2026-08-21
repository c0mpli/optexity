from optexity.schema.actions.interaction_action import BaseAction
from optexity.schema.automation import ActionNode, Automation
from optexity.schema.memory_layer import VerificationReport

LOCATOR_FIELDS = ("click_element", "input_text", "select_option", "upload_file")


def locator_action(node: ActionNode) -> BaseAction | None:
    """The node's interaction that carries a command, if it has one."""
    for field in LOCATOR_FIELDS:
        action = getattr(node.interaction_action, field, None)
        if action is not None:
            return action
    return None


def apply_verdicts(automation: Automation, report: VerificationReport) -> None:
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
