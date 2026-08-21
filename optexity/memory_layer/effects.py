import asyncio
import time
from pathlib import Path
from urllib.parse import urlsplit

from optexity.memory_layer.trace import TraceRow

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
        done = sorted(n for n in arrived if not n.endswith((".crdownload", ".tmp")))
        if done:
            return done[0]
        await asyncio.sleep(0.1)
    return None


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
