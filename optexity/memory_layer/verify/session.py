import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from optexity.inference.infra.actual_browser import ActualBrowser
from optexity.inference.infra.browser import Browser
from optexity.inference.models import normalize_model
from optexity.schema.automation import Automation
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


def build_task(automation: Automation, endpoint_name: str = "verify") -> Task:
    """run_action_node posts a trajectory every fifth step, so without
    upload_artifacts a verification pass fills the production store."""
    parameters = automation.parameters
    return Task(
        task_id=str(uuid.uuid4()),
        user_id="local",
        recording_id="local",
        company_id="local",
        endpoint_name=endpoint_name,
        automation=automation,
        input_parameters={k: list(v) for k, v in parameters.input_parameters.items()},
        secure_parameters={k: list(v) for k, v in parameters.secure_parameters.items()},
        unique_parameter_names=[],
        created_at=datetime.now(timezone.utc),
        status="running",
        api_key="local",
        max_retries=0,
        upload_artifacts=False,
    )


def missing_parameters(automation: Automation) -> list[str]:
    """An empty password makes every later node measure an error page, turning
    one missing input into a page of fabricated verdicts."""
    return [
        name
        for name, values in automation.parameters.input_parameters.items()
        if not values or not str(values[0]).strip()
    ]


def make_build_session(headless: bool, port: int) -> Callable[..., Awaitable]:
    """A factory for one browser and task per call.

    Built per call rather than per process because a warm profile would let a
    login verify against a session that was already authenticated, and because
    run_action_node substitutes parameter placeholders into nodes in place.
    """

    async def build_session(label: str, automation: Automation):
        task = build_task(automation)
        memory = Memory(unique_child_arn=label)
        memory.update_system_info()
        memory.automation_state.step_index = -1
        memory.automation_state.try_index = 0

        actual_browser = ActualBrowser(
            channel=automation.browser_channel,
            unique_child_arn=label,
            port=port,
            headless=headless,
            allow_cookies=automation.allow_cookies,
        )
        browser = None

        async def teardown():
            if browser is not None:
                try:
                    await asyncio.wait_for(browser.stop(), timeout=30)
                except Exception as e:
                    logger.warning(f"error stopping browser: {e}")
            try:
                await actual_browser.stop()
            except Exception as e:
                logger.warning(f"error stopping actual browser: {e}")

        try:
            await actual_browser.start()
            if actual_browser.cdp_url is None:
                raise RuntimeError("browser started but exposed no CDP url")
            browser = Browser(
                memory=memory,
                cdp_url=str(actual_browser.cdp_url),
                llm_model=normalize_model(task.llm_provider, task.llm_model_name),
            )
            await browser.start()
            await browser.go_to_url("about:blank")
            await browser.go_to_url(automation.url, retry_count=3)
        except Exception:
            await teardown()
            raise
        return task, memory, browser, teardown

    return build_session
