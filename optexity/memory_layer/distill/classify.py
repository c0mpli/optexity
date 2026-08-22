from optexity.memory_layer.distill.candidates import (
    MINIMUM_STABILITY_SCORE,
    build_candidates,
)
from optexity.memory_layer.distill.compile import ACTION_FIELD
from optexity.schema.memory_layer import Classification, Trace, TraceRow

DETERMINISTIC_ACTIONS = {*ACTION_FIELD, "navigate", "go_back"}

READ_ONLY_ACTIONS = {
    "screenshot",
    "evaluate",
    "done",
    "wait",
    "find_text",
    "dropdown_options",
    "read_file",
}

# browser-use emits a submit as a separate send_keys, so typing then Enter would
# distil to typing only. key_press cannot express it: handle_keypress takes
# Enter/Tab/Space, KEY_NAMES has none.
ENTER_KEYS = {"Enter", "Return", "\n"}
# evaluate runs whatever javascript the agent wrote, so "read-only" is a claim
# about the code, not about the action. Anything that could write is kept.
MUTATING_JS = (
    ".click(",
    ".submit(",
    ".value=",
    ".value =",
    ".checked",
    ".dispatchEvent(",
    ".setAttribute(",
    ".remove(",
    ".removeAttribute(",
    "innerHTML",
    "outerHTML",
    "document.write",
)


def _mutates_the_page(row: TraceRow) -> bool:
    code = str(row.params.get("code", "")).replace(" ", "")
    return any(marker.replace(" ", "") in code for marker in MUTATING_JS)


def _mark(row: TraceRow, classification: Classification, reason: str) -> None:
    row.classification, row.reason = classification, reason


def classify(trace: Trace, automation_url: str | None) -> None:
    seen_first_action = False

    for row in trace.rows:
        if not row.executed:
            _mark(row, Classification.REDUNDANT, "action never executed")
            continue
        if row.error:
            _mark(row, Classification.REDUNDANT, f"action errored ({row.error[:120]})")
            continue
        if row.action == "scroll":
            _mark(
                row,
                Classification.REDUNDANT,
                "the command path already scrolls the element into view",
            )
            continue
        if row.action not in DETERMINISTIC_ACTIONS:
            if row.action in READ_ONLY_ACTIONS and not _mutates_the_page(row):
                _mark(
                    row,
                    Classification.REDUNDANT,
                    f"'{row.action}' only observes the page",
                )
            elif row.action == "evaluate":
                _mark(
                    row,
                    Classification.NON_DETERMINISTIC,
                    "evaluate ran javascript that can write to the page",
                )
            else:
                _mark(
                    row,
                    Classification.NON_DETERMINISTIC,
                    f"no equivalent for '{row.action}'",
                )
            continue
        if (
            row.action == "navigate"
            and not seen_first_action
            and automation_url
            and (row.params.get("url") or "").rstrip("/") == automation_url.rstrip("/")
        ):
            _mark(
                row,
                Classification.REDUNDANT,
                "the run already begins on the automation url",
            )
            continue

        if row.action in ACTION_FIELD:
            if row.element is None:
                _mark(
                    row,
                    Classification.NON_DETERMINISTIC,
                    "no element was recorded for this action",
                )
                continue
            if row.element.is_in_subframe:
                _mark(
                    row,
                    Classification.NON_DETERMINISTIC,
                    f"element is inside frame {row.element.frame_id}",
                )
                continue

            row.candidates = build_candidates(row.element)
            best = row.best_candidate
            if best is None or best.stability_score < MINIMUM_STABILITY_SCORE:
                row.classification = Classification.NON_DETERMINISTIC
                row.reason = (
                    f"best locator scores {best.stability_score if best else 0} "
                    f"< {MINIMUM_STABILITY_SCORE}"
                )
                continue
            row.reason = f"locator {best.kind} score={best.stability_score}"

        row.classification = Classification.DETERMINISTIC
        row.reason = row.reason or f"deterministic {row.action}"
        seen_first_action = True

    _fold_enter_into_preceding_input(trace)
    _mark_superseded_interactions(trace)


def _fold_enter_into_preceding_input(trace: Trace) -> None:
    """Left alone, a search that types then submits distils to one that types."""
    previous_kept: TraceRow | None = None
    for row in trace.rows:
        if row.action == "send_keys" and previous_kept is not None:
            keys = str(row.params.get("keys", ""))
            if keys in ENTER_KEYS and previous_kept.action == "input":
                previous_kept.params["press_enter"] = True
                _mark(
                    row,
                    Classification.REDUNDANT,
                    f"folded into step {previous_kept.step} as press_enter",
                )
                continue
        if row.classification == Classification.DETERMINISTIC:
            previous_kept = row


def _mark_superseded_interactions(trace: Trace) -> None:
    """Of a run of identical consecutive actions, only the last had any effect.

    fill replaces contents; navigating to one url twice is idempotent. Clicks are
    neither -- two on a checkbox cancel out, two on a stepper count twice -- so a
    repeated click is left alone. Element identity comes from element_hash, which
    cannot find an element on a page but can tell two recorded rows apart."""
    deterministic = trace.rows_with(Classification.DETERMINISTIC)
    for row, following in zip(deterministic, deterministic[1:], strict=False):
        if row.action != following.action:
            continue
        if row.action == "navigate":
            if row.params.get("url") != following.params.get("url"):
                continue
        elif row.action == "input":
            if row.element is None:
                continue
            if not row.element.is_same_element_as(following.element):
                continue
            if row.params.get("press_enter"):
                continue
        else:
            continue
        row.classification = Classification.REDUNDANT
        row.reason = f"superseded by the same {row.action} at step {following.step}"
