import ast
import json
import logging
from pathlib import Path

from optexity.memory_layer.agent_history import element_from_record, load_trace
from optexity.memory_layer.capture import RECOVERY_HISTORY_FILENAME
from optexity.memory_layer.distill.candidates import (
    MINIMUM_STABILITY_SCORE,
    build_candidates,
)
from optexity.memory_layer.distill.classify import classify
from optexity.memory_layer.distill.compile import compile_nodes
from optexity.memory_layer.verify.verdicts import LOCATOR_FIELDS, locator_action
from optexity.schema.automation import ActionNode, Automation
from optexity.schema.memory_layer import (
    Classification,
    HealReport,
    NodeGrowth,
    NodeHeal,
    RecordedLocator,
    by_score,
)

logger = logging.getLogger(__name__)

LOCATOR_CANDIDATES_FILENAME = "locator_candidates.json"
INTERACTED_ELEMENT_FILENAME = "interacted_element.json"
# Written only when a node reaches the prompt path, so it separates a node that
# fell back from one whose command worked. locator_candidates.json does not:
# handle_command records those for a successful command too, which would count
# every healthy node as rescued and let a working command be rewritten.
FALLBACK_EVIDENCE_FILENAME = "final_prompt.txt"
# An inserted step is optional by construction: the overlay it clears may simply
# not be there next run. One try and no prompt makes a miss cost a failed locator
# lookup and nothing else -- no LLM call, and no raise, since the command path
# returns its error rather than throwing unless assert_locator_presence is set.
RECOVERY_MAX_TRIES = 1


def command_from_recorded(expression: str) -> str | None:
    """The command form of a locator the fallback recorded.

    log_interacted_locator writes ``page.<locator><method>`` for a human to paste
    into a console; browser.py evaluates ``eval(f"page.{command}")``. Both the
    prefix and the trailing call have to come back off, and the call is dropped
    structurally so that a locator ending in .first survives.
    """
    try:
        call = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
        return None
    locator = ast.unparse(call.func.value)
    return locator[len("page.") :] if locator.startswith("page.") else None


def rescued_locators(step_directory: Path) -> list[RecordedLocator]:
    """What the LLM fallback found, or nothing if this node never needed it."""
    if not (step_directory / FALLBACK_EVIDENCE_FILENAME).is_file():
        return []
    path = step_directory / LOCATOR_CANDIDATES_FILENAME
    if not path.is_file():
        return []
    try:
        recorded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"could not read {path}: {e}")
        return []
    return [RecordedLocator.model_validate(entry) for entry in recorded]


def rescored_locators(step_directory: Path) -> list[RecordedLocator]:
    """Score the rescued element by this layer's rules rather than the runtime's.

    The runtime drops any attribute that looks generated, which on a form whose
    only handle is name="04fullname" leaves a positional xpath and nothing else.
    The same ladder distillation uses re-admits those, so a node that fell back
    on such a page can still heal into something stable.
    """
    path = step_directory / INTERACTED_ELEMENT_FILENAME
    if not path.is_file():
        return []
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"could not read {path}: {e}")
        return []
    return [
        # Rebuilt as page.<locator>.x() so command_from_recorded can strip it
        # back off the same way it does for a runtime-recorded line.
        RecordedLocator(
            locator=f"page.{candidate.command}.x()",
            kind=candidate.kind,
            score=candidate.stability_score,
        )
        for candidate in build_candidates(element_from_record(record))
    ]


def best_command(
    candidates: list[RecordedLocator],
) -> tuple[str, RecordedLocator] | None:
    for candidate in sorted(candidates, key=by_score, reverse=True):
        if candidate.score < MINIMUM_STABILITY_SCORE:
            # Below this the recording is a positional xpath or bare text, which
            # is how the command being healed drifted in the first place.
            return None
        command = command_from_recorded(candidate.locator)
        if command:
            return command, candidate
    return None


def recovered_nodes(step_directory: Path, automation: Automation) -> list[ActionNode]:
    """The steps an overlay-closing agent had to take, as optional nodes.

    A blocked node makes the error classifier fire a popup closer, which is an
    agent solving a step the recording never contained. Distilling what it did is
    the only way the path grows -- healing alone can repair a node that moved but
    never add one the site introduced.

    Only deterministic rows are kept. Compiling an agentic node here would make
    every future run pay an LLM to re-derive the same dismissal.
    """
    path = step_directory / RECOVERY_HISTORY_FILENAME
    if not path.is_file():
        return []

    trace = load_trace(path)
    classify(trace, automation.url)
    for row in trace.rows:
        if row.classification != Classification.DETERMINISTIC:
            row.classification = Classification.REDUNDANT

    nodes = []
    # Declare into the automation's own parameters: a recovery that types
    # something would otherwise emit a placeholder nothing substitutes, and the
    # field would receive the literal {name[0]}.
    parameters = automation.parameters.input_parameters
    for node in compile_nodes(trace, parameters):
        interaction = node.get("interaction_action") or {}
        action = next(
            (interaction[field] for field in LOCATOR_FIELDS if field in interaction),
            None,
        )
        if action is None:
            continue
        interaction["max_tries"] = RECOVERY_MAX_TRIES
        action["skip_prompt"] = True
        nodes.append(ActionNode.model_validate(node))
    return nodes


def heal(automation: Automation, logs_directory: str | Path) -> HealReport:
    """Fold what the LLM fallback found back into the automation.

    A node whose command still works costs nothing and leaves nothing behind. A
    node that fell back leaves the element browser-use actually acted on, which
    is a stronger record than the recording this automation was compiled from.

    Reads only what a finished run wrote, so it needs no browser and no replay --
    the one healing strategy available to a flow that cannot be run twice.
    """
    report = HealReport()
    logs = Path(logs_directory)
    insertions: list[tuple[int, list[ActionNode]]] = []

    for position, node in enumerate(automation.nodes):
        grown = recovered_nodes(logs / f"step_{position}", automation)
        if grown:
            insertions.append((position, grown))
            report.growth.append(
                NodeGrowth(
                    before=position,
                    commands=[locator_action(n).command for n in grown],
                )
            )

        action = locator_action(node)
        if action is None or not action.command:
            continue
        report.nodes += 1

        step_directory = logs / f"step_{position}"
        candidates = rescued_locators(step_directory)
        if not candidates:
            continue
        report.rescued += 1

        chosen = best_command(candidates + rescored_locators(step_directory))
        if chosen is None:
            logger.info(f"node {position}: rescued, but nothing stable enough to keep")
            continue
        command, recorded = chosen
        if command == action.command:
            continue

        report.heals.append(
            NodeHeal(
                node=position,
                was=action.command,
                now=command,
                kind=recorded.kind,
                score=recorded.score,
            )
        )
        action.command = command

    # Last first, so an earlier insertion does not shift a later position.
    for position, nodes in reversed(insertions):
        automation.nodes[position:position] = nodes

    return report


def format_report(report: HealReport) -> str:
    lines = [
        f"  {report.rescued}/{report.nodes} nodes needed the LLM "
        f"({report.determinism:.0%} deterministic)",
        "",
    ]
    for node_heal in report.heals:
        lines.append(f"  node {node_heal.node}: {node_heal.kind} {node_heal.score}")
        lines.append(f"    was  {node_heal.was}")
        lines.append(f"    now  {node_heal.now}")
    for growth in report.growth:
        lines.append(f"  new step(s) before node {growth.before}:")
        lines.extend(f"    + {command}" for command in growth.commands)
    if not (report.heals or report.growth):
        lines.append("  nothing to heal")
    return "\n".join(lines)
