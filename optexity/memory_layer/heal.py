import argparse
import ast
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from optexity.memory_layer.agent_history import load_trace
from optexity.memory_layer.candidates import MINIMUM_STABILITY_SCORE
from optexity.memory_layer.capture import RECOVERY_HISTORY_FILENAME
from optexity.memory_layer.distill import classify, compile_nodes
from optexity.memory_layer.trace import Classification
from optexity.memory_layer.verdicts import LOCATOR_FIELDS, locator_action
from optexity.schema.automation import ActionNode, Automation

logger = logging.getLogger(__name__)

LOCATOR_CANDIDATES_FILENAME = "locator_candidates.json"
# An inserted step is optional by construction: the overlay it clears may simply
# not be there next run. One try and no prompt makes a miss cost a failed locator
# lookup and nothing else -- no LLM call, and no raise, since the command path
# returns its error rather than throwing unless assert_locator_presence is set.
RECOVERY_MAX_TRIES = 1


class NodeHeal(BaseModel):
    node: int
    was: str | None
    now: str
    kind: str
    score: int


class NodeGrowth(BaseModel):
    before: int
    commands: list[str]


class HealReport(BaseModel):
    heals: list[NodeHeal] = Field(default_factory=list)
    growth: list[NodeGrowth] = Field(default_factory=list)
    rescued: int = 0
    nodes: int = 0

    @property
    def determinism(self) -> float:
        """Share of locator-driven nodes that ran without the LLM."""
        return 1.0 if not self.nodes else 1 - self.rescued / self.nodes


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


def rescued_locators(step_directory: Path) -> list[dict]:
    """What the LLM fallback found, or nothing if this node never needed it.

    log_interacted_locator runs only from the index-based path, which the handlers
    reach only after the command failed. So this file existing is the signal.
    """
    path = step_directory / LOCATOR_CANDIDATES_FILENAME
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"could not read {path}: {e}")
        return []


def best_command(candidates: list[dict]) -> tuple[str, str, int] | None:
    for candidate in sorted(
        candidates, key=lambda c: c.get("score") or 0, reverse=True
    ):
        score = candidate.get("score") or 0
        if score < MINIMUM_STABILITY_SCORE:
            # Below this the recording is a positional xpath or bare text, which
            # is how the command being healed drifted in the first place.
            return None
        command = command_from_recorded(candidate.get("locator") or "")
        if command:
            return command, candidate.get("kind") or "", score
    return None


def recovered_nodes(step_directory: Path, url: str | None) -> list[ActionNode]:
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
    classify(trace, url)
    for row in trace.rows:
        if row.classification != Classification.DETERMINISTIC:
            row.classification = Classification.REDUNDANT

    nodes = []
    for node in compile_nodes(trace, {}, set()):
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
        grown = recovered_nodes(logs / f"step_{position}", automation.url)
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

        candidates = rescued_locators(logs / f"step_{position}")
        if not candidates:
            continue
        report.rescued += 1

        chosen = best_command(candidates)
        if chosen is None:
            logger.info(f"node {position}: rescued, but nothing stable enough to keep")
            continue
        command, kind, score = chosen
        if command == action.command:
            continue

        report.heals.append(
            NodeHeal(
                node=position, was=action.command, now=command, kind=kind, score=score
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m optexity.memory_layer.heal",
        description="Fold a finished run's LLM fallbacks back into its automation.",
    )
    parser.add_argument("automation", type=Path, help="the automation that was run")
    parser.add_argument("logs", type=Path, help="that run's logs directory")
    parser.add_argument("-o", "--out", type=Path, help="write the healed automation")
    args = parser.parse_args(argv)

    logging.getLogger("optexity").setLevel(logging.INFO)
    for path in (args.automation, args.logs):
        if not path.exists():
            parser.error(f"no such path: {path}")

    automation = Automation.model_validate_json(args.automation.read_text())
    report = heal(automation, args.logs)
    print(f"\n{args.automation}")
    print(format_report(report))
    print()

    if args.out:
        args.out.write_text(automation.model_dump_json(indent=2, exclude_none=True))
    return 0 if report.rescued == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
