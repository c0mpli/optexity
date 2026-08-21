import argparse
import ast
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from optexity.memory_layer.candidates import MINIMUM_STABILITY_SCORE
from optexity.memory_layer.verify import locator_action
from optexity.schema.automation import Automation

logger = logging.getLogger(__name__)

LOCATOR_CANDIDATES_FILENAME = "locator_candidates.json"


class NodeHeal(BaseModel):
    node: int
    was: str | None
    now: str
    kind: str
    score: int


class HealReport(BaseModel):
    heals: list[NodeHeal] = Field(default_factory=list)
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

    for position, node in enumerate(automation.nodes):
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
    if not report.heals:
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
