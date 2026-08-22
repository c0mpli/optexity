from pathlib import Path

from optexity.memory_layer.agent_history import load_trace
from optexity.memory_layer.distill.classify import classify
from optexity.memory_layer.distill.compile import trace_to_automation
from optexity.schema.automation import Automation
from optexity.schema.memory_layer import Trace

CLASSIFICATION_MARKS = {
    "deterministic": "keep",
    "redundant": "drop",
    "non_deterministic": "llm",
}


def distill(history_path: Path, url: str | None = None) -> tuple[Automation, Trace]:
    trace = load_trace(history_path)
    classify(trace, url or trace.url)
    return trace_to_automation(trace, url), trace


def print_summary(
    trace: Trace, automation: Automation, history_path: Path, out_path: Path
) -> None:
    counts = trace.counts_by_classification()
    print(f"\n{history_path}  ->  {out_path}")
    for label in ("total", "deterministic", "redundant", "non_deterministic"):
        print(f"  {label:<18}: {counts.get(label, 0)}")
    print(f"  {'parameters':<18}: {list(automation.parameters.input_parameters)}\n")
    for row in trace.rows:
        mark = CLASSIFICATION_MARKS.get(row.classification, "????")
        print(f"  [{mark:>4}] {row.action:<16} {row.reason}")
    print()
