# memory_layer

An `agentic_task` node hands the page to browser-use, which reasons its way to the
goal with an LLM on **every** run and forgets everything when it returns. Every
other node type already has a fast path: `command` (a Playwright locator) is tried
first, and `prompt_instructions` only runs if it fails.

This package closes that gap. It records one agentic run, works out which of its
steps were load-bearing, and compiles them into locator-driven nodes — so the
second run costs no tokens.

```
capture ──▶ distill ──▶ verify ──▶ enrich      (Bonus A)
   │           │           │
   │           │           └─ measures each locator on the live page
   │           └─ deterministic / redundant / non-deterministic
   └─ saves the AgentHistoryList every caller used to discard
                    │
                    └──▶ loop ──▶ replay, re-learn, recompile   (Bonus B)
```

| file | does |
|---|---|
| `capture.py` | saves `agent_history.json` per agentic node; sums both halves of the token bill |
| `trace.py` | the recorded run as rows: action, element, candidate locators, timings |
| `candidates.py` | ranked Playwright locators for a recorded element |
| `distill.py` | classifies rows, emits an `Automation` |
| `verify.py` | walks the automation against a live page, verdict per node |
| `enrich.py` | lets an LLM improve it under pydantic validation, without authoring selectors |
| `loop.py` | replays, re-learns the agentic steps, recompiles, repeats |
| `verify_runner.py` | CLI for verify and the loop |

## Run it

Distilling needs no browser and no API key, so start here:

```bash
python -m optexity.memory_layer.distill \
    automations/roboform_agent_history.json -o /tmp/roboform.json
```

That fixture is a real captured run, committed so the pipeline is runnable from a
clean checkout. It prints the classification of every recorded action and writes a
deterministic `Automation`.

Measuring the result against the live page needs a browser:

```bash
python -m optexity.memory_layer.verify_runner \
    automations/roboform_agent_history.json -o /tmp/verified.json
```

Each node is probed for a unique match, run, and then checked for an observable
effect — a url change, a value on the page, a downloaded file. A node that runs
without evidence it acted stops the walk rather than being recorded as a pass. The
report ends with the agentic run's cost next to the measured one.

`--rounds N` runs the Bonus B loop instead: each round replays, re-distils the
steps that still needed the agent, and recompiles. It stops when every step is
deterministic, when a round verifies fewer nodes than the last, when the walk
stops on a node it cannot measure, or when a round learns nothing new.

Bonus A is a separate step, since an LLM is optional to the pipeline:

```bash
python -m optexity.memory_layer.enrich \
    automations/roboform_agent_history.json -o /tmp/enriched.json \
    --objective "fill in the contact form"
```

It names parameters and writes fallback instructions. It cannot emit a `command`:
`NodePatch` has no field to carry one, so a hallucinated selector has nowhere to go.

## Capturing your own run

The three CLIs above all take an `agent_history.json`. To produce one, run an
agentic automation through the worker with the local override pointed at it:

```bash
export DEPLOYMENT=dev                 # keeps delete_local_data from rm -rf'ing the cache
export API_KEY=...                    # from the optexity dashboard
export LLM_MODEL=... LLM_MODEL_API_KEY=...
export TEST_AUTOMATION_PATH=$PWD/automations/the_internet_agentic.json

uv run optexity inference --port 9000 --child_process_id 0
curl -X POST localhost:9000/inference -H 'Content-Type: application/json' \
     -d '{"endpoint_name":"<any>","input_parameters":{},"unique_parameter_names":[]}'
```

The override replaces whatever automation the endpoint resolves to, so
`endpoint_name` can be anything that exists. The history lands in the task's
`logs_directory` under `step_<n>/agent_history.json`.

Three agentic fixtures are in `automations/`: `roboform_agentic.json` (one form),
`the_internet_agentic.json` (login → secure area → download, three page
transitions), `checkboxes_agentic.json` (two toggles).

## What it does not do

- The walk stops at the first node it cannot measure and reports the rest
  `not_reached`, rather than guessing. It refuses instead of inventing.
- A non-deterministic **input** row halts the walk, so the loop cannot currently
  re-learn one — only agentic clicks and navigations round-trip.
- Redundancy is decided from recorded effects, not by ablation: a step is dropped
  because a later one supersedes it, never because removing it was tried.
