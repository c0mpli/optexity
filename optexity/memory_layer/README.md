# memory_layer

An `agentic_task` node hands the page to browser-use, which reasons its way to the
goal with an LLM on **every** run and forgets everything when it returns. Every
other node type already has a fast path: `command` (a Playwright locator) is tried
first, and `prompt_instructions` only runs if it fails.

This package closes that gap. It records one agentic run, works out which of its
steps were load-bearing, and compiles them into locator-driven nodes — so the
second run costs no tokens.

```
build once                          then, every production run
──────────                          ─────────────────────────
capture ──▶ distill ──▶ verify      run ──▶ did a node fall back?
   │           │          │                        │
   │           │          │                   heal ─┘
   │           │          └─ measures each locator on the live page
   │           └─ deterministic / redundant / non-deterministic
   └─ saves the AgentHistoryList every caller used to discard
              │
              ├──▶ enrich   (Bonus A)
              └──▶ loop     (Bonus B)
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
| `heal.py` | folds a production run's LLM fallbacks back into the automation |
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

## Keeping it deterministic

Distilling is a one-off; a site changing is not. When a node's locator drifts, the
`prompt_instructions` fallback rescues the run — correctly, but at LLM cost, and
it will pay that cost again on every run after it.

`heal` closes that loop:

```bash
python -m optexity.memory_layer.heal /tmp/roboform.json <task-logs-dir> -o /tmp/healed.json
```

The handlers reach `log_interacted_locator` only from the index-based path, which
they only take once the command has failed. So `step_N/locator_candidates.json`
existing means precisely: this node's locator broke, and the LLM found the element
anyway — here it is. `heal` reads that, takes the strongest candidate above the
stability threshold, and writes it back, so the next run is deterministic again.

It reads only what a finished run already wrote. No browser, no replay, no second
LLM call — which is what makes it usable on a flow that submits a form or takes a
payment and therefore cannot be run twice to check.

It exits non-zero when any node needed the LLM, so a scheduled run can alert on an
automation that is decaying. The report leads with that number:

```
  1/4 nodes needed the LLM (75% deterministic)

  node 1: label 80
    was  locator("input[name='10address1']")
    now  get_by_label('Address 1')
```

A rescue that turns up nothing above the threshold is counted but not applied —
a positional xpath is how the original command drifted in the first place.

### Growing the path

Healing repairs a node that moved. A site that *adds* a step — a consent dialog,
a new interstitial — needs the path to grow instead.

That recovery already happens: when a node is blocked, the error classifier
returns `overlay_popup_blocking` and fires a `CloseOverlayPopupAction`, an agent
that dismisses the overlay so the node can retry. It is captured under
`step_N/recovery_history.json`, distilled, and inserted **before** the node it
unblocked, so the next run dismisses the dialog with a plain click.

An inserted step is optional by construction, because the overlay may simply not
be there next time — a cookie banner is gone once the cookie is set. It carries
`max_tries: 1` and `skip_prompt: True`, and `command_based_action_with_retry`
returns its error rather than raising unless `assert_locator_presence` is set. So
a miss costs one failed locator lookup: no LLM call, no exception, run continues.
This is the one node shape where `skip_prompt: True` is right — the step is
meant to be skippable, so a silent no-op is the goal rather than a hidden failure.

Only deterministic rows are kept. Compiling an agentic node here would make every
future run pay an LLM to re-derive the same dismissal.

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
- The path grows only where the runtime already recovers, which today means an
  overlay the error classifier recognises. A site that inserts a step nothing
  recovers from still fails the run.
- Redundancy is decided from recorded effects, not by ablation: a step is dropped
  because a later one supersedes it, never because removing it was tried.
