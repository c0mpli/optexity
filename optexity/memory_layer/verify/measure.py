import logging
import time
from typing import TYPE_CHECKING

from optexity.memory_layer.distill.candidates import (
    MINIMUM_STABILITY_SCORE,
    propose_bundle,
)
from optexity.schema.memory_layer import TraceRow, by_stability, verified_candidates

if TYPE_CHECKING:
    from optexity.inference.infra.browser import Browser

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 5.0
PROBE_BUDGET_SECONDS = 20.0
VERIFIED_CANDIDATES_WANTED = 2
# A checkbox list can be long; an index past this is not a locator worth having.
POSITIONAL_MAX_MATCHES = 20


async def probe(
    command: str, browser: "Browser", timeout: float = PROBE_TIMEOUT_SECONDS
) -> int | None:
    """None and zero differ: an unparseable command tells us nothing."""
    from optexity.inference.core.run_automation import count_locator_matches

    try:
        return await count_locator_matches(command, timeout, browser)
    except Exception as e:
        # A malformed command raises in eval, before count_locator_matches.
        if "TargetClosed" in type(e).__name__:
            raise
        logger.debug(f"probe failed for {command!r}: {e}")
        return None


async def probe_row(row: TraceRow, browser: "Browser") -> None:
    """Bounded twice: every probe costs a second of an authenticated session."""
    started = time.monotonic()
    verified = 0

    for candidate in sorted(row.candidates, key=by_stability, reverse=True):
        if verified >= VERIFIED_CANDIDATES_WANTED:
            break
        if time.monotonic() - started > PROBE_BUDGET_SECONDS:
            break
        candidate.match_count = await probe(candidate.command, browser)
        if candidate.matches_exactly_one_element:
            verified += 1


async def _positional_command(
    row: TraceRow, browser: "Browser"
) -> tuple[str, str] | None:
    """Disambiguate identical controls by their order on the page.

    Two bare checkboxes carry nothing to tell them apart, so every stable
    selector matches both and the only unique one is a brittle xpath. Intersect
    the two: the xpath says which element was recorded, and the index it falls
    at makes the stable selector unique without shipping the xpath.
    """
    oracle = next(
        (c for c in row.candidates if c.matches_exactly_one_element and c.command), None
    )
    ambiguous = next(
        (
            c
            for c in sorted(row.candidates, key=by_stability, reverse=True)
            if c.match_count is not None
            and 1 < c.match_count <= POSITIONAL_MAX_MATCHES
            and c.stability_score >= MINIMUM_STABILITY_SCORE
        ),
        None,
    )
    if oracle is None or ambiguous is None:
        return None

    for index in range(ambiguous.match_count or 0):
        command = f"{ambiguous.command}.nth({index})"
        # Timeout zero: most indexes are the wrong element and would each wait
        # out the full probe timeout before reporting the zero we expect.
        if await probe(f"{command}.and_(page.{oracle.command})", browser, 0) == 1:
            return command, (
                f"{ambiguous.kind} score={ambiguous.stability_score}, "
                f"matched {ambiguous.match_count}; disambiguated to index {index}"
            )
    return None


async def choose_command(row: TraceRow, browser: "Browser") -> tuple[str | None, str]:
    verified = verified_candidates(row.candidates)
    if not verified:
        measured = sum(
            1 for candidate in row.candidates if candidate.match_count is not None
        )
        if not measured:
            return None, "no candidate could be measured"
        return None, f"no candidate matched exactly one element ({measured} probed)"

    best = verified[0]
    # Probing can demote the top candidate and leave only a far weaker one
    # unique; shipping that is the guess this layer exists to avoid.
    if best.stability_score < MINIMUM_STABILITY_SCORE:
        positional = await _positional_command(row, browser)
        if positional is not None:
            return positional
        return None, (
            f"no verified candidate above threshold "
            f"(best verified: {best.kind} score={best.stability_score})"
        )

    bundle = propose_bundle(row.candidates)
    if bundle is not None:
        # Only the assembled expression can settle whether the union is unique.
        bundle.match_count = await probe(bundle.command, browser)
        row.candidates.append(bundle)
        if bundle.matches_exactly_one_element:
            return bundle.command, (
                f"{bundle.kind}, both measured and the pair matched 1"
            )
        logger.debug(
            f"bundle matched {bundle.match_count}; shipping the single best locator"
        )

    return best.command, f"{best.kind} score={best.stability_score}, matched 1"
