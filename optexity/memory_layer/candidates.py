import ast
import re
from types import SimpleNamespace

from optexity.inference.core.interaction.utils import LocatorExtraction
from optexity.memory_layer.trace import (
    Candidate,
    Element,
    by_stability,
    verified_candidates,
)

# _looks_dynamic discards names like RoboForm's 04fullname and sits on the live
# LLM-fallback path, so rather than change it those are re-admitted a rung lower.
REJECTED_ATTRIBUTE_SCORES = {
    "data-testid": 100,
    "data-test-id": 98,
    "data-test": 98,
    "data-cy": 98,
    "data-qa": 98,
    "id": 92,
    "name": 84,
}
READMISSION_PENALTY = 10

# Upstream _css_attr interpolates raw, so a value carrying a quote cannot parse.
CSS_ATTRIBUTE_GROUP = re.compile(r"\[([^\[\]]*)\]")
WELL_FORMED_CSS_ATTRIBUTE = re.compile(r"^[\w:-]+([~^|*$]?=)'(?:[^'\\]|\\.)*'$")
MINIMUM_STABILITY_SCORE = 40

UNSCORED_ATTRIBUTE_LOCATORS = {
    "title": (74, "get_by_title"),
    "alt": (73, "get_by_alt_text"),
}
HREF_SCORE = 60

GROUPED_INPUT_TYPES = {"radio", "checkbox"}
NAME_VALUE_SCORE = 88

# A control's type says what it does, so it outlives the restyle that renames the
# class beside it -- the upstream ladder reads class and never this. Only values
# that name a function: type='text' describes every text box on the page.
FUNCTIONAL_TYPES = {"submit", "button", "reset", "checkbox", "radio", "file"}
FUNCTIONAL_TYPE_SCORE = 62

XPATH_ANCHOR_TAGS = ("dialog", "form", "table", "nav", "main", "article", "section")
ANCHORED_XPATH_SCORE = 15

# The last rung of workflow-use's selector ladder: text survives the classes and
# ids around it changing. Scored above the bare-text rung because the tag
# constrains it -- text alone resolves to whichever ancestor also contains it --
# and below css, because visible copy is still translated and reworded.
TAG_TEXT_SCORE = 45
MAX_TAG_TEXT_CHARS = 60

# browser-use records an accessible name but no role, so utils.py:225-238's
# mapping is recomputed here.
ROLE_BY_TAG = {"button": "button", "select": "combobox", "textarea": "textbox"}
ROLE_BY_INPUT_TYPE = {
    "checkbox": "checkbox",
    "radio": "radio",
    "button": "button",
    "submit": "button",
    "reset": "button",
    "text": "textbox",
    "search": "searchbox",
    "email": "textbox",
    "tel": "textbox",
    "url": "textbox",
    "password": "textbox",
    "number": "spinbutton",
}

NARROWINGS = (":not([type='hidden'])", ":visible")
NARROWING_PENALTY = 5

# or_() is a union, so bundling is only unambiguous if each leg was separately
# measured at exactly one match.
MAX_BUNDLED_CANDIDATES = 2
KIND_FAMILY_ALIASES = {"css+text": "css", "role+text": "text"}


def computed_role(element: Element) -> str:
    explicit = (element.attributes.get("role") or "").strip()
    if explicit:
        return explicit
    if element.tag_name == "a":
        return "link" if element.attributes.get("href") else ""
    if element.tag_name == "input":
        input_type = (element.attributes.get("type") or "text").lower()
        return ROLE_BY_INPUT_TYPE.get(input_type, "textbox")
    return ROLE_BY_TAG.get(element.tag_name, "")


def _escaped(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _css_attribute_selector(tag_name: str, attribute: str, value: str) -> str:
    """Escaped counterpart of upstream _css_attr, which interpolates raw."""
    return f"{tag_name}[{attribute}='{_escaped(value)}']"


class _RecordedElementAsNode:
    """ax_node unlocks _scored_candidates' role+name rung, the only family
    independent of id/name/class."""

    def __init__(self, element: Element):
        self.attributes = element.attributes
        self.tag_name = element.tag_name
        self.xpath = element.xpath
        accessible_name = (element.accessible_name or "").strip()
        role = computed_role(element)
        self.ax_node = (
            SimpleNamespace(role=role, name=accessible_name)
            if (role or accessible_name)
            else None
        )
        self._accessible_name = accessible_name

    def get_meaningful_text_for_llm(self) -> str:
        return self._accessible_name


def _is_parseable(command: str) -> bool:
    """Checks the decoded selector, not the command text: the command carries a
    layer of Python escaping that eval strips before the css engine sees it."""
    quoted = re.match(r'^locator\(("(?:[^"\\]|\\.)*")[,)]', command)
    if not quoted:
        return True
    try:
        selector = ast.literal_eval(quoted.group(1))
    except (SyntaxError, ValueError):
        return False
    for attribute_group in CSS_ATTRIBUTE_GROUP.findall(selector):
        if "=" not in attribute_group:
            continue
        if not WELL_FORMED_CSS_ATTRIBUTE.match(attribute_group):
            return False
    return True


def _locator_candidate(selector: str, kind: str, score: int) -> Candidate:
    quoted = LocatorExtraction._quote_locator_value(selector, 400)
    return Candidate(command=f"locator({quoted})", kind=kind, stability_score=score)


def _readmitted_candidates(element: Element) -> list[Candidate]:
    readmitted = []
    tag_name = element.tag_name or "*"
    for attribute, score in REJECTED_ATTRIBUTE_SCORES.items():
        value = (element.attributes.get(attribute) or "").strip()
        if not value:
            continue
        rejected = LocatorExtraction._looks_dynamic(value)
        needs_escaping = "'" in value or "\\" in value
        # Rebuilt-for-escaping values keep their full score.
        if not rejected and not needs_escaping:
            continue
        readmitted.append(
            _locator_candidate(
                _css_attribute_selector(tag_name, attribute, value),
                f"{attribute} (re-admitted)" if rejected else attribute,
                score - (READMISSION_PENALTY if rejected else 0),
            )
        )
    return readmitted


def _tag_text_candidates(element: Element) -> list[Candidate]:
    """A control identified by its own visible text, scoped to its tag."""
    text = (element.accessible_name or "").strip()
    tag = (element.tag_name or "").strip()
    if not (text and tag) or len(text) > MAX_TAG_TEXT_CHARS:
        return []
    if LocatorExtraction._looks_dynamic(text):
        return []
    return [
        Candidate(
            command=f'locator("{tag}").filter('
            f"has_text={LocatorExtraction._quote_locator_value(text)})",
            kind="tag+text",
            stability_score=TAG_TEXT_SCORE,
        )
    ]


def _unscored_attribute_candidates(element: Element) -> list[Candidate]:
    candidates = []
    for attribute, (score, locator) in UNSCORED_ATTRIBUTE_LOCATORS.items():
        value = (element.attributes.get(attribute) or "").strip()
        if value and not LocatorExtraction._looks_dynamic(value):
            quoted = LocatorExtraction._quote_locator_value(value)
            candidates.append(
                Candidate(
                    command=f"{locator}({quoted})",
                    kind=attribute,
                    stability_score=score,
                )
            )

    href = (element.attributes.get("href") or "").strip()
    if element.tag_name == "a" and href and not LocatorExtraction._looks_dynamic(href):
        candidates.append(
            _locator_candidate(
                _css_attribute_selector("a", "href", href), "href", HREF_SCORE
            )
        )
    return candidates


def _name_value_candidates(element: Element) -> list[Candidate]:
    if (element.attributes.get("type") or "").lower() not in GROUPED_INPUT_TYPES:
        return []
    name = (element.attributes.get("name") or "").strip()
    value = (element.attributes.get("value") or "").strip()
    if not name or not value:
        return []
    selector = _css_attribute_selector(element.tag_name or "input", "name", name)
    return [
        _locator_candidate(
            f"{selector}[value='{_escaped(value)}']",
            "name+value",
            NAME_VALUE_SCORE,
        )
    ]


def _functional_type_candidates(element: Element) -> list[Candidate]:
    """A control identified by what it does, optionally scoped to its form."""
    control_type = (element.attributes.get("type") or "").lower()
    if control_type not in FUNCTIONAL_TYPES:
        return []
    selector = _css_attribute_selector(element.tag_name or "*", "type", control_type)
    scoped = [f"form {selector}"] if "/form" in element.xpath else []
    return [
        _locator_candidate(candidate_selector, "type", FUNCTIONAL_TYPE_SCORE)
        for candidate_selector in [*scoped, selector]
    ]


def _anchored_xpath_candidates(element: Element) -> list[Candidate]:
    segments = [segment for segment in element.xpath.split("/") if segment]
    for position in range(len(segments) - 2, -1, -1):
        tag = segments[position].split("[")[0].lower()
        if tag not in XPATH_ANCHOR_TAGS:
            continue
        relative_path = "/".join(segments[position:])
        quoted = LocatorExtraction._quote_locator_value(f"xpath=//{relative_path}", 400)
        return [
            Candidate(
                command=f"locator({quoted})",
                kind="anchored-xpath",
                stability_score=ANCHORED_XPATH_SCORE,
            )
        ]
    return []


def _narrowed_candidates(candidates: list[Candidate]) -> list[Candidate]:
    narrowed = []
    prefix, suffix = 'locator("', '")'
    for candidate in candidates:
        if not candidate.command.startswith(prefix):
            continue
        if not candidate.command.endswith(suffix):
            continue
        selector = candidate.command[len(prefix) : -len(suffix)]
        # A command with a trailing call also ends in "), so the slice can run past
        # the selector into a later argument. Quoting proves it: _quote_locator_value
        # wraps in double quotes, so a selector never contains one.
        if '"' in selector:
            continue
        # css pseudo-classes: meaningless on an xpath selector, pointless twice.
        if selector.startswith("xpath=") or any(
            narrowing in selector for narrowing in NARROWINGS
        ):
            continue
        for narrowing in NARROWINGS:
            narrowed.append(
                Candidate(
                    command=f'locator("{selector}{narrowing}")',
                    kind=f"{candidate.kind} + {narrowing}",
                    stability_score=candidate.stability_score - NARROWING_PENALTY,
                )
            )
    return narrowed


def propose_bundle(candidates: list[Candidate]) -> Candidate | None:
    """The schema stores one command per node, so an or_() chain is the only way
    to persist a fallback. Returned UNPROBED and must be measured: or_() is a set
    union, so two legs each matching one element can match two together, and the
    family rule below makes that more likely by picking unrelated signals."""
    verified = verified_candidates(candidates)
    if len(verified) < 2:
        return None

    # A differently-derived leg, not a variant that breaks on the same change.
    legs: list[Candidate] = []
    families: set[str] = set()
    for candidate in verified:
        family = candidate.kind.split(" +")[0].split(" (")[0]
        family = KIND_FAMILY_ALIASES.get(family, family)
        if family in families:
            continue
        families.add(family)
        legs.append(candidate)
        if len(legs) == MAX_BUNDLED_CANDIDATES:
            break

    if len(legs) < 2:
        return None
    return Candidate(
        command=legs[0].command + f".or_(page.{legs[1].command})",
        kind=f"{legs[0].kind} or {legs[1].kind}",
        stability_score=legs[0].stability_score,
    )


def build_candidates(element: Element) -> list[Candidate]:
    heuristic_candidates = [
        Candidate(command=command, kind=kind, stability_score=score)
        for score, kind, command in LocatorExtraction._scored_candidates(
            _RecordedElementAsNode(element)
        )
    ]

    seen_commands = {candidate.command for candidate in heuristic_candidates}
    candidates = list(heuristic_candidates)
    for extra_candidate in (
        _readmitted_candidates(element)
        + _unscored_attribute_candidates(element)
        + _name_value_candidates(element)
        + _tag_text_candidates(element)
        + _functional_type_candidates(element)
        + _anchored_xpath_candidates(element)
    ):
        if extra_candidate.command not in seen_commands:
            seen_commands.add(extra_candidate.command)
            candidates.append(extra_candidate)

    for narrowed in _narrowed_candidates(candidates):
        if narrowed.command not in seen_commands:
            seen_commands.add(narrowed.command)
            candidates.append(narrowed)

    # _quote_locator_value elides long values, leaving a locator matching nothing.
    candidates = [
        candidate
        for candidate in candidates
        if "..." not in candidate.command and _is_parseable(candidate.command)
    ]
    candidates.sort(key=by_stability, reverse=True)
    return candidates
