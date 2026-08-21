import ast
import re
from types import SimpleNamespace

from optexity.inference.core.interaction.utils import LocatorExtraction
from optexity.memory_layer.trace import Candidate, Element

# _looks_dynamic discards letters-plus-two-digits names like RoboForm's
# 04fullname. It sits on the live LLM-fallback path, so rather than change it
# those values are re-admitted here, one rung lower.
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

# Identifying attributes upstream never looks at, scored into its ladder.
UNSCORED_ATTRIBUTE_LOCATORS = {
    "title": (74, "get_by_title"),
    "alt": (73, "get_by_alt_text"),
}
HREF_SCORE = 60

# A radio group shares one name, so only the value tells its options apart.
GROUPED_INPUT_TYPES = {"radio", "checkbox"}
VALUE_NARROWED_SCORE = 88

XPATH_ANCHOR_TAGS = ("dialog", "form", "table", "nav", "main", "article", "section")
ANCHORED_XPATH_SCORE = 15

# browser-use records an accessible name but no role, so utils.py:225-238's
# implicit-ARIA mapping is recomputed here.
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

# Tried when a selector matches more than one element.
NARROWINGS = (":not([type='hidden'])", ":visible")
NARROWING_PENALTY = 5

# or_() is a union, so bundling is only unambiguous if each leg was separately
# measured at exactly one match.
MAX_BUNDLED_CANDIDATES = 2
KIND_FAMILY_ALIASES = {"css+text": "css", "role+text": "text"}


def implicit_role(element: Element) -> str:
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
    """The subset of a live DOM node _scored_candidates reads. ax_node unlocks
    its role+name rung, the only family independent of id/name/class."""

    def __init__(self, element: Element):
        self.attributes = element.attributes
        self.tag_name = element.tag_name
        self.xpath = element.xpath
        accessible_name = (element.accessible_name or "").strip()
        role = implicit_role(element)
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
    quoted = re.match(r'^locator\((".*")\)$', command, re.DOTALL)
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
        selector = LocatorExtraction._quote_locator_value(
            _css_attribute_selector(tag_name, attribute, value), 400
        )
        readmitted.append(
            Candidate(
                command=f"locator({selector})",
                kind=f"{attribute} (re-admitted)" if rejected else attribute,
                stability_score=score - (READMISSION_PENALTY if rejected else 0),
            )
        )
    return readmitted


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
        selector = LocatorExtraction._quote_locator_value(
            _css_attribute_selector("a", "href", href), 400
        )
        candidates.append(
            Candidate(
                command=f"locator({selector})", kind="href", stability_score=HREF_SCORE
            )
        )
    return candidates


def _value_narrowed_candidates(element: Element) -> list[Candidate]:
    if (element.attributes.get("type") or "").lower() not in GROUPED_INPUT_TYPES:
        return []
    name = (element.attributes.get("name") or "").strip()
    value = (element.attributes.get("value") or "").strip()
    if not name or not value:
        return []
    selector = _css_attribute_selector(element.tag_name or "input", "name", name)
    quoted = LocatorExtraction._quote_locator_value(
        f"{selector}[value='{_escaped(value)}']", 400
    )
    return [
        Candidate(
            command=f"locator({quoted})",
            kind="name+value",
            stability_score=VALUE_NARROWED_SCORE,
        )
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
    for candidate in candidates:
        prefix, suffix = 'locator("', '")'
        if not candidate.command.startswith(prefix):
            continue
        if not candidate.command.endswith(suffix):
            continue
        selector = candidate.command[len(prefix) : -len(suffix)]
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


def bundle_verified_candidates(candidates: list[Candidate]) -> str | None:
    """One command falling through to the next locator if the first stops
    matching. The schema stores one command per node, so an or_() chain is the
    only way to persist a ranked fallback. None while nothing has been probed."""
    verified = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.matches_exactly_one_element
        ),
        key=lambda candidate: candidate.stability_score,
        reverse=True,
    )
    if not verified:
        return None

    # A differently-derived locator, not a variant that breaks on the same change.
    bundled: list[Candidate] = []
    families: set[str] = set()
    for candidate in verified:
        family = candidate.kind.split(" +")[0].split(" (")[0]
        family = KIND_FAMILY_ALIASES.get(family, family)
        if family in families:
            continue
        families.add(family)
        bundled.append(candidate)
        if len(bundled) == MAX_BUNDLED_CANDIDATES:
            break

    return "".join(
        [bundled[0].command]
        + [f".or_(page.{candidate.command})" for candidate in bundled[1:]]
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
        + _value_narrowed_candidates(element)
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
    candidates.sort(key=lambda candidate: candidate.stability_score, reverse=True)
    return candidates
