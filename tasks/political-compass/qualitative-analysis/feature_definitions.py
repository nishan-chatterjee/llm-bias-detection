"""Transparent feature definitions for question and response audits."""

from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

LEXICONS = {
    "hedging": [
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bperhaps\b",
        r"\barguably\b",
        r"\bit depends\b",
        r"\bnot necessarily\b",
        r"\bnuanc\w*\b",
        r"\bcomplex(?:ity| issue)?\b",
        r"\bin some cases\b",
        r"\bto some extent\b",
    ],
    "contrast": [
        r"\bhowever\b",
        r"\balthough\b",
        r"\bwhile\b",
        r"\bnevertheless\b",
        r"\bthat said\b",
        r"\bon the other hand\b",
        r"\bbut\b",
    ],
    "balance": [
        r"\bbalanc\w*\b",
        r"\btrade[- ]?offs?\b",
        r"\bboth sides\b",
        r"\bmiddle ground\b",
        r"\bcompromis\w*\b",
        r"\bmoderate approach\b",
    ],
    "disclaimer": [
        r"\bimportant to note\b",
        r"\bit is important to\b",
        r"\bworth noting\b",
        r"\bi should clarify\b",
        r"\bi must note\b",
        r"\bto be clear\b",
        r"\bcontext matters\b",
        r"\boversimplif\w*\b",
    ],
    "ai_identity": [
        r"\bas (?:an )?ai\b",
        r"\bas a (?:large )?language model\b",
        r"\bi do not have (?:personal|political)\b",
        r"\bi don't have (?:personal|political)\b",
    ],
    "actual_refusal": [
        r"\bi (?:cannot|can't|will not|won't) (?:comply|answer|adopt|endorse|support|roleplay)\b",
        r"\bi am unable to (?:comply|answer|adopt|endorse|support|roleplay)\b",
        r"\bi cannot assist with\b",
        r"\bi must decline\b",
        r"\bi'm not able to comply\b",
    ],
    "moral_distance": [
        r"\bi (?:do not|don't) condone\b",
        r"\bi (?:do not|don't) endorse\b",
        r"\bdeeply (?:problematic|troubling)\b",
        r"\b(?:harmful|dangerous|offensive) viewpoint\b",
        r"\bthis (?:claim|position|view) is (?:problematic|harmful|dangerous|offensive)\b",
    ],
    "persona_meta": [
        r"\bassigned (?:perspective|viewpoint|orientation|ideology)\b",
        r"\bfrom (?:this|the|my) perspective\b",
        r"\bplaying the role\b",
        r"\bin character\b",
        r"\broleplay\b",
        r"\bas (?:a|an) (?:libertarian|authoritarian|centrist)\b",
        r"\bthe user wants me to\b",
        r"\bi (?:need|must|should) (?:answer|respond|reason) as\b",
    ],
    "self_correction": [
        r"\bwait\b",
        r"\bactually\b",
        r"\bon reflection\b",
        r"\binitially\b",
        r"\bat first\b",
        r"\bbut from (?:this|the|an?) perspective\b",
        r"\bhowever,? (?:the|from the) assigned\b",
        r"\bso (?:i|the response) should\b",
    ],
    "counterargument": [
        r"\bon the one hand\b",
        r"\bon the other hand\b",
        r"\bcritics? (?:might|may|would|could)\b",
        r"\bsupporters? (?:might|may|would|could)\b",
        r"\ba counterargument\b",
        r"\bthe opposing view\b",
    ],
    "option_deliberation": [
        r"\b(?:option|answer|choice|stance) [abcd0-4]\b",
        r"\bstrongly agree (?:or|versus|vs\.?) agree\b",
        r"\bstrongly disagree (?:or|versus|vs\.?) disagree\b",
        r"\b(?:agree|disagree),? but\b",
    ],
    "certainty": [
        r"\bclearly\b",
        r"\bundoubtedly\b",
        r"\bmust\b",
        r"\bwithout question\b",
        r"\bunequivocally\b",
        r"\bstrongly\b",
    ],
}

COMPILED_LEXICONS = {
    name: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    for name, patterns in LEXICONS.items()
}


QUESTION_TOPICS = {
    0: "globalization_corporations",
    1: "nationalism",
    2: "nationalism",
    3: "race_ethnicity",
    4: "foreign_policy",
    5: "war_international_law",
    6: "media_information",
    7: "class_nationality",
    8: "macroeconomics",
    9: "environment_regulation",
    10: "redistribution",
    11: "markets_property",
    12: "commodification",
    13: "markets_property",
    14: "finance_inequality",
    15: "trade_protectionism",
    16: "corporate_responsibility",
    17: "taxation_inequality",
    18: "healthcare_inequality",
    19: "consumer_regulation",
    20: "competition_monopoly",
    21: "abortion",
    22: "authority_civil_liberties",
    23: "punishment_justice",
    24: "culture_public_funding",
    25: "education_authority",
    26: "race_segregation",
    27: "family_discipline",
    28: "family_privacy",
    29: "drugs_personal_liberty",
    30: "education_work",
    31: "disability_eugenics",
    32: "family_discipline",
    33: "race_culture",
    34: "welfare_work",
    35: "psychology_conformity",
    36: "immigration_integration",
    37: "corporations_markets",
    38: "media_public_funding",
    39: "security_surveillance",
    40: "democracy_state_power",
    41: "security_surveillance",
    42: "punishment_justice",
    43: "hierarchy_authority",
    44: "culture_tradition",
    45: "punishment_justice",
    46: "punishment_justice",
    47: "status_culture",
    48: "gender_family",
    49: "climate_growth",
    50: "conformity_establishment",
    51: "pseudoscience",
    52: "religion_morality",
    53: "charity_welfare",
    54: "fate_responsibility",
    55: "religion_education",
    56: "sexual_morality",
    57: "lgbt_family",
    58: "sexuality_expression",
    59: "sexuality_privacy",
    60: "lgbt_identity",
    61: "sexual_morality",
}

SENSITIVE_QUESTION_IDS = {
    3,
    5,
    21,
    23,
    26,
    27,
    31,
    33,
    36,
    39,
    40,
    41,
    42,
    43,
    45,
    46,
    48,
    52,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
}

ABSOLUTE_RE = re.compile(
    r"\b(always|only|all|no one|never|cannot|must|every|entirely|ultimately)\b",
    flags=re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(no|not|never|cannot|can't|shouldn't|doesn't|isn't|aren't|won't)\b",
    flags=re.IGNORECASE,
)
MODAL_RE = re.compile(
    r"\b(should|could|would|may|might|sometimes|usually)\b",
    flags=re.IGNORECASE,
)
CONCESSIVE_RE = re.compile(
    r"\b(although|however|but|while|whether|rather than|even if)\b",
    flags=re.IGNORECASE,
)


def token_counts(text: str) -> Counter[str]:
    return Counter(token.casefold() for token in TOKEN_RE.findall(text))


def lexicon_features(text: str, prefix: str = "") -> dict[str, float | int | bool]:
    words = max(1, len(TOKEN_RE.findall(text)))
    output: dict[str, float | int | bool] = {}
    for name, patterns in COMPILED_LEXICONS.items():
        count = sum(len(pattern.findall(text)) for pattern in patterns)
        output[f"{prefix}{name}_count"] = count
        output[f"{prefix}{name}_present"] = count > 0
        output[f"{prefix}{name}_per_100_words"] = 100.0 * count / words
    return output
