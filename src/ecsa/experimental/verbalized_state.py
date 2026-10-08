"""Deterministic verbalization of *full* JSON observations, no learned semantics.

Conversion is used only for experimental text encoder input. Stable IDs and
paths are treated as identifiers; numeric and comparison values are spelled
out with exact decimal digits. RAW_PLUS retains byte-for-byte raw JSON.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum


class VerbalizationMode(StrEnum):
    RAW = "raw"
    SEMANTIC = "semantic_words"
    DIGIT = "digit_words"
    RAW_PLUS = "raw_plus_words"
    NUMERIC_CONTROL = "numeric_control"
    HYBRID_CONTROL = "raw_plus_numeric_control"


_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_TEENS = (
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = ("", "thousand", "million", "billion", "trillion")
_OPS = {
    ">=": "greater than or equal to", "<=": "less than or equal to",
    "!=": "not equal to", "==": "equal to", ">": "greater than",
    "<": "less than", "=": "equals", "*": "multiplied by",
    "/": "divided by", "+": "plus", "−": "minus", "×": "multiplied by",
    "÷": "divided by", "≥": "greater than or equal to",
    "≤": "less than or equal to", "≠": "not equal to",
}
_OP_PATTERN = re.compile(r">=|<=|!=|==|[><=*/+−×÷≥≤≠]")
_NUMBER = re.compile(r"(?<![\w.])(?P<n>[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?)(?![\w.])")
_EXACT_NUMBER = re.compile(r"[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?\Z")
_IDENTIFIERS = frozenset({
    "id", "uuid", "identity", "identifier", "local_ref", "source_identity",
    "ref", "reference", "schema_id", "transition_id", "observation_id",
    "entity_id", "object_id", "timestamp", "date", "name", "path",
})


def _integer_words(value: int) -> str:
    if value == 0:
        return "zero"
    if value < 0:
        return "minus " + _integer_words(-value)
    if value >= 1000 ** len(_SCALES):
        return " ".join(_DIGITS[int(n)] for n in str(value))
    units: list[str] = []
    for magnitude, scale in reversed(tuple(enumerate(_SCALES))):
        amount = (value // (1000 ** magnitude)) % 1000
        if not amount:
            continue
        words: list[str] = []
        if amount >= 100:
            words.append(_DIGITS[amount // 100] + " hundred")
            amount %= 100
        if 10 <= amount < 20:
            words.append(_TEENS[amount - 10])
        elif amount >= 20:
            tens = _TENS[amount // 10]
            words.append(tens + (" " + _DIGITS[amount % 10] if amount % 10 else ""))
        elif amount:
            words.append(_DIGITS[amount])
        if scale:
            words.append(scale)
        units.append(" ".join(words))
    return " ".join(units)


def number_words(number: str, *, digits: bool = False) -> str:
    """Spell exact lexical decimals; 20.10 != 20.1 in word sequence.

    Does not round using binary floating point. Scientific notation is explicit.
    """
    if not isinstance(number, str):
        raise TypeError("string-encoded numeric scalar required")
    value = number.replace("−", "-")
    if not _EXACT_NUMBER.fullmatch(value):
        raise ValueError("not a numeric literal")
    negative = value.startswith("-")
    unsigned = value[1:] if value.startswith(("+", "-")) else value
    exponent: int | None = None
    if "e" in unsigned.lower():
        mantissa, exp = re.split("[eE]", unsigned, maxsplit=1)
        try:
            exponent = int(exp)
        except (ValueError, InvalidOperation) as exc:
            raise ValueError("invalid exponent") from exc
        if abs(exponent) > 9999:
            raise ValueError("exponent too large for verbalization")
        unsigned = mantissa
    whole, dot, fraction = unsigned.partition(".")
    whole = whole or "0"
    before = (
        " ".join(_DIGITS[int(c)] for c in whole)
        if digits else _integer_words(int(whole))
    )
    result = before + (" decimal point " if digits else " point ") + " ".join(
        _DIGITS[int(c)] for c in fraction
    ) if dot else before
    if negative:
        result = "minus " + result
    elif value.startswith("+"):
        result = "positive " + result
    if exponent is not None:
        result += " times ten to the power of " + _integer_words(exponent)
    return result


def _is_identifier(path: tuple[str, ...]) -> bool:
    return bool(path) and path[-1].lower() in _IDENTIFIERS


def _worded_expression(value: str, *, digits: bool) -> str:
    """Rewrite tokens in expressions; never rewrite embedded ID digits."""
    if not value:
        return value
    # First recognize numbers as atomic spans, then operators outside them.
    pieces: list[str] = []
    last = 0
    numeric_spans: list[tuple[int, int]] = []
    for match in _NUMBER.finditer(value):
        numeric_spans.append((match.start(), match.end()))
        segment = value[last:match.start()]
        pieces.append(_replace_operators(segment))
        pieces.append(number_words(match.group("n"), digits=digits))
        last = match.end()
    pieces.append(_replace_operators(value[last:]))
    return "".join(pieces).strip()


def _replace_operators(value: str) -> str:
    # Avoid changing hyphens within IDs or words; arithmetic minus is handled
    # as a sign by numeric spans, and as a word only if visibly separated.
    result = _OP_PATTERN.sub(lambda m: " " + _OPS[m.group()] + " ", value)
    return re.sub(r"(?<=\s)\-(?=\s)", " minus ", result)


def _leaves(node: object, path: tuple[str, ...] = ()):
    if isinstance(node, dict):
        for key in sorted(node):
            yield from _leaves(node[key], (*path, key))
    elif isinstance(node, list):
        for i, child in enumerate(node):
            yield from _leaves(child, (*path, f"item_{i}"))
    else:
        yield path, node


def verbalize_json(raw: str, mode: VerbalizationMode) -> str:
    """Transform canonical JSON without knowing any world-domain feature names.

    Identifier leaves stay unchanged; all measurement numeric leaves and
    comparison/operator strings become English. In RAW_PLUS the *entire*
    original JSON is retained; no precision, metadata or schema is dropped.
    """
    if not isinstance(mode, VerbalizationMode) or not isinstance(raw, str):
        raise TypeError("typed verbalization mode and JSON string required")
    if mode in (VerbalizationMode.RAW, VerbalizationMode.HYBRID_CONTROL,
                VerbalizationMode.NUMERIC_CONTROL):
        return raw
    try:
        parsed = json.loads(raw, parse_float=Decimal, parse_int=Decimal)
    except json.JSONDecodeError as exc:
        raise ValueError("expected canonical JSON document") from exc
    if not isinstance(parsed, (dict, list)):
        raise ValueError("expected JSON object or array")
    lines = []
    for path, value in _leaves(parsed):
        key = ".".join(path)
        if _is_identifier(path):
            worded = str(value)
        elif isinstance(value, Decimal):
            worded = number_words(str(value), digits=mode is VerbalizationMode.DIGIT)
        elif type(value) is bool:
            worded = "true" if value else "false"
        elif value is None:
            worded = "unknown"
        elif isinstance(value, str):
            worded = _worded_expression(value, digits=mode is VerbalizationMode.DIGIT)
        else:
            raise TypeError("unexpected JSON leaf")
        lines.append(f"{key}: {worded}")
    description = "\n".join(lines)
    if mode is VerbalizationMode.RAW_PLUS:
        return "Original JSON:\n" + raw + "\nVerbalized observation:\n" + description
    return description
