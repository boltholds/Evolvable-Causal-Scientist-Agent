"""RED -> GREEN: digits, comparator operators, identity and complete raw JSON."""
import json
from decimal import Decimal

import pytest

from ecsa.experimental.verbalized_state import (
    VerbalizationMode, verbalize_json, number_words,
)


@pytest.mark.parametrize("literal,expected", [
    ("20.10", "twenty point one zero"),
    ("20.01", "twenty point zero one"),
    ("-0.75", "minus zero point seven five"),
    ("+0.001", "positive zero point zero zero one"),
    ("125.04", "one hundred twenty five point zero four"),
    ("1.2e-3", "one point two times ten to the power of minus three"),
])
def test_semantic_numeric_exact_values(literal,expected):
    assert number_words(literal)==expected


def test_decimal_trailing_zeros_and_digit_word_mode():
    assert number_words("20.10", digits=True) == "two zero decimal point one zero"
    assert number_words("20.01",digits=True) == "two zero decimal point zero one"
    assert number_words("-0.75",digits=True) == "minus zero decimal point seven five"
    assert number_words("7", digits=True) == "seven"
    assert number_words("0007",digits=True)=="zero zero zero seven"
    assert number_words("0007") == "seven"


def test_identity_and_actions_unchanged_numeric_values_and_math_operators_expressed():
    payload = json.dumps({
        "object": {"id":"PLC_007", "uuid":9872, "name":"sensor-17", "identity":"unit-099"},
        "scene": "pressure >= -0.75 bar",
        "readings": {"primary": 20.10, "limit": 20.01},
        "predicate":{"operator": "!=", "active":True},
    },sort_keys=True, separators=(",",":"))
    verbal = verbalize_json(payload, VerbalizationMode.SEMANTIC)
    assert "PLC_007" in verbal and "sensor-17" in verbal
    assert "object.uuid: 9872" in verbal
    assert "greater than or equal to" in verbal
    assert "not equal to" in verbal
    assert "twenty point" in verbal
    assert "predicate.active: true" in verbal


def test_raw_plus_preserves_exact_raw_bytes_and_still_supplies_words():
    raw='{"meter":20.10,"other":20.01,"identifier":"PLC_007","rule":">="}'
    combined=verbalize_json(raw, VerbalizationMode.RAW_PLUS)
    assert raw in combined
    assert "twenty point one zero" in combined
    assert "greater than or equal to" in combined
    assert verbalize_json(raw,VerbalizationMode.RAW)==raw


def test_comparison_operators_and_unary_minus_have_no_collisions():
    for op,phrase in ((">","greater than"),("<","less than"),("==","equal to"),
                      ("!=","not equal to"),("<=","less than or equal to"),
                      (">=","greater than or equal to")):
        raw=json.dumps({"operator":op,"expression":"sensor "+op+" -0.007"})
        verbal=verbalize_json(raw,VerbalizationMode.SEMANTIC)
        assert phrase in verbal and "minus zero point zero zero seven" in verbal
    assert "PLC-007" in verbalize_json('{"identity":"PLC-007"}',VerbalizationMode.SEMANTIC)


def test_bad_inputs_fail_closed():
    with pytest.raises(ValueError):
        verbalize_json("bad JSON",VerbalizationMode.SEMANTIC)
    with pytest.raises(ValueError):
        number_words("1_200")
    with pytest.raises(TypeError):
        verbalize_json('{}',"semantic_words")
    with pytest.raises(ValueError):
        verbalize_json('1',VerbalizationMode.DIGIT)


def test_semantic_contrast_preserves_subtle_differences():
    cases=("-0.01","0.01","0.10","0.01","1e-3","1e-4")
    result=[verbalize_json('{"measurement":'+c+'}',VerbalizationMode.SEMANTIC) for c in cases]
    assert len({result[0],result[1],result[2],result[4],result[5]})==5
    assert result[1]==result[3]
