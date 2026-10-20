from __future__ import annotations

import json

import pandas as pd
import pytest

from va_model_code.decoder_va.filters import (
    TRAIN_EVAL_TEXT_OVERLAP_CONTRACT,
    apply_training_text_overlap_policy,
    exclude_eval_text_overlaps,
    training_text_overlap_contract,
    validate_training_text_overlap_contract,
)


def test_overlap_excludes_all_matching_training_rows_independent_of_labels() -> None:
    train = pd.DataFrame(
        {
            "text": ["KEEP", " Ｈｅｌｌｏ\u00a0WORLD ", "hello\tworld", "Straße", "other"],
            "valence": [0.1, 0.2, 0.9, 0.4, 0.5],
            "dataset_of_origin": ["A", "A", "B", "A", "B"],
        },
        index=[9, 7, 7, 2, 1],
    )
    evaluation = pd.DataFrame(
        {
            "text": ["Hello world", "STRASSE", "hello  world", "unseen"],
            "valence": [0.8, 0.9, 0.8, 0.4],
            "dataset_of_origin": ["C"] * 4,
        },
        index=[30, 20, 10, 0],
    )
    original_train = train.copy(deep=True)
    original_eval = evaluation.copy(deep=True)

    retained, audit = exclude_eval_text_overlaps(train, evaluation)

    pd.testing.assert_frame_equal(retained, train.iloc[[0, 4]])
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(evaluation, original_eval)
    assert audit == {
        "contract": TRAIN_EVAL_TEXT_OVERLAP_CONTRACT,
        "train_rows_before": 5,
        "train_rows_after": 2,
        "excluded_train_rows": 3,
        "eval_rows": 4,
        "overlapping_eval_rows": 3,
        "shared_normalized_texts": 2,
    }
    assert json.loads(json.dumps(audit)) == audit
    retained.loc[9, "text"] = "changed copy"
    pd.testing.assert_frame_equal(train, original_train)
    audit["contract"]["version"] = 99
    assert TRAIN_EVAL_TEXT_OVERLAP_CONTRACT["version"] == 1


def test_overlap_preserves_unique_training_duplicates_and_punctuation() -> None:
    train = pd.DataFrame({"text": ["Hello!", "hello!", "different", "different"]})
    evaluation = pd.DataFrame({"text": ["hello", "not present"]})

    retained, audit = exclude_eval_text_overlaps(train, evaluation)

    pd.testing.assert_frame_equal(retained, train)
    assert retained is not train
    assert audit["excluded_train_rows"] == 0
    assert audit["shared_normalized_texts"] == 0
    assert audit["overlapping_eval_rows"] == 0


def test_keep_overlap_preserves_all_original_rows_and_audits_normalized_matches() -> None:
    """Keep differing labels and duplicate indices while retaining an honest overlap audit."""

    train = pd.DataFrame(
        {
            "text": ["KEEP", " Ｈｅｌｌｏ\u00a0WORLD ", "hello\tworld", "Straße", "other"],
            "valence": [0.1, 0.2, 0.9, 0.4, 0.5],
            "dataset_of_origin": ["A", "A", "B", "A", "B"],
        },
        index=[9, 7, 7, 2, 1],
    )
    evaluation = pd.DataFrame({"text": ["Hello world", "STRASSE", "hello  world", "unseen"]})
    original_train = train.copy(deep=True)
    original_eval = evaluation.copy(deep=True)

    retained, audit = apply_training_text_overlap_policy(train, evaluation, policy="keep")

    pd.testing.assert_frame_equal(retained, original_train)
    pd.testing.assert_frame_equal(evaluation, original_eval)
    assert retained is not train
    assert audit == {
        "contract": {
            "version": 1,
            "normalization": "NFKC+casefold+whitespace-collapse",
            "policy": "keep_all_training_rows",
            "evaluation_rows": "unchanged",
        },
        "train_rows_before": 5,
        "train_rows_after": 5,
        "excluded_train_rows": 0,
        "eval_rows": 4,
        "overlapping_eval_rows": 3,
        "shared_normalized_texts": 2,
    }
    retained.iloc[1, 1] = 0.7
    pd.testing.assert_frame_equal(train, original_train)


@pytest.mark.parametrize("policy", ["exclude", "keep"])
def test_overlap_contract_validates_canonical_metadata_and_returns_independent_copy(policy) -> None:
    """Saved policy provenance must survive JSON without sharing mutable globals."""

    contract = training_text_overlap_contract(policy)
    validated = validate_training_text_overlap_contract(json.loads(json.dumps(contract)))
    assert validated == contract
    validated["policy"] = "changed"
    assert training_text_overlap_contract(policy) == contract


@pytest.mark.parametrize("policy", ["exclude", "keep"])
@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"version": False},
        {"version": 1.0},
        {"policy": "unknown"},
        {"evaluation_rows": "deduplicated"},
        {"normalization": "exact"},
        {"unexpected": "field"},
    ],
)
def test_overlap_contract_rejects_noncanonical_metadata(policy, change) -> None:
    """Prevent permissive numeric equality or unknown fields from hiding changed semantics."""

    contract = {**training_text_overlap_contract(policy), **change}
    with pytest.raises(ValueError):
        validate_training_text_overlap_contract(contract)


@pytest.mark.parametrize("value", [None, {}, "keep", {"version": 1}])
def test_overlap_contract_requires_explicit_complete_metadata(value) -> None:
    with pytest.raises(ValueError):
        validate_training_text_overlap_contract(value)


def test_overlap_unknown_policy_cannot_silently_select_keep() -> None:
    frame = pd.DataFrame({"text": ["same"]})
    with pytest.raises(ValueError):
        apply_training_text_overlap_policy(frame, frame, policy="all")


def test_overlap_empty_strings_are_valid_keys_and_all_overlap_returns_empty() -> None:
    train = pd.DataFrame({"text": ["", " \t\n", "same"]}, index=[5, 4, 3])
    evaluation = pd.DataFrame({"text": ["\u00a0", "SAME"]})

    retained, audit = exclude_eval_text_overlaps(train, evaluation)

    pd.testing.assert_frame_equal(retained, train.iloc[0:0])
    assert audit["excluded_train_rows"] == 3
    assert audit["shared_normalized_texts"] == 2
    assert audit["overlapping_eval_rows"] == 2


@pytest.mark.parametrize("empty_split", ["train", "eval", "both"])
def test_overlap_accepts_empty_frames_with_a_text_column(empty_split: str) -> None:
    train = pd.DataFrame({"text": [] if empty_split in ("train", "both") else ["a"]})
    evaluation = pd.DataFrame({"text": [] if empty_split in ("eval", "both") else ["a"]})

    retained, audit = exclude_eval_text_overlaps(train, evaluation)

    pd.testing.assert_frame_equal(retained, train)
    assert audit["excluded_train_rows"] == 0
    assert audit["shared_normalized_texts"] == 0


@pytest.mark.parametrize("bad_text", [None, float("nan"), pd.NA, 42, b"word"])
@pytest.mark.parametrize("invalid_split", ["train", "eval"])
@pytest.mark.parametrize("policy", ["exclude", "keep"])
def test_overlap_rejects_missing_or_non_string_text(bad_text: object, invalid_split: str, policy: str) -> None:
    valid = pd.DataFrame({"text": ["word"]})
    invalid = pd.DataFrame({"text": ["fine", bad_text]})
    train, evaluation = (invalid, valid) if invalid_split == "train" else (valid, invalid)

    with pytest.raises(ValueError, match="text at row position 1 must be a string"):
        apply_training_text_overlap_policy(train, evaluation, policy=policy)


@pytest.mark.parametrize("invalid_split", ["train", "eval"])
@pytest.mark.parametrize("columns", [["other"], ["text", "text"]])
@pytest.mark.parametrize("policy", ["exclude", "keep"])
def test_overlap_rejects_missing_or_duplicate_text_columns(
    invalid_split: str, columns: list[str], policy: str
) -> None:
    valid = pd.DataFrame({"text": ["word"]})
    invalid = pd.DataFrame(columns=columns)
    train, evaluation = (invalid, valid) if invalid_split == "train" else (valid, invalid)

    with pytest.raises(ValueError, match="must contain exactly one text column"):
        apply_training_text_overlap_policy(train, evaluation, policy=policy)
