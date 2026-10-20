"""Leakage controls, lexical experience retrieval, parsing, and insight updates."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def text_key(text: str) -> str:
    """Group texts independently of casing, compatibility forms, and whitespace."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(value: object) -> str:
    """Hash a JSON-serializable experiment identity deterministically."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def json_value(text: str):
    """Accept one JSON value, optionally wrapped in a Markdown JSON fence."""

    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(value)


def parse_va(text: str) -> tuple[float, float]:
    """Reject malformed, nonfinite, boolean, and out-of-scale VA predictions."""

    payload = json_value(text)
    if not isinstance(payload, dict) or set(payload) != {"valence", "arousal"}:
        raise ValueError("Expected exactly valence and arousal JSON keys.")
    scores = []
    for name in ("valence", "arousal"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("VA scores must be JSON numbers.")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("VA scores must be finite and within [0, 1].")
        scores.append(float(value))
    return tuple(scores)


def validated_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate normalized labels and stable identities before building memories."""

    required = {"index", "text", "dataset_of_origin", "valence", "arousal"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing data columns: {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    if frame.empty or not frame["text"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Each split needs nonempty text rows.")
    scores = frame[["valence", "arousal"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Labels must be finite and normalized to [0, 1].")
    frame["text_key"] = frame["text"].map(text_key)
    frame["row_id"] = (
        frame["dataset_of_origin"].astype(str) + ":" + frame["index"].astype(str)
    )
    if frame["row_id"].duplicated().any():
        raise ValueError("Duplicate source/index identities within a split.")
    return frame.reset_index(drop=True)


def isolate_train(train: pd.DataFrame, evaluation: pd.DataFrame):
    """Exclude every outer-evaluation text from training before any subsampling."""

    train, evaluation = validated_frame(train), validated_frame(evaluation)
    keep = ~train["text_key"].isin(set(evaluation["text_key"]))
    clean = train.loc[keep].reset_index(drop=True)
    if clean.empty:
        raise ValueError("No training rows remain after evaluation-text exclusion.")
    if set(clean["row_id"]) & set(evaluation["row_id"]):
        raise ValueError("Training and evaluation row identities overlap.")
    return clean, evaluation, {
        "training_rows_before": len(train),
        "training_rows_after": len(clean),
        "excluded_training_rows": int((~keep).sum()),
        "evaluation_rows_full": len(evaluation),
        "normalization": "NFKC+casefold+whitespace-collapse",
    }


def sample_rows(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Sample reproducibly in source-proportional round-robin order without labels."""

    if count < 0:
        raise ValueError("Sample size must be nonnegative; zero means all rows.")
    if count == 0 or count >= len(frame):
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = {
        str(name): rng.permutation(group.index).tolist()
        for name, group in frame.groupby("dataset_of_origin", sort=True)
    }
    selected = []
    allocated = dict.fromkeys(groups, 0)
    while len(selected) < count:
        available = [name for name in groups if allocated[name] < len(groups[name])]
        name = min(available, key=lambda x: (allocated[x] / len(groups[x]), x))
        selected.append(groups[name][allocated[name]])
        allocated[name] += 1
    return frame.loc[selected].reset_index(drop=True)


def records(frame: pd.DataFrame) -> list[dict]:
    """Export only explicit public fields, keeping scores as ordinary floats."""

    return [
        {
            "row_id": str(row.row_id), "text": row.text,
            "source": str(row.dataset_of_origin),
            "valence": float(row.valence), "arousal": float(row.arousal),
        }
        for row in frame.itertuples()
    ]


class ExperienceIndex:
    """Retrieve by text alone; never encode labels or fit on evaluation texts."""

    def __init__(self, examples: Sequence[dict]):
        unique = {}
        for example in examples:
            unique.setdefault(text_key(example["text"]), dict(example))
        self.examples = list(unique.values())
        if not self.examples:
            raise ValueError("The experience pool is empty.")
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), norm="l2")
        self.matrix = self.vectorizer.fit_transform([x["text"] for x in self.examples])

    def search(self, text: str, k: int) -> list[dict]:
        """Return deterministic cosine neighbors, excluding the query text itself."""

        if k <= 0:
            raise ValueError("k must be positive.")
        scores = (self.matrix @ self.vectorizer.transform([text]).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")
        return [
            {**self.examples[i], "similarity": float(scores[i])}
            for i in order if text_key(self.examples[i]["text"]) != text_key(text)
        ][:k]


@dataclass
class Insight:
    """One natural-language rule with stable identity and evidence votes."""

    id: int
    text: str
    votes: int = 2


class InsightBook:
    """Apply bounded ExPeL-style operations without silently replacing all rules."""

    def __init__(self, max_rules: int = 6):
        if max_rules <= 0:
            raise ValueError("max_rules must be positive.")
        self.max_rules = max_rules
        self.rules: list[Insight] = []
        self.next_id = 1

    def snapshot(self) -> list[dict]:
        """Return an independently serializable copy."""

        return [asdict(rule) for rule in self.rules]

    def apply(self, output: str) -> list[dict]:
        """Validate the whole operation batch before atomically updating rules."""

        payload = json_value(output)
        if not isinstance(payload, dict) or set(payload) != {"operations"}:
            raise ValueError("Expected a JSON object containing operations.")
        operations = payload["operations"]
        if not isinstance(operations, list) or len(operations) > 6:
            raise ValueError("Expected at most six insight operations.")
        updated = [Insight(**rule) for rule in self.snapshot()]
        next_id = self.next_id
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("An insight operation must be an object.")
            action = operation.get("op")
            expected = {
                "ADD": {"op", "text"}, "EDIT": {"op", "id", "text"},
                "UPVOTE": {"op", "id"}, "DOWNVOTE": {"op", "id"},
            }
            if action not in expected or set(operation) != expected[action]:
                raise ValueError("Invalid insight operation schema.")
            if action in {"ADD", "EDIT"}:
                text = operation["text"]
                if not isinstance(text, str) or not text.strip() or len(text) > 350:
                    raise ValueError("Rule text must contain 1 to 350 characters.")
            if action == "ADD":
                if any(text_key(x.text) == text_key(text) for x in updated):
                    continue
                if len(updated) < self.max_rules:
                    updated.append(Insight(next_id, text.strip()))
                    next_id += 1
                continue
            rule_id = operation["id"]
            if type(rule_id) is not int:
                raise ValueError("Rule ID must be an integer.")
            rule = next((x for x in updated if x.id == rule_id), None)
            if rule is None:
                raise ValueError("An operation referenced a nonexistent rule.")
            if action == "EDIT":
                rule.text = text.strip()
                rule.votes += 1
            elif action == "UPVOTE":
                rule.votes += 1
            else:
                rule.votes -= 1
                if rule.votes <= 0:
                    updated.remove(rule)
        self.rules, self.next_id = updated, next_id
        return operations
