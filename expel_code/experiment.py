"""Supervised experiential VA prompting with matched ablations and outer-fold isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from va_model_code.decoder_va.evaluation import calculate_va_metrics

from .core import ExperienceIndex, InsightBook, json_value, parse_va, records, sample_rows, text_key
from .prompts import insight_messages, prediction_messages, retry_messages


ARMS = ("baseline", "retrieval", "insights", "expel")


def save_json(path: Path, value):
    """Atomically save standards-compliant JSON, replacing undefined metrics with null."""

    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def predict(backend, text, examples=(), insights=()):
    """Predict without a target label, pruning optional context before failing explicitly."""

    examples, insights = list(examples), list(insights)
    while True:
        messages = prediction_messages(text, examples, insights)
        if backend.token_count(messages) + 192 <= backend.context_tokens:
            break
        if examples:
            examples.pop()
        elif insights:
            insights.pop()
        else:
            raise ValueError("Target text exceeds context budget; increase --context-tokens.")
    outputs = [backend.generate(messages, max_new_tokens=48)]
    parsed = None
    for attempt in range(2):
        try:
            parsed = parse_va(outputs[-1])
            break
        except (ValueError, TypeError):
            if attempt == 0:
                repair = messages + [
                    {"role": "assistant", "content": outputs[-1]},
                    {"role": "user", "content": (
                        "Return only valid JSON with exactly valence and arousal keys. "
                        "Both values must be numbers from 0 to 1. No other text."
                    )},
                ]
                outputs.append(backend.generate(repair, max_new_tokens=48))
    return parsed, {
        "outputs": outputs, "valid": parsed is not None,
        "example_ids": [row["row_id"] for row in examples],
        "insight_ids": [row["id"] for row in insights],
        "prompt_tokens": backend.token_count(messages),
    }, messages


def gather(backend, samples, seeds, tolerance, max_retries):
    """Collect training errors and label-assisted retries; mark their provenance explicitly."""

    experiences, successes = [], []
    for index, row in enumerate(samples):
        references = [x for x in seeds if text_key(x["text"]) != text_key(row["text"])]
        initial, initial_audit, messages = predict(backend, row["text"], references)
        truth = (row["valence"], row["arousal"])

        def successful(prediction):
            return prediction is not None and max(abs(a - b) for a, b in zip(prediction, truth)) <= tolerance

        final, retry_prediction, retries = initial, None, []
        for _ in range(max_retries):
            if successful(final):
                break
            previous_output = retries[-1]["output"] if retries else initial_audit["outputs"][-1]
            retry_prompt = retry_messages(messages, previous_output, truth)
            output = backend.generate(retry_prompt, max_new_tokens=48)
            try:
                final = parse_va(output)
            except (ValueError, TypeError):
                final = None
            retry_prediction = final
            retries.append({"output": output, "prediction": final})
        experience = {
            **row, "reference": truth, "initial_prediction": initial,
            "initial_success": successful(initial), "retry_prediction": retry_prediction,
            "final_success": successful(final), "initial_audit": initial_audit,
            "retries": retries, "retry_feedback": "gold_training_scores",
        }
        experiences.append(experience)
        if successful(final):
            successes.append({
                **row, "valence": final[0], "arousal": final[1],
                "memory_origin": "unassisted_success" if successful(initial) else "label_assisted_retry",
            })
        print(f"Experience {index + 1}/{len(samples)}: initial={successful(initial)}, final={successful(final)}", flush=True)
    return experiences, successes


def extract(backend, experiences, max_rules, chunk_size):
    """Iteratively update training-only insights and preserve every rejected response."""

    book, log = InsightBook(max_rules), []
    for start in range(0, len(experiences), chunk_size):
        chunk = experiences[start:start + chunk_size]
        messages = insight_messages(chunk, book.snapshot())
        if backend.token_count(messages) + 256 > backend.context_tokens:
            raise ValueError("Insight batch exceeds context budget; reduce --insight-batch-size.")
        output = backend.generate(messages, max_new_tokens=192)
        before = book.snapshot()
        repair_output = None
        serialization = "model_operations"

        def apply_checked(value):
            payload = json_value(value)
            if isinstance(payload, dict) and isinstance(payload.get("operations"), list):
                for operation in payload["operations"]:
                    if not isinstance(operation, dict):
                        continue
                    rule_text = operation.get("text", "")
                    if isinstance(rule_text, str) and any(
                        len(text_key(row["text"])) >= 20
                        and text_key(row["text"]) in text_key(rule_text)
                        for row in chunk
                    ):
                        raise ValueError("Copied training text is not a general insight.")
            return book.apply(value)

        try:
            operations = apply_checked(output)
            error = None
        except (ValueError, TypeError):
            serialization = "plain_rule_serialized_as_ADD"
            repair_output = backend.generate(insight_messages(chunk, before, plain=True), max_new_tokens=96)
            rule = repair_output.strip()
            try:
                if (
                    not 25 <= len(rule) <= 350 or "\n" in rule
                    or any(mark in rule for mark in ("{", "}", "```"))
                    or not rule.endswith((".", "!", "?"))
                ):
                    raise ValueError("The fallback must contain one short prose rule.")
                operations = apply_checked(json.dumps({"operations": [{"op": "ADD", "text": rule}]}))
                error = None
            except (ValueError, TypeError):
                operations, error = [], "invalid_operations_and_plain_rule"
        log.append({
            "training_row_ids": [x["row_id"] for x in chunk],
            "before": before, "output": output, "operations": operations,
            "repair_output": repair_output, "serialization": serialization,
            "error": error, "after": book.snapshot(),
        })
        print(f"Insights batch {len(log)}: {len(book.rules)} rules, valid={error is None}", flush=True)
    return book.snapshot(), log


def evaluate(backend, queries, seeds, pool, insights, fallback, k):
    """Evaluate identical targets in all arms with explicit failures and fixed fallback."""

    index = ExperienceIndex(pool)
    outputs = []
    for position, query in enumerate(queries):
        neighbors = index.search(query["text"], k)
        fixed = [x for x in seeds if text_key(x["text"]) != text_key(query["text"])][:k]
        for arm in ARMS:
            examples = neighbors if arm in {"retrieval", "expel"} else fixed
            rules = insights if arm in {"insights", "expel"} else []
            prediction, audit, _ = predict(backend, query["text"], examples, rules)
            outputs.append({
                **query, "arm": arm, "prediction": prediction if prediction is not None else list(fallback),
                "used_fallback": prediction is None, "audit": audit,
            })
        neighbor_mean = (
            np.mean([[x["valence"], x["arousal"]] for x in neighbors], axis=0)
            if neighbors else fallback
        )
        for arm, prediction in (("train_mean", fallback), ("retrieval_mean", neighbor_mean)):
            outputs.append({
                **query, "arm": arm, "prediction": list(prediction), "used_fallback": False,
            })
        print(f"Evaluation {position + 1}/{len(queries)}: all four Qwen arms complete", flush=True)
    return outputs


def score_predictions(predictions):
    """Score every row including parse failures, rather than dropping difficult cases."""

    summary = {}
    for arm in (*ARMS, "train_mean", "retrieval_mean"):
        rows = [row for row in predictions if row["arm"] == arm]
        labels = np.asarray([[row["valence"], row["arousal"]] for row in rows])
        values = np.asarray([row["prediction"] for row in rows])
        metrics = calculate_va_metrics(labels, values)
        metrics["fallback_count"] = sum(row["used_fallback"] for row in rows)
        metrics["fallback_rate"] = metrics["fallback_count"] / len(rows)
        summary[arm] = metrics
    return summary


def run_fold(backend, train, evaluation, output_dir, args):
    """Keep each outer fold's examples, rules, audits, and results entirely separate."""

    unique_train = train.drop_duplicates("text_key").reset_index(drop=True)
    seed_frame = sample_rows(unique_train, args.top_k, args.seed)
    seeds = [{**row, "memory_origin": "fixed_gold_demonstration"} for row in records(seed_frame)]
    remaining = unique_train.loc[~unique_train["text_key"].isin(seed_frame["text_key"])].reset_index(drop=True)
    if remaining.empty:
        raise ValueError("Training split is too small after selecting fixed demonstrations.")
    experience_frame = sample_rows(remaining, args.experience_samples, args.seed + 1)
    evaluation_frame = sample_rows(evaluation, args.eval_samples, args.seed + 2)
    save_json(output_dir / "selection.json", {
        "fixed_demonstrations": seeds,
        "experience_row_ids": experience_frame["row_id"].tolist(),
        "evaluation_row_ids": evaluation_frame["row_id"].tolist(),
        "training_rows": len(train), "evaluation_rows_full": len(evaluation),
    })
    experiences, successes = gather(
        backend, records(experience_frame), seeds, args.success_tolerance, args.max_retries,
    )
    save_json(output_dir / "experiences.json", experiences)
    pool = seeds + successes
    save_json(output_dir / "memory.json", pool)
    insights, insight_log = extract(
        backend, experiences, args.max_rules, args.insight_batch_size,
    )
    save_json(output_dir / "insights.json", {"rules": insights, "updates": insight_log})
    fallback = train[["valence", "arousal"]].to_numpy(float).mean(axis=0)
    predictions = evaluate(
        backend, records(evaluation_frame), seeds, pool, insights, fallback, args.top_k,
    )
    save_json(output_dir / "predictions.json", predictions)
    summary = {
        "metrics": score_predictions(predictions),
        "n_experiences": len(experiences), "n_memory": len(pool), "n_rules": len(insights),
        "initial_successes": sum(row["initial_success"] for row in experiences),
        "label_assisted_successes": sum(row["final_success"] and not row["initial_success"] for row in experiences),
        "invalid_insight_batches": sum(row["error"] is not None for row in insight_log),
        "rule_extraction_succeeded": bool(insights),
        "success_tolerance_per_dimension": args.success_tolerance,
    }
    save_json(output_dir / "summary.json", summary)
    return predictions, summary
