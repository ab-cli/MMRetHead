"""Ranking metrics for MMDocIR page retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def ranked_page_ids(scores: Mapping[str | int, float]) -> list[int]:
    return [
        int(page_id)
        for page_id, _score in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), int(item[0])),
        )
    ]


def evaluate_page_retrieval(
    predictions: list[dict[str, Any]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float | int]:
    """Compute recall@k and MRR for MMDocIR page retrieval predictions."""
    if not predictions:
        return {"num_queries": 0, "mrr": 0.0, **{f"recall@{k}": 0.0 for k in ks}}

    recall_hits = {k: 0 for k in ks}
    reciprocal_ranks = []

    for prediction in predictions:
        gold = {int(page_id) for page_id in prediction["gold_page_ids"]}
        ranking = ranked_page_ids(prediction["scores"])

        for k in ks:
            recall_hits[k] += int(bool(gold.intersection(ranking[:k])))

        first_rank = 0
        for rank, page_id in enumerate(ranking, start=1):
            if page_id in gold:
                first_rank = rank
                break
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

    num_queries = len(predictions)
    metrics: dict[str, float | int] = {
        "num_queries": num_queries,
        "mrr": sum(reciprocal_ranks) / num_queries,
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_hits[k] / num_queries
    return metrics
