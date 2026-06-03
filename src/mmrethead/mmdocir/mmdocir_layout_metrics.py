"""Official-style metrics for MMDocIR layout retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DOMAIN_LIST = [
    "Research report / Introduction",
    "Administration/Industry file",
    "Tutorial/Workshop",
    "Academic paper",
    "Brochure",
    "Financial report",
    "Guidebook",
    "Government",
    "Laws",
    "News",
]


def ranked_layout_ids(scores: Mapping[str | int, float]) -> list[int]:
    return [
        int(layout_id)
        for layout_id, _score in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), int(item[0])),
        )
    ]


def overlap_area(bbox1: list[float], bbox2: list[float]) -> float:
    """Match the official MMDocIR bbox convention and overlap computation."""
    top1, left1, bottom1, right1 = bbox1
    top2, left2, bottom2, right2 = bbox2
    inter_top = max(top1, top2)
    inter_left = max(left1, left2)
    inter_bottom = min(bottom1, bottom2)
    inter_right = min(right1, right2)
    if inter_top < inter_bottom and inter_left < inter_right:
        return float((inter_bottom - inter_top) * (inter_right - inter_left))
    return 0.0


def bbox_area(bbox: list[float]) -> float:
    top, left, bottom, right = bbox
    return float(max(0.0, bottom - top) * max(0.0, right - left))


def layout_recall_at_k(
    ranked_ids: list[int],
    candidates_by_id: dict[int, dict[str, Any]],
    layout_mapping: list[dict[str, Any]],
    k: int,
) -> float:
    recall_area = 0.0
    for layout_id in ranked_ids[:k]:
        candidate = candidates_by_id.get(int(layout_id))
        if candidate is None:
            continue
        for gold in layout_mapping:
            if int(candidate["page_id"]) == int(gold["page"]):
                recall_area += overlap_area(list(candidate["bbox"]), list(gold["bbox"]))

    gt_area = sum(bbox_area(list(gold["bbox"])) for gold in layout_mapping)
    if gt_area == 0:
        return 0.0
    return recall_area / gt_area


def evaluate_layout_retrieval(
    predictions: list[dict[str, Any]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    if not predictions:
        return {
            "num_queries": 0,
            **{f"recall@{k}": 0.0 for k in ks},
            "domain_metrics": {},
        }

    totals = {k: 0.0 for k in ks}
    counts_by_domain = {domain: 0 for domain in DOMAIN_LIST}
    totals_by_domain = {
        domain: {k: 0.0 for k in ks}
        for domain in DOMAIN_LIST
    }

    for prediction in predictions:
        ranked_ids = ranked_layout_ids(prediction["scores"])
        candidates = {
            int(candidate["layout_id"]): candidate
            for candidate in prediction["candidate_layouts"]
        }
        layout_mapping = prediction.get("layout_mapping") or []
        domain = prediction["domain"]
        counts_by_domain.setdefault(domain, 0)
        totals_by_domain.setdefault(domain, {k: 0.0 for k in ks})
        counts_by_domain[domain] += 1

        for k in ks:
            score = layout_recall_at_k(ranked_ids, candidates, layout_mapping, k)
            totals[k] += score
            totals_by_domain[domain][k] += score

    num_queries = len(predictions)
    metrics: dict[str, Any] = {
        "num_queries": num_queries,
        **{f"recall@{k}": totals[k] / num_queries for k in ks},
    }
    domain_metrics = {}
    for domain in DOMAIN_LIST:
        count = counts_by_domain.get(domain, 0)
        if count:
            domain_metrics[domain] = {
                "num_queries": count,
                **{
                    f"recall@{k}": totals_by_domain[domain][k] / count
                    for k in ks
                },
            }
        else:
            domain_metrics[domain] = {
                "num_queries": 0,
                **{f"recall@{k}": 0.0 for k in ks},
            }
    metrics["domain_metrics"] = domain_metrics
    metrics["macro_domain"] = {
        f"recall@{k}": sum(domain_metrics[domain][f"recall@{k}"] for domain in DOMAIN_LIST) / len(DOMAIN_LIST)
        for k in ks
    }
    return metrics
