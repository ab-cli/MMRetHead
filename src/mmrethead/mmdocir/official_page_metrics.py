#!/usr/bin/env python3
"""Compute MMDocIR-style page retrieval metrics from retriever scores."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .mmdocir_metrics import ranked_page_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path, help="retrieval_scores.jsonl or aggregate JSONL.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional domain metrics CSV.")
    return parser.parse_args()


def load_scores(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def official_recall_at_k(row: dict[str, Any], k: int) -> float:
    """MMDocIR page recall: fraction of gold pages recovered in top-k."""
    gold = {int(page_id) for page_id in row["gold_page_ids"]}
    if not gold:
        return 0.0
    retrieved = set(ranked_page_ids(row["scores"])[:k])
    return len(gold.intersection(retrieved)) / len(gold)


def any_hit_at_k(row: dict[str, Any], k: int) -> float:
    """Legacy sanity metric: one point if any gold page is in top-k."""
    gold = {int(page_id) for page_id in row["gold_page_ids"]}
    retrieved = set(ranked_page_ids(row["scores"])[:k])
    return float(bool(gold.intersection(retrieved)))


def reciprocal_rank(row: dict[str, Any]) -> float:
    gold = {int(page_id) for page_id in row["gold_page_ids"]}
    for rank, page_id in enumerate(ranked_page_ids(row["scores"]), start=1):
        if page_id in gold:
            return 1.0 / rank
    return 0.0


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "num_queries": 0,
            "mrr": 0.0,
            **{f"official_recall@{k}": 0.0 for k in (1, 3, 5, 10)},
            **{f"anyhit_recall@{k}": 0.0 for k in (1, 3, 5, 10)},
        }

    metrics: dict[str, float | int] = {"num_queries": len(rows)}
    for k in (1, 3, 5, 10):
        metrics[f"official_recall@{k}"] = sum(official_recall_at_k(row, k) for row in rows) / len(rows)
        metrics[f"anyhit_recall@{k}"] = sum(any_hit_at_k(row, k) for row in rows) / len(rows)
    metrics["mrr"] = sum(reciprocal_rank(row) for row in rows) / len(rows)
    return metrics


def macro_average(domain_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not domain_rows:
        return {}
    metric_keys = [key for key in domain_rows[0] if key not in {"domain", "num_queries"}]
    return {
        key: sum(float(row[key]) for row in domain_rows) / len(domain_rows)
        for key in metric_keys
    }


def main() -> None:
    args = parse_args()
    rows = load_scores(args.scores)

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[row.get("domain", "<missing>")].append(row)

    domain_rows = [
        {"domain": domain, **compute_metrics(domain_predictions)}
        for domain, domain_predictions in by_domain.items()
    ]
    domain_rows.sort(key=lambda row: (-int(row["num_queries"]), str(row["domain"])))

    output = {
        "scores": str(args.scores),
        "definition": (
            "official_recall@k follows MMDocIR page-level retrieval: "
            "len(top_k_pages intersect gold_page_ids) / len(gold_page_ids), averaged over queries. "
            "anyhit_recall@k is included only for comparison with earlier smoke checks."
        ),
        "overall_micro": compute_metrics(rows),
        "overall_macro": macro_average(domain_rows),
        "by_domain": domain_rows,
    }

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.write_text(text, encoding="utf-8")
    else:
        print(text)

    if args.output_csv:
        fields = [
            "domain",
            "num_queries",
            "official_recall@1",
            "official_recall@3",
            "official_recall@5",
            "official_recall@10",
            "anyhit_recall@1",
            "anyhit_recall@3",
            "anyhit_recall@5",
            "anyhit_recall@10",
            "mrr",
        ]
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in domain_rows:
                writer.writerow({field: row[field] for field in fields})


if __name__ == "__main__":
    main()
