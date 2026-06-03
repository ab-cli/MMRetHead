"""Compact output helpers for MMDocIR retriever runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def top_score_items(scores: Mapping[str | int, Any], limit: int = 10) -> list[tuple[str, float]]:
    items = []
    for key, value in scores.items():
        try:
            items.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue
    return sorted(items, key=lambda item: (-item[1], item[0]))[:limit]


def hit_at(ids: list[str], gold_ids: set[str], k: int) -> int:
    return int(bool(gold_ids.intersection(ids[:k])))


def page_topk_rows(predictions: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    rows = []
    for row in predictions:
        scores = row.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        top_items = top_score_items(scores, limit)
        top_ids = [item[0] for item in top_items]
        gold_pages = {str(page_id) for page_id in row.get("gold_page_ids", [])}
        rows.append(
            {
                "qid": row.get("qid", ""),
                "doc_name": row.get("doc_name", ""),
                "domain": row.get("domain", ""),
                "question_type": row.get("question_type", ""),
                "gold_page_ids": " ".join(sorted(gold_pages, key=_numeric_sort_key)),
                "top1_id": top_ids[0] if top_ids else "",
                "top1_score": top_items[0][1] if top_items else "",
                "top5_ids": " ".join(top_ids[:5]),
                "top10_ids": " ".join(top_ids[:10]),
                "hit@1": hit_at(top_ids, gold_pages, 1),
                "hit@5": hit_at(top_ids, gold_pages, 5),
                "hit@10": hit_at(top_ids, gold_pages, 10),
            }
        )
    return rows


def layout_topk_rows(predictions: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    rows = []
    for row in predictions:
        scores = row.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        layout_to_page = {
            str(item.get("layout_id")): str(item.get("page_id"))
            for item in row.get("candidate_layouts", [])
            if isinstance(item, dict)
        }
        top_items = top_score_items(scores, limit)
        top_ids = [item[0] for item in top_items]
        top_pages = [layout_to_page.get(layout_id, "") for layout_id in top_ids]
        gold_pages = {str(page_id) for page_id in row.get("gold_page_ids", [])}
        rows.append(
            {
                "qid": row.get("qid", ""),
                "doc_name": row.get("doc_name", ""),
                "domain": row.get("domain", ""),
                "question_type": row.get("question_type", ""),
                "gold_page_ids": " ".join(sorted(gold_pages, key=_numeric_sort_key)),
                "top1_layout_id": top_ids[0] if top_ids else "",
                "top1_page_id": top_pages[0] if top_pages else "",
                "top1_score": top_items[0][1] if top_items else "",
                "top5_layout_ids": " ".join(top_ids[:5]),
                "top10_layout_ids": " ".join(top_ids[:10]),
                "top5_page_ids": " ".join(top_pages[:5]),
                "top10_page_ids": " ".join(top_pages[:10]),
                "hit@1": hit_at(top_pages, gold_pages, 1),
                "hit@5": hit_at(top_pages, gold_pages, 5),
                "hit@10": hit_at(top_pages, gold_pages, 10),
            }
        )
    return rows


def selected_manifest(
    *,
    total_queries: int,
    doc_names: list[str],
    candidate_count: int,
    candidate_kind: str,
) -> dict[str, Any]:
    return {
        "num_queries": total_queries,
        "num_documents": len(doc_names),
        "documents": doc_names,
        f"num_candidate_{candidate_kind}": candidate_count,
    }


def weighted_metric_average(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row.get("num_queries", 0) or 0) for row in metric_rows)
    keys = [
        key
        for row in metric_rows
        for key, value in row.items()
        if key != "num_queries" and isinstance(value, (int, float))
    ]
    output: dict[str, Any] = {"num_queries": total}
    for key in sorted(set(keys)):
        if total:
            output[key] = sum(float(row.get(key, 0.0) or 0.0) * int(row.get("num_queries", 0) or 0) for row in metric_rows) / total
        else:
            output[key] = 0.0
    return output


def aggregate_domain_metric_rows(
    domain_rows: Iterable[dict[str, Any]],
    metric_func: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in domain_rows:
        grouped[str(row.get("domain", "<missing>"))].append(row)
    output = [{"domain": domain, **metric_func(rows)} for domain, rows in grouped.items()]
    output.sort(key=lambda row: (-int(row.get("num_queries", 0) or 0), str(row["domain"])))
    return output


def concat_csvs(inputs: Iterable[Path], output: Path, add_source_column: bool = True) -> int:
    rows = []
    for path in sorted(inputs):
        if not path.exists():
            continue
        for row in read_csv(path):
            if add_source_column:
                row = {"source": str(path), **row}
            rows.append(row)
    return write_csv(output, rows)


def _numeric_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):08d}") if value.isdigit() else (1, value)
