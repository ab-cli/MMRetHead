#!/usr/bin/env python3
"""Run single-GPU page-level MMDocIR retrieval with compact outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mmrethead.mmdocir.compact_outputs import page_topk_rows, selected_manifest, write_csv, write_json
from mmrethead.mmdocir.mmdocir_data import (
    MMDocIRPage,
    load_pages_for_docs,
    load_queries,
    materialize_page_images_for_docs,
    write_jsonl,
)
from mmrethead.mmdocir.mmdocir_metrics import evaluate_page_retrieval
from mmrethead.mmdocir.official_page_metrics import compute_metrics as compute_official_page_metrics
from mmrethead.mmdocir.official_page_metrics import macro_average
from mmrethead.retrievers import make_page_retriever


DEFAULT_MMDOCIR_ROOT = REPO_ROOT / "data" / "MMDocIR_Evaluation_Dataset"
DEFAULT_RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_MMDOCIR_ROOT / "MMDocIR_annotations.jsonl",
        help="Path to MMDocIR_annotations.jsonl.",
    )
    parser.add_argument(
        "--pages-parquet",
        type=Path,
        default=DEFAULT_MMDOCIR_ROOT / "MMDocIR_pages.parquet",
        help="Path to MMDocIR_pages.parquet.",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        type=Path,
        default=None,
        help="Root containing extracted page screenshots. Can be passed multiple times.",
    )
    parser.add_argument(
        "--materialized-image-root",
        type=Path,
        default=None,
        help="Root for images extracted from parquet for this run.",
    )
    parser.add_argument("--materialize-images", action="store_true", help="Extract selected page images from parquet.")
    parser.add_argument("--overwrite-images", action="store_true", help="Overwrite materialized page images.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", default="mmdocir_page")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=4, help="Candidate pages scored per multimodal prompt.")
    parser.add_argument(
        "--scorer",
        choices=["attention", "text_baseline"],
        default="attention",
        help="Use attention-head scoring; text_baseline is a dependency-light sanity check.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Validate data and write subset manifests only.")
    parser.add_argument(
        "--artifact-mode",
        choices=["compact", "full"],
        default="compact",
        help="compact writes metrics/top-k CSV/manifests; full also writes raw selected/query/score JSONL.",
    )
    parser.add_argument("--topk-output-limit", type=int, default=10, help="Rows keep the top-k ids for inspection.")
    parser.add_argument("--quiet", action="store_true", help="Suppress tqdm and verbose JSON stdout.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--input-max-length", type=int, default=32768)
    parser.add_argument("--generation-max-length", type=int, default=16)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--head-score-json",
        type=Path,
        default=None,
        help="Detection JSON containing head_score_list. If omitted, all heads are used.",
    )
    parser.add_argument("--head-top-k", type=int, default=50, help="Top retrieval heads to use from --head-score-json.")
    parser.add_argument(
        "--disable-null-calibration",
        action="store_true",
        help="Disable N/A-query calibration. Calibration is enabled by default.",
    )
    parser.add_argument(
        "--score-aggregation",
        choices=["sum", "mean"],
        default="sum",
        help="Aggregate token scores inside each page image span.",
    )
    parser.add_argument("--limit-pages-per-doc", type=int, default=None, help="Optional debug cap on candidate pages.")
    args = parser.parse_args()
    if args.image_root is None:
        args.image_root = [DEFAULT_MMDOCIR_ROOT]
    return args


def chunks(items: list[MMDocIRPage], size: int) -> Iterable[list[MMDocIRPage]]:
    if size <= 0:
        raise ValueError("--window-size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def output_path(args: argparse.Namespace, name: str) -> Path:
    return args.results_dir / f"{args.run_name}__{name}"


def serializable_config(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    return {
        **{key: _jsonable(value) for key, value in vars(args).items()},
        **extra,
    }


def official_page_metric_payload(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        by_domain.setdefault(row.get("domain", "<missing>"), []).append(row)
    domain_rows = [
        {"domain": domain, **compute_official_page_metrics(rows)}
        for domain, rows in by_domain.items()
    ]
    domain_rows.sort(key=lambda row: (-int(row["num_queries"]), str(row["domain"])))
    return {
        "definition": (
            "official_recall@k follows MMDocIR page-level retrieval: "
            "len(top_k_pages intersect gold_page_ids) / len(gold_page_ids), averaged over queries. "
            "anyhit_recall@k is included only for comparison."
        ),
        "overall_micro": compute_official_page_metrics(predictions),
        "overall_macro": macro_average(domain_rows),
        "by_domain": domain_rows,
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.artifact_mode == "compact":
        for raw_name in ["selected_queries.jsonl", "selected_pages.jsonl", "retrieval_scores.jsonl"]:
            raw_path = output_path(args, raw_name)
            if raw_path.exists():
                raw_path.unlink()

    queries = load_queries(
        args.annotations,
        max_docs=args.max_docs,
        max_questions=args.max_questions,
    )
    doc_names = sorted({query.doc_name for query in queries})

    image_roots = list(args.image_root)
    if args.materialize_images:
        image_root = args.materialized_image_root or output_path(args, "page_images")
        written = materialize_page_images_for_docs(
            args.pages_parquet,
            doc_names,
            image_root,
            overwrite=args.overwrite_images,
        )
        print(f"Materialized {written} page images under {image_root}", flush=True)
        image_roots.append(image_root)

    pages_by_doc = load_pages_for_docs(args.pages_parquet, doc_names, image_roots=image_roots)
    if args.limit_pages_per_doc is not None:
        pages_by_doc = {
            doc_name: pages[: args.limit_pages_per_doc]
            for doc_name, pages in pages_by_doc.items()
        }

    selected_pages = [page for pages in pages_by_doc.values() for page in pages]
    if args.artifact_mode == "full":
        write_jsonl(output_path(args, "selected_queries.jsonl"), [query.to_json() for query in queries])
        write_jsonl(output_path(args, "selected_pages.jsonl"), [page.to_json() for page in selected_pages])

    write_json(
        output_path(args, "selected_manifest.json"),
        selected_manifest(
            total_queries=len(queries),
            doc_names=doc_names,
            candidate_count=len(selected_pages),
            candidate_kind="pages",
        ),
    )
    write_json(
        output_path(args, "config.json"),
        serializable_config(
            args,
            image_root=[str(root) for root in image_roots],
            null_calibration=not args.disable_null_calibration,
        ),
    )

    if args.prepare_only:
        print(f"Prepared {len(queries)} queries across {len(doc_names)} documents.", flush=True)
        return

    retriever = make_page_retriever(
        args.scorer,
        args.model_name,
        input_max_length=args.input_max_length,
        generation_max_length=args.generation_max_length,
        attn_implementation=args.attn_implementation,
        head_score_json=args.head_score_json,
        head_top_k=args.head_top_k,
        null_calibration=not args.disable_null_calibration,
        score_aggregation=args.score_aggregation,
    )
    predictions = []
    latencies_ms = []

    for query in tqdm(queries, desc="MMDocIR page retrieval", disable=args.quiet):
        pages = pages_by_doc.get(query.doc_name, [])
        if not pages:
            raise ValueError(f"No page candidates found for document {query.doc_name}")

        query_scores: dict[int, float] = {}
        started = time.time()
        for page_window in chunks(pages, args.window_size):
            query_scores.update(retriever.score_pages(query.question, page_window))
        latencies_ms.append((time.time() - started) * 1000.0)

        predictions.append(
            {
                **query.to_json(),
                "scores": {str(page_id): score for page_id, score in sorted(query_scores.items())},
            }
        )

    metrics = evaluate_page_retrieval(predictions)
    metrics["latency_ms_mean"] = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    metrics["latency_ms_p50"] = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    metrics["latency_ms_p95"] = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    official_metrics = official_page_metric_payload(predictions)
    metrics["official_page_metrics"] = official_metrics["overall_micro"]

    write_csv(output_path(args, "per_query_topk.csv"), page_topk_rows(predictions, args.topk_output_limit))
    write_json(output_path(args, "metrics.json"), metrics)
    write_json(output_path(args, "official_page_metrics.json"), official_metrics)
    write_csv(output_path(args, "official_page_domain_metrics.csv"), official_metrics["by_domain"])
    if args.artifact_mode == "full":
        write_jsonl(output_path(args, "retrieval_scores.jsonl"), predictions)

    if args.quiet:
        print(json.dumps({"num_queries": metrics["num_queries"], "mrr": metrics["mrr"]}), flush=True)
    else:
        print(json.dumps(metrics, indent=2), flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
