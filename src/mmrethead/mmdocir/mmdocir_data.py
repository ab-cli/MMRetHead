"""MMDocIR data loading helpers for page-level retrieval experiments.

The official MMDocIR evaluation files are large. These helpers avoid loading
image binaries by default and work with extracted page images when available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re
from typing import Iterable


@dataclass(frozen=True)
class MMDocIRQuery:
    qid: str
    doc_name: str
    domain: str
    question: str
    answer: str
    question_type: str
    gold_page_ids: list[int]
    layout_mapping: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MMDocIRPage:
    doc_name: str
    domain: str
    page_id: int
    passage_id: str
    image_path: str
    image_file: str | None
    ocr_text: str
    vlm_text: str

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MMDocIRLayout:
    doc_name: str
    domain: str
    layout_id: int
    page_id: int
    layout_type: str
    bbox: list[float]
    page_size: list[float]
    image_path: str
    image_file: str | None
    text: str
    ocr_text: str
    vlm_text: str

    def to_json(self) -> dict:
        return asdict(self)


def normalize_doc_name(doc_name: object) -> str:
    """Normalize annotation/parquet document names for joining.

    MMDocIR annotations keep names such as ``foo.pdf`` while page parquet rows
    commonly use ``foo``. The original name is preserved in outputs; this is
    only used for lookup.
    """
    text = Path(str(doc_name)).name
    return text[:-4] if text.lower().endswith(".pdf") else text


def parse_page_id(value: object) -> int:
    """Parse MMDocIR page ids from ints, numeric strings, or path-like ids."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid page id: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)

    text = str(value)
    if text.isdigit():
        return int(text)

    match = re.search(r"(?:page|_)(\d+)(?:\D*)$", text)
    if match:
        return int(match.group(1))

    trailing = re.search(r"(\d+)$", text)
    if trailing:
        return int(trailing.group(1))

    raise ValueError(f"Could not parse page id from {value!r}")


def load_queries(
    annotations_path: Path,
    max_docs: int | None = None,
    max_questions: int | None = None,
    domains: set[str] | None = None,
) -> list[MMDocIRQuery]:
    """Load page-level QA labels from MMDocIR_annotations.jsonl."""
    queries: list[MMDocIRQuery] = []
    docs_seen = 0

    with annotations_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if max_docs is not None and docs_seen >= max_docs:
                break

            if not line.strip():
                continue

            item = json.loads(line)
            domain = str(item.get("domain", ""))
            if domains is not None and domain not in domains:
                continue

            doc_name = str(item["doc_name"])
            docs_seen += 1

            for question_idx, question in enumerate(item.get("questions", [])):
                if max_questions is not None and len(queries) >= max_questions:
                    return queries

                gold_page_ids = [parse_page_id(page_id) for page_id in question.get("page_id", [])]
                if not gold_page_ids:
                    continue

                queries.append(
                    MMDocIRQuery(
                        qid=f"{doc_name}::q{question_idx}",
                        doc_name=doc_name,
                        domain=domain,
                        question=str(question.get("Q", "")),
                        answer=str(question.get("A", "")),
                        question_type=str(question.get("type", "")),
                        gold_page_ids=gold_page_ids,
                        layout_mapping=list(question.get("layout_mapping") or []),
                    )
                )

    return queries


def _available_columns(parquet_path: Path) -> set[str]:
    pq = _require_pyarrow_parquet()
    return set(pq.ParquetFile(parquet_path).schema_arrow.names)


def _read_pages_metadata(pages_parquet: Path) -> pd.DataFrame:
    pd = _require_pandas()
    available = _available_columns(pages_parquet)
    columns = [
        column
        for column in [
            "doc_name",
            "domain",
            "page_id",
            "passage_id",
            "image_path",
            "ocr_text",
            "vlm_text",
        ]
        if column in available
    ]
    missing = {"doc_name", "image_path"} - set(columns)
    if missing:
        raise ValueError(f"{pages_parquet} is missing required columns: {sorted(missing)}")

    return pd.read_parquet(pages_parquet, columns=columns)


def _read_layouts_metadata(layouts_parquet: Path) -> pd.DataFrame:
    pd = _require_pandas()
    available = _available_columns(layouts_parquet)
    columns = [
        column
        for column in [
            "doc_name",
            "domain",
            "page_id",
            "layout_id",
            "type",
            "bbox",
            "page_size",
            "image_path",
            "text",
            "ocr_text",
            "vlm_text",
        ]
        if column in available
    ]
    missing = {"doc_name", "page_id", "bbox"} - set(columns)
    if missing:
        raise ValueError(f"{layouts_parquet} is missing required columns: {sorted(missing)}")

    return pd.read_parquet(layouts_parquet, columns=columns)


def _resolve_image_file(image_path: str, roots: Iterable[Path]) -> str | None:
    relative = Path(image_path)
    for root in roots:
        candidates = [
            root / relative,
            root / "page_images" / relative,
            root / "layout_images" / relative,
            root / "doc_miscellaneous" / "page_images" / relative,
            root / "doc_miscellaneous" / "layout_images" / relative,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return None


def load_pages_for_docs(
    pages_parquet: Path,
    doc_names: Iterable[str],
    image_roots: Iterable[Path] = (),
) -> dict[str, list[MMDocIRPage]]:
    """Load page metadata for selected documents.

    Image bytes are intentionally not read here. Use extracted page screenshots
    or run materialize_page_images_for_docs first.
    """
    requested_doc_names = [str(doc_name) for doc_name in doc_names]
    requested_doc_name_set = set(requested_doc_names)
    requested_norms = {normalize_doc_name(doc_name) for doc_name in requested_doc_names}
    if not requested_doc_name_set:
        return {}

    frame = _read_pages_metadata(pages_parquet)
    frame = frame[frame["doc_name"].map(normalize_doc_name).isin(requested_norms)]

    pages_by_norm: dict[str, list[MMDocIRPage]] = {doc_name: [] for doc_name in requested_norms}
    roots = [Path(root) for root in image_roots]

    for row in frame.to_dict("records"):
        doc_name = str(row["doc_name"])
        doc_norm = normalize_doc_name(doc_name)
        passage_id = str(row.get("passage_id", row.get("page_id", "")))
        page_id_source = row.get("page_id", passage_id)
        page_id = parse_page_id(page_id_source)
        image_path = str(row.get("image_path", ""))
        pages_by_norm.setdefault(doc_norm, []).append(
            MMDocIRPage(
                doc_name=doc_name,
                domain=str(row.get("domain", "")),
                page_id=page_id,
                passage_id=passage_id,
                image_path=image_path,
                image_file=_resolve_image_file(image_path, roots) if roots else None,
                ocr_text=str(row.get("ocr_text", "") or ""),
                vlm_text=str(row.get("vlm_text", "") or ""),
            )
        )

    for pages in pages_by_norm.values():
        pages.sort(key=lambda page: page.page_id)

    pages_by_doc: dict[str, list[MMDocIRPage]] = {}
    for requested_doc_name in requested_doc_names:
        pages_by_doc[requested_doc_name] = list(pages_by_norm.get(normalize_doc_name(requested_doc_name), []))
    return pages_by_doc


def _coerce_float_list(value: object) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def load_layouts_for_docs(
    layouts_parquet: Path,
    doc_names: Iterable[str],
    image_roots: Iterable[Path] = (),
) -> dict[str, list[MMDocIRLayout]]:
    """Load layout metadata for selected documents."""
    requested_doc_names = [str(doc_name) for doc_name in doc_names]
    requested_norms = {normalize_doc_name(doc_name) for doc_name in requested_doc_names}
    if not requested_doc_names:
        return {}

    frame = _read_layouts_metadata(layouts_parquet)
    frame = frame[frame["doc_name"].map(normalize_doc_name).isin(requested_norms)]

    layouts_by_norm: dict[str, list[MMDocIRLayout]] = {doc_name: [] for doc_name in requested_norms}
    roots = [Path(root) for root in image_roots]

    for row in frame.to_dict("records"):
        doc_name = str(row["doc_name"])
        doc_norm = normalize_doc_name(doc_name)
        layout_id = int(row.get("layout_id", len(layouts_by_norm.setdefault(doc_norm, []))))
        image_path = str(row.get("image_path", ""))
        layouts_by_norm.setdefault(doc_norm, []).append(
            MMDocIRLayout(
                doc_name=doc_name,
                domain=str(row.get("domain", "")),
                layout_id=layout_id,
                page_id=parse_page_id(row.get("page_id", 0)),
                layout_type=str(row.get("type", "")),
                bbox=_coerce_float_list(row.get("bbox")),
                page_size=_coerce_float_list(row.get("page_size")),
                image_path=image_path,
                image_file=_resolve_image_file(image_path, roots) if image_path and roots else None,
                text=str(row.get("text", "") or ""),
                ocr_text=str(row.get("ocr_text", "") or ""),
                vlm_text=str(row.get("vlm_text", "") or ""),
            )
        )

    for layouts in layouts_by_norm.values():
        layouts.sort(key=lambda layout: (layout.page_id, layout.layout_id))

    layouts_by_doc: dict[str, list[MMDocIRLayout]] = {}
    for requested_doc_name in requested_doc_names:
        layouts_by_doc[requested_doc_name] = list(
            layouts_by_norm.get(normalize_doc_name(requested_doc_name), [])
        )
    return layouts_by_doc


def materialize_page_images_for_docs(
    pages_parquet: Path,
    doc_names: Iterable[str],
    output_root: Path,
    overwrite: bool = False,
    batch_size: int = 32,
) -> int:
    """Extract selected page JPEG binaries from MMDocIR_pages.parquet.

    This is a convenience path for smoke subsets when page_images.rar has not
    been extracted. It still scans the parquet file, so extracted images remain
    the preferred setup for larger runs.
    """
    available = _available_columns(pages_parquet)
    required = {"doc_name", "image_path", "image_binary"}
    missing = required - available
    if missing:
        raise ValueError(f"{pages_parquet} is missing required columns: {sorted(missing)}")

    requested_norms = {normalize_doc_name(doc_name) for doc_name in doc_names}
    output_root.mkdir(parents=True, exist_ok=True)

    written = 0
    pq = _require_pyarrow_parquet()
    parquet_file = pq.ParquetFile(pages_parquet)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["doc_name", "image_path", "image_binary"],
    ):
        frame = batch.to_pandas()
        frame = frame[frame["doc_name"].map(normalize_doc_name).isin(requested_norms)]
        for row in frame.to_dict("records"):
            target = output_root / str(row["image_path"])
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(row["image_binary"])
            written += 1

    return written


def materialize_layout_images_for_docs(
    layouts_parquet: Path,
    doc_names: Iterable[str],
    output_root: Path,
    overwrite: bool = False,
    batch_size: int = 32,
) -> int:
    """Extract selected layout image binaries from MMDocIR_layouts.parquet."""
    available = _available_columns(layouts_parquet)
    required = {"doc_name", "image_path", "image_binary"}
    missing = required - available
    if missing:
        raise ValueError(f"{layouts_parquet} is missing required columns: {sorted(missing)}")

    requested_norms = {normalize_doc_name(doc_name) for doc_name in doc_names}
    output_root.mkdir(parents=True, exist_ok=True)

    written = 0
    pq = _require_pyarrow_parquet()
    parquet_file = pq.ParquetFile(layouts_parquet)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["doc_name", "image_path", "image_binary"],
    ):
        frame = batch.to_pandas()
        frame = frame[frame["doc_name"].map(normalize_doc_name).isin(requested_norms)]
        for row in frame.to_dict("records"):
            target = output_root / str(row["image_path"])
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(row["image_binary"])
            written += 1

    return written


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "MMDocIR parquet loading requires pandas. Install MMRetHead with "
            "`python -m pip install -e .` in the target environment."
        ) from exc
    return pd


def _require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "MMDocIR parquet loading requires pyarrow. Install MMRetHead with "
            "`python -m pip install -e .` in the target environment."
        ) from exc
    return pq
