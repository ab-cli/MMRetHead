"""MMDocIR data loading, compact outputs, and metrics."""

from .mmdocir_data import (
    MMDocIRLayout,
    MMDocIRPage,
    MMDocIRQuery,
    load_layouts_for_docs,
    load_pages_for_docs,
    load_queries,
    materialize_layout_images_for_docs,
    materialize_page_images_for_docs,
    write_jsonl,
)

__all__ = [
    "MMDocIRLayout",
    "MMDocIRPage",
    "MMDocIRQuery",
    "load_layouts_for_docs",
    "load_pages_for_docs",
    "load_queries",
    "materialize_layout_images_for_docs",
    "materialize_page_images_for_docs",
    "write_jsonl",
]
