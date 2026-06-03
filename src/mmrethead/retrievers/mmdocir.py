"""Single-GPU MMDocIR retrievers built from multimodal attention-head scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol
import json

from mmrethead.mmdocir.mmdocir_data import MMDocIRLayout, MMDocIRPage


ScoreAggregation = Literal["sum", "mean"]


class MMDocIRPageRetriever(Protocol):
    """Protocol for page-level MMDocIR scorers."""

    def score_pages(self, query: str, pages: list[MMDocIRPage]) -> dict[int, float]:
        """Return one retrieval score per page id."""
        ...


class MMDocIRLayoutRetriever(Protocol):
    """Protocol for layout-level MMDocIR scorers."""

    def score_layouts(self, query: str, layouts: list[MMDocIRLayout]) -> dict[int, float]:
        """Return one retrieval score per layout id."""
        ...


class TextPageRetriever:
    """Dependency-light lexical baseline for validating page pipelines."""

    def score_pages(self, query: str, pages: list[MMDocIRPage]) -> dict[int, float]:
        query_terms = _query_terms(query)
        scores = {}
        for page in pages:
            text = f"{page.ocr_text}\n{page.vlm_text}".lower()
            scores[page.page_id] = float(sum(1 for term in query_terms if term in text))
        return scores


class TextLayoutRetriever:
    """Dependency-light lexical baseline for validating layout pipelines."""

    def score_layouts(self, query: str, layouts: list[MMDocIRLayout]) -> dict[int, float]:
        query_terms = _query_terms(query)
        scores = {}
        for layout in layouts:
            text = f"{layout.text}\n{layout.ocr_text}\n{layout.vlm_text}".lower()
            scores[layout.layout_id] = float(sum(1 for term in query_terms if term in text))
        return scores


class Qwen3VLMMDocIRPageRetriever:
    """Page retriever built from Qwen3-VL query-to-image attention scores."""

    candidate_label = "page"
    system_prompt = "Relevant page ids:"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        input_max_length: int = 32768,
        generation_max_length: int = 16,
        attn_implementation: str = "flash_attention_2",
        head_score_json: Path | None = None,
        head_top_k: int = 50,
        null_calibration: bool = True,
        score_aggregation: ScoreAggregation = "sum",
    ) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info

            from mmrethead.vlm_model.model_utils import (
                get_inverse_offset_mapping,
                get_span_indices_by_text,
            )
            from mmrethead.vlm_model.qwen3_vl import Qwen3VLModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL retrieval requires the VLM environment with torch, "
                "transformers, qwen_vl_utils, and flash-attn installed."
            ) from exc

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.get_inverse_offset_mapping = get_inverse_offset_mapping
        self.get_span_indices_by_text = get_span_indices_by_text
        self.null_calibration = null_calibration
        self.score_aggregation = score_aggregation
        self.head_ids = load_head_ids(head_score_json, head_top_k)

        self.llm = Qwen3VLModel(
            model_name,
            max_length=input_max_length,
            generation_max_length=generation_max_length,
            do_sample=False,
            use_chat_template=True,
            attn_implementation=attn_implementation,
        )

    @staticmethod
    def _prompt(query: str, pages: list[MMDocIRPage]) -> str:
        page_lines = [f"Candidate page {page.page_id}: <image>" for page in pages]
        return (
            "You are given candidate page screenshots from one document. "
            "Identify which pages contain evidence relevant to the question.\n\n"
            + "\n".join(page_lines)
            + f"\n\nQuestion: {query}"
        )

    def _build_inputs(self, query: str, pages: list[MMDocIRPage]) -> tuple[Any, tuple[int, int], list[tuple[int, int]]]:
        image_list = _require_images(pages, image_label=self.candidate_label)
        text = self._prompt(query, pages)
        messages = self.llm.format_chat(text, image_list, system_prompt=self.system_prompt)
        rendered = self.llm.processor.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
        )
        image_inputs, video_inputs = self.process_vision_info(
            messages,
            image_patch_size=self.llm.processor.image_processor.patch_size,
        )
        inputs = self.llm.processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        expanded_text = self.llm.expand_visual_tokens(rendered, inputs)[0]
        offset_mapping = inputs.pop("offset_mapping")[0]
        inverse_offsets = self.get_inverse_offset_mapping(offset_mapping)
        page_spans = [
            self.llm.get_span_indices_by_image_id(expanded_text, inverse_offsets, image_idx)
            for image_idx in range(len(pages))
        ]
        query_span = self.get_span_indices_by_text(expanded_text, inverse_offsets, f"Question: {query}")
        return inputs, query_span, page_spans

    def _selected_token_scores(self, per_token_scores: Any) -> Any:
        if not self.head_ids:
            return per_token_scores.sum(dim=(0, 1))

        selected = []
        for layer, head in self.head_ids:
            if layer < per_token_scores.shape[0] and head < per_token_scores.shape[1]:
                selected.append(per_token_scores[layer, head])
        if not selected:
            raise ValueError("No selected retrieval heads are valid for this model shape.")
        return self.torch.stack(selected, dim=0).sum(dim=0)

    @staticmethod
    def _token_span_from_char_range(
        inverse_offsets: dict[int, int],
        char_start: int,
        char_end: int,
    ) -> tuple[int, int]:
        token_start = None
        for offset in range(char_start, char_end):
            token_start = inverse_offsets.get(offset)
            if token_start is not None:
                break

        token_end = None
        for offset in range(char_end - 1, char_start - 1, -1):
            token_end = inverse_offsets.get(offset)
            if token_end is not None:
                break

        if token_start is None or token_end is None:
            raise ValueError("Could not map character query span to token span.")
        return token_start, token_end

    def _get_query_span(self, expanded_text: str, inverse_offsets: dict[int, int], query: str) -> tuple[int, int]:
        query_block = f"Question: {query}"
        try:
            return self.get_span_indices_by_text(expanded_text, inverse_offsets, query_block)
        except ValueError:
            query_start = expanded_text.rfind("Question:")
            if query_start == -1:
                raise
            query_end_candidates = [
                idx
                for marker in [self.system_prompt, "Relevant page ids:", "Relevant layout ids:", "<end_of_turn>", "<|im_end|>"]
                for idx in [expanded_text.find(marker, query_start)]
                if idx != -1
            ]
            query_end = min(query_end_candidates) if query_end_candidates else len(expanded_text)
            return self._token_span_from_char_range(inverse_offsets, query_start, query_end)

    def _candidate_scores(
        self,
        query: str,
        candidates: list[Any],
        score_ids: list[int],
    ) -> dict[int, float]:
        inputs, query_span, candidate_spans = self._build_inputs(query, candidates)
        per_token_scores, _argmax_idx, _argmax_val, cache = self.llm._compute_per_token_scores(inputs, query_span)
        del cache

        if self.null_calibration:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            null_inputs, null_query_span, _null_spans = self._build_inputs("N/A", candidates)
            null_scores, _idx, _val, null_cache = self.llm._compute_per_token_scores(null_inputs, null_query_span)
            del null_cache
            min_len = min(per_token_scores.shape[-1], null_scores.shape[-1])
            per_token_scores = per_token_scores[:, :, :min_len] - null_scores[:, :, :min_len]

        token_scores = self._selected_token_scores(per_token_scores)
        results = {}
        for score_id, (start, end) in zip(score_ids, candidate_spans):
            span_scores = token_scores[start : min(end + 1, token_scores.shape[-1])]
            score = span_scores.mean() if self.score_aggregation == "mean" else span_scores.sum()
            results[score_id] = float(score.detach().cpu())
        return results

    def score_pages(self, query: str, pages: list[MMDocIRPage]) -> dict[int, float]:
        if not pages:
            return {}
        return self._candidate_scores(query, pages, [page.page_id for page in pages])


class Gemma3MMDocIRPageRetriever(Qwen3VLMMDocIRPageRetriever):
    """Page retriever built from Gemma 3 query-to-image attention scores."""

    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",
        *,
        input_max_length: int = 32768,
        generation_max_length: int = 16,
        attn_implementation: str = "flash_attention_2",
        head_score_json: Path | None = None,
        head_top_k: int = 50,
        null_calibration: bool = True,
        score_aggregation: ScoreAggregation = "sum",
    ) -> None:
        try:
            import torch

            from mmrethead.vlm_model.gemma3 import Gemma3VLModel
            from mmrethead.vlm_model.model_utils import (
                get_inverse_offset_mapping,
                get_span_indices_by_text,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Gemma 3 retrieval requires the VLM environment with torch, "
                "transformers, and flash-attn installed."
            ) from exc

        self.torch = torch
        self.process_vision_info = None
        self.get_inverse_offset_mapping = get_inverse_offset_mapping
        self.get_span_indices_by_text = get_span_indices_by_text
        self.null_calibration = null_calibration
        self.score_aggregation = score_aggregation
        self.head_ids = load_head_ids(head_score_json, head_top_k)

        self.llm = Gemma3VLModel(
            model_name,
            max_length=input_max_length,
            generation_max_length=generation_max_length,
            do_sample=False,
            use_chat_template=True,
            attn_implementation=attn_implementation,
        )

    def _build_inputs(self, query: str, pages: list[MMDocIRPage]) -> tuple[Any, tuple[int, int], list[tuple[int, int]]]:
        image_list = _require_images(pages, image_label=self.candidate_label)
        text = self._prompt(query, pages)
        messages = self.llm.format_chat(text, image_list, system_prompt=self.system_prompt)
        inputs = self.llm.processor.apply_chat_template(
            messages,
            tokenize=True,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        expanded_text = self.llm.processor.tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=False,
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        inverse_offsets = self.get_inverse_offset_mapping(offset_mapping)
        page_spans = [
            self.llm.get_span_indices_by_image_id(expanded_text, inverse_offsets, image_idx)
            for image_idx in range(len(pages))
        ]
        query_span = self._get_query_span(expanded_text, inverse_offsets, query)
        return inputs, query_span, page_spans


class Qwen3VLMMDocIRLayoutRetriever(Qwen3VLMMDocIRPageRetriever):
    """Layout-crop retriever built from Qwen3-VL query-to-image attention scores."""

    candidate_label = "layout"
    system_prompt = "Relevant layout ids:"

    @staticmethod
    def _prompt(query: str, layouts: list[MMDocIRLayout]) -> str:
        layout_lines = [
            f"Candidate layout {layout.layout_id} on page {layout.page_id}: <image>"
            for layout in layouts
        ]
        return (
            "You are given candidate layout crops from one document. "
            "Identify which layouts contain evidence relevant to the question.\n\n"
            + "\n".join(layout_lines)
            + f"\n\nQuestion: {query}"
        )

    def score_layouts(self, query: str, layouts: list[MMDocIRLayout]) -> dict[int, float]:
        if not layouts:
            return {}
        return self._candidate_scores(query, layouts, [layout.layout_id for layout in layouts])


class Gemma3MMDocIRLayoutRetriever(Gemma3MMDocIRPageRetriever):
    """Layout-crop retriever built from Gemma 3 query-to-image attention scores."""

    candidate_label = "layout"
    system_prompt = "Relevant layout ids:"
    _prompt = staticmethod(Qwen3VLMMDocIRLayoutRetriever._prompt)

    def score_layouts(self, query: str, layouts: list[MMDocIRLayout]) -> dict[int, float]:
        if not layouts:
            return {}
        return self._candidate_scores(query, layouts, [layout.layout_id for layout in layouts])


def load_head_ids(head_score_json: Path | None, top_k: int) -> list[tuple[int, int]]:
    """Load `(layer, head)` tuples from a detection `head_score_list` JSON."""
    if head_score_json is None:
        return []
    payload = json.loads(Path(head_score_json).read_text(encoding="utf-8"))
    head_ids = []
    for head_id, _score in payload["head_score_list"][:top_k]:
        layer, head = str(head_id).split("-")
        head_ids.append((int(layer), int(head)))
    return head_ids


def make_page_retriever(
    scorer: Literal["attention", "text_baseline"],
    model_name: str,
    **kwargs: Any,
) -> MMDocIRPageRetriever:
    """Build a page retriever from CLI-style options."""
    if scorer == "text_baseline":
        return TextPageRetriever()
    if "gemma-3" in model_name.lower():
        return Gemma3MMDocIRPageRetriever(model_name=model_name, **kwargs)
    return Qwen3VLMMDocIRPageRetriever(model_name=model_name, **kwargs)


def make_layout_retriever(
    scorer: Literal["attention", "text_baseline"],
    model_name: str,
    **kwargs: Any,
) -> MMDocIRLayoutRetriever:
    """Build a layout retriever from CLI-style options."""
    if scorer == "text_baseline":
        return TextLayoutRetriever()
    if "gemma-3" in model_name.lower():
        return Gemma3MMDocIRLayoutRetriever(model_name=model_name, **kwargs)
    return Qwen3VLMMDocIRLayoutRetriever(model_name=model_name, **kwargs)


def _query_terms(query: str) -> set[str]:
    return {
        token.lower()
        for token in query.replace("/", " ").replace("-", " ").split()
        if len(token) > 2
    }


def _require_images(candidates: list[Any], *, image_label: str) -> list[str]:
    image_list = [candidate.image_file for candidate in candidates]
    if any(image_file is None for image_file in image_list):
        missing = [candidate.image_path for candidate in candidates if candidate.image_file is None]
        raise FileNotFoundError(
            f"Missing {image_label} image files. Provide --image-root or use --materialize-images. "
            f"First missing paths: {missing[:3]}"
        )
    return [str(image_file) for image_file in image_list if image_file is not None]
