# Causal Masking

This folder contains the original project causal masking runners copied from
`MMRetrievalHead`:

```text
needle_in_haystack_with_mask_text_needle.py
needle_in_haystack_with_mask_image_needle.py
```

Only the imports were adapted so the scripts use the `MMRetHead/src/mmrethead`
package layout. The masking loop, `--mask_topk` convention, context/depth grid
arguments, and nested result layout follow the original scripts.

Use `needle_in_haystack_with_mask_text_needle.py` for text-needle tasks such as
MM-NIAH text retrieval, rendered text-needle images, NQ, and DocQA-style
standard VLM prompts. Use `needle_in_haystack_with_mask_image_needle.py` for
image-needle or image-query tasks such as MM-NIAH image multiple choice,
identical-image, and VRAG-style image-query prompts.

## Data

The MM-NIAH examples expect MMLongBench under `data/MMLongBench/` by default.
See the root `README.md` for official download commands. The runners infer the
default JSONL from `--dataset_name` and `--max_context_len`, and default
`--task_image_dir` to `data/MMLongBench/mmlb_image`.

Pass `--task_data_path` and `--task_image_dir` only when using a nonstandard
data location or a non-MM-NIAH task.

## Image Multiple-Choice Example

```bash
python experiments/causal_masking/needle_in_haystack_with_mask_image_needle.py \
  --dataset_name mm_image \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --head_score_path results/detection/Qwen3-VL-8B-Instruct/mm_niah_retrieval-image_retrieval-image_test_K128_dep6_in131072_size20_sampFalse_addnullTrue_42.json \
  --mask_topk 58 \
  --min_context_len 131072 \
  --max_context_len 131072 \
  --ctx_len_intervals 0 \
  --document_depth_percent_min 50 \
  --document_depth_percent_max 50 \
  --document_depth_percent_intervals 0 \
  --example_id mm-niah-23 \
  --save_path results/causal_masking/mm_image_mc
```

## Text-Needle Example

```bash
python experiments/causal_masking/needle_in_haystack_with_mask_text_needle.py \
  --dataset_name mm_text \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --head_score_path results/detection/Qwen3-VL-8B-Instruct/mm_niah_retrieval-text_retrieval-text_test_K128_dep6_in131072_size20_sampFalse_addnullTrue_42.json \
  --mask_topk 58 \
  --min_context_len 131072 \
  --max_context_len 131072 \
  --ctx_len_intervals 0 \
  --document_depth_percent_min 50 \
  --document_depth_percent_max 50 \
  --document_depth_percent_intervals 0 \
  --example_id mm-niah-24 \
  --save_path results/causal_masking/mm_text
```

## Original Masking Convention

- `--mask_topk 0`: baseline, no heads masked.
- `--mask_topk 58`: mask the top 5% Qwen3-VL-8B task-specific detection heads.
- `--mask_topk -58`: mask 58 random heads.
- Use task-specific full detection outputs from `results/detection/` for causal
  masking. The curated four-task union files under `configs/retriever/` are for
  retrieval runs.
- Prefill-plus-decoding masking is the default.
- `--no_mask_prefill`: apply the mask during decoding only.
- Use task-specific `--save_path` values when multiple causal tasks share the
  same model/context/depth settings.

Outputs are written in the original nested format:

```text
<save_path>/<model_and_mask_name>/<model>_len_<context_length>_depth_<depth>_results.json
```
