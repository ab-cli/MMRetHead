# MMRetHead

This repository is the official code release for the paper **[Can Retrieval Heads See Images? Multimodal Retrieval Heads in Long-Context Vision-Language Models](https://arxiv.org/abs/2605.27243)**.

The implementation builds on the Needle-in-a-Haystack evaluation setting and extends retrieval-head analysis from long-context language models to long-context vision-language models. It detects retrieval heads by measuring which attention heads route query tokens toward annotated textual or visual evidence spans. The same retrieval-head mechanism can then be reused as an attention-based retriever for MMDocIR page and layout retrieval.

The intended workflow is:

1. Score retrieval heads on a long-context text or vision-language benchmark.
2. Save the ranked `head_score_list`.
3. Optionally mask those heads to test causal importance.
4. Reuse the strongest heads as a lightweight retriever over MMDocIR candidates.

This repository includes the model wrappers, scoring code, curated retriever configs, and single-GPU MMDocIR runners used by the project. Generated outputs are written under `results/`, which is kept separate from curated reusable configs under `configs/`.

## Contents

- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [CUDA And FlashAttention](#cuda-and-flashattention)
- [Supported VLM Wrappers](#supported-vlm-wrappers)
- [Data Requirements](#data-requirements)
- [Retrieval-Head Detection](#retrieval-head-detection)
- [Causal Masking](#causal-masking)
- [MMDocIR Retrieval](#mmdocir-retrieval)
- [Outputs](#outputs)
- [Citation](#citation)
- [Credits](#credits)

## Repository Layout

```text
configs/detection/                  Reusable MM-NIAH detection task configs.
configs/retriever/                  Curated retrieval-head retriever configs.
experiments/causal_masking/         Causal masking runner with compact outputs.
experiments/mmdocir_retrieval/      Page and layout retrieval runners.
src/mmrethead/                      Package source.
src/mmrethead/vlm_model/            VLM model wrappers.
src/mmrethead/mmdocir/              MMDocIR data, metrics, and output helpers.
src/mmrethead/retrievers/           Attention and text-baseline retrievers.
```

## Installation

Start from the repository root:

```bash
cd /path/to/MMRetHead
python -m pip install -e .
```

The base install is enough for MMDocIR data loading, metrics, compact output writing, and `--scorer text_baseline` smoke tests.

For model runs, use separate Python environments for Qwen and Gemma when possible. They share PyTorch, Transformers, Accelerate, and FlashAttention, but the Qwen VL wrappers also require Qwen-specific vision preprocessing.

| Use case | Install command | Notes |
| --- | --- | --- |
| Qwen VLM detection/retrieval | `python -m pip install -e ".[qwen]"` | Use for Qwen2.5-VL, Qwen3-VL, and QVQ-style wrappers. Includes `qwen-vl-utils`. |
| Gemma 3 VLM detection/retrieval | `python -m pip install -e ".[gemma]"` | Use for Gemma 3 VLM runs. Does not include `qwen-vl-utils`. |
| Combined Qwen + Gemma | `python -m pip install -e ".[vlm]"` | Convenience environment when one environment must run both families. |

Install PyTorch for the target CUDA stack before the extras if the cluster provides a pinned wheel or module. The custom wrappers also expect a Transformers build that contains the relevant model modules, for example `transformers.models.qwen3_vl` for Qwen3-VL and `transformers.models.gemma3` for Gemma 3.

## CUDA And FlashAttention

Both Qwen and Gemma wrappers default to `flash_attention_2`. Install FlashAttention separately in CUDA environments that support it:

```bash
python -m pip install flash-attn --no-build-isolation
```

CUDA version rules:

- Treat the PyTorch CUDA build as the source of truth:

  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda)"
  ```

- `nvidia-smi` reports the maximum CUDA runtime supported by the driver. It does not prove that PyTorch or FlashAttention were built for that same CUDA version.
- When building `flash-attn` from source, load a CUDA toolkit whose `nvcc --version` matches the CUDA version used by PyTorch.
- Avoid upgrading `torch` after FlashAttention has been built.
- One validated environment used driver CUDA `13.0`, `nvcc` `13.0`, PyTorch CUDA `13.0`, and `flash_attn` `2.7.4.post1`.

Quick environment check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
try:
    import flash_attn
    print("flash_attn:", flash_attn.__version__)
except Exception as exc:
    print("flash_attn import failed:", exc)
PY
```

If FlashAttention is not available, `--attn_implementation sdpa` can be used for dependency smoke tests or non-blocking debug runs. It is not a full replacement for every path.

Compatibility notes:

- Do not enable `torch.compile` for the retrieval-head wrappers unless you are revalidating the cache path. The custom `DynamicCacheWithQuery` path is not compile-safe in the current implementation.
- Qwen2.5-VL previously hit a Dynamo/FlashAttention conflict where Dynamo expected `max_seqlen_q` as a tensor while FlashAttention expected an integer. The wrapper leaves compile disabled for that path.
- Head-blocking or ablation paths that pass `block_list` require `flash_attention_2`; the custom model code asserts on this.

## Supported VLM Wrappers

`mmrethead.vlm_model.load_LLM` selects wrappers by `--model_name_or_path`.

| Family | Examples |
| --- | --- |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-*` |
| Qwen3-VL | `Qwen/Qwen3-VL-*` |
| Gemma 3 VLM | `google/gemma-3-*` |

## Data Requirements

Raw benchmark data is not bundled with this repository. The reusable configs and
curated result files are included, but rerunning detection or retrieval requires
local copies of the corresponding datasets. By default, scripts look under
`data/` inside the `MMRetHead` repository. Raw files under `data/` are ignored
by git; `data/README.md` is only a layout guide.

### MMLongBench

Retrieval-head detection and causal masking use MMLongBench, especially the
MM-NIAH subset. Download the official dataset from
[ZhaoweiWang/MMLongBench](https://huggingface.co/datasets/ZhaoweiWang/MMLongBench).

For the bundled MM-NIAH detection configs and causal masking examples, the
minimal MMLongBench download is:

```bash
python -m pip install -U huggingface_hub hf_xet

mkdir -p data/MMLongBench

huggingface-cli download ZhaoweiWang/MMLongBench 0_mmlb_data.tar.gz \
  --repo-type dataset \
  --local-dir data/MMLongBench
huggingface-cli download ZhaoweiWang/MMLongBench 2_mm-niah_image.tar.gz \
  --repo-type dataset \
  --local-dir data/MMLongBench

tar -xzf data/MMLongBench/0_mmlb_data.tar.gz -C data/MMLongBench
tar -xzf data/MMLongBench/2_mm-niah_image.tar.gz -C data/MMLongBench
```

For the full MMLongBench image set, download and extract all image archives:

```bash
for file in \
  1_vrag_image.tar.gz \
  2_vh_image.tar.gz \
  2_mm-niah_image.tar.gz \
  3_icl_image.tar.gz \
  4_summ_image.tar.gz \
  5_docqa_image.tar.gz; do
  huggingface-cli download ZhaoweiWang/MMLongBench "$file" \
    --repo-type dataset \
    --local-dir data/MMLongBench
  tar -xzf "data/MMLongBench/$file" -C data/MMLongBench
done
```

After extracting MMLongBench, the repository expects this layout:

```text
data/MMLongBench/
  mmlb_data/
    NIAH/
      retrieval-image_test_K8_dep6.jsonl
      retrieval-image_test_K16_dep6.jsonl
      retrieval-image_test_K32_dep6.jsonl
      retrieval-image_test_K64_dep6.jsonl
      retrieval-image_test_K128_dep6.jsonl
      retrieval-text_test_K8_dep6.jsonl
      retrieval-text_test_K16_dep6.jsonl
      retrieval-text_test_K32_dep6.jsonl
      retrieval-text_test_K64_dep6.jsonl
      retrieval-text_test_K128_dep6.jsonl
  mmlb_image/
    mm-niah/
      ... images referenced by image_list, choices_image, and needle_image_list ...
      text_needle_image/
        ... rendered PNGs for the text-needle-image task ...
```

Detection defaults to `data/MMLongBench/mmlb_data` and
`data/MMLongBench/mmlb_image`. Causal masking defaults to the same image root
and infers the MM-NIAH JSONL from the runner, `--dataset_name`, and
`--max_context_len`. If your data lives elsewhere, pass `--test_file_root` /
`--image_file_root` for detection, or `--task_data_path` / `--task_image_dir`
for causal masking.

The bundled detection configs cover four MM-NIAH conditions used for the
curated retriever-head unions:

- `mm_niah_image_*`: `NIAH/retrieval-image_test_K*_dep6.jsonl` plus MM-NIAH images.
- `mm_niah_image_identical_*`: the same image JSONL, converted into an exact-image presence task.
- `mm_niah_text_*`: `NIAH/retrieval-text_test_K*_dep6.jsonl`.
- `mm_niah_text_needle_image_*`: the same text JSONL, but the positive text span is rendered as an image; this requires PNGs under `data/MMLongBench/mmlb_image/mm-niah/text_needle_image/`.

See `configs/detection/README.md` for task provenance, answer formats, and
path-resolution details.

MMLongBench is required to reproduce detection runs or run causal masking. It is
not required just to use the curated head-score JSON files in `configs/retriever/`
for MMDocIR retrieval.

### MMDocIR

Download the official evaluation data from
[MMDocIR/MMDocIR_Evaluation_Dataset](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset).
The minimum files needed by this repository are:

- `MMDocIR_annotations.jsonl`
- `MMDocIR_pages.parquet`
- `MMDocIR_layouts.parquet`

```bash
python -m pip install -U huggingface_hub hf_xet

mkdir -p data/MMDocIR_Evaluation_Dataset

for file in \
  MMDocIR_annotations.jsonl \
  MMDocIR_pages.parquet \
  MMDocIR_layouts.parquet; do
  huggingface-cli download MMDocIR/MMDocIR_Evaluation_Dataset "$file" \
    --repo-type dataset \
    --local-dir data/MMDocIR_Evaluation_Dataset
done
```

For VLM attention retrieval, the candidate pages or layouts are scored as
screenshots: the retriever measures attention from question tokens to the image
tokens for each candidate. The parquet text fields are enough for
`--scorer text_baseline` smoke tests, but attention retrieval needs image files.
Provide those images in one of two ways:

- Pass `--materialize-images` to extract selected page/layout images from the
  parquet `image_binary` columns.
- Or download and extract the optional image archives from the same Hugging Face
  dataset, especially `doc_miscellaneous/page_images.rar` and
  `doc_miscellaneous/layout_images.rar`, then pass the extracted root with
  `--image-root`.

```bash
for file in \
  doc_miscellaneous/page_images.rar \
  doc_miscellaneous/layout_images.rar; do
  huggingface-cli download MMDocIR/MMDocIR_Evaluation_Dataset "$file" \
    --repo-type dataset \
    --local-dir data/MMDocIR_Evaluation_Dataset
done
```

Extract the RAR files with `unrar` or `7z`. The image root passed to
`--image-root` should contain the relevant `page_images/` or `layout_images/`
directory, or the corresponding `doc_miscellaneous/page_images/` or
`doc_miscellaneous/layout_images/` subdirectory. MMDocIR scripts default to
`data/MMDocIR_Evaluation_Dataset` for annotations, parquet files, and
image-root lookup.

## Retrieval-Head Detection

Detection computes a score for each attention head by measuring attention from query tokens to the annotated evidence span. Null-query calibration is enabled by default and can be disabled with `--add_null_score False`.

Task/context configs live in `configs/detection/`. They define the MM-NIAH task,
test file, context length, generation length, and sample count. If MMLongBench
is under the default `data/MMLongBench/` location, only the config and model are
needed.

Qwen3-VL example:

```bash
python -m mmrethead.eval \
  --config configs/detection/mm_niah_image_k128.yaml \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct
```

Gemma 3 VLM example:

```bash
python -m mmrethead.eval \
  --config configs/detection/mm_niah_text_k32.yaml \
  --model_name_or_path google/gemma-3-12b-it
```

Useful detection flags:

- `--output_dir`: destination for full scored detection JSONs. If omitted, outputs go to `results/detection/<model_basename>/`.
- `--test_file_root`, `--image_file_root`: override the default `data/MMLongBench/mmlb_data` and `data/MMLongBench/mmlb_image` roots.
- `--config configs/detection/<task>_<context>.yaml`: load one of the reusable task/context configs.
- `--max_test_samples 128`: override the bundled `size20` configs for a larger rerun.
- `--dry_run`: after model initialization, load data without running attention scoring.
- `--count_tokens`: after model initialization, tokenize samples and report input length statistics without running attention scoring.
- `--overwrite`: rerun even when the output JSON already exists.
- `--add_null_score False`: disable null-query calibration.
- `--save_activation_frequency`: write an activation-frequency `.npz` sidecar.
- `--attn_implementation sdpa`: debug fallback when not using paths that require FlashAttention.

Detection writes one JSON file under `--output_dir`. The main fields are:

```python
{
  "head_score_list": [["layer-head", score], ...],
  "activation_frequency_head_list": [["layer-head", score], ...] or None,
  "activation_frequency_path": "path/to/sidecar.npz" or None,
  "args": {...},
  "averaged_metrics": {...},
  "data": [...]
}
```

`head_score_list` is sorted from strongest to weakest head.

```python
import json

with open("results/detection/Qwen3-VL-8B-Instruct/mm_niah_retrieval-image_retrieval-image_test_K128_dep6_in131072_size20_sampFalse_addnullTrue_42.json") as f:
    result = json.load(f)

top_heads = result["head_score_list"][:10]
print(top_heads)
```

Curated retriever configs live in `configs/retriever/`. The individual `*_head_score.json` files can be passed directly to the MMDocIR retrievers. The folder includes:

- `*_four_tasks_top5pct_union_detection_head_score.json`: unions of the top-5% heads from MM-NIAH image-MC, image-YN, text, and text-needle-image detection runs.
- `index.json`: machine-readable metadata for all bundled retriever configs, including the recommended `--head-top-k`.

## Causal Masking

The causal masking experiment uses the original project runners, copied into
`experiments/causal_masking/` with only package-import adaptation for this
refactored repository. Use the text-needle runner,
`needle_in_haystack_with_mask_text_needle.py`, for text-evidence,
rendered-text, NQ, and DocQA-style tasks. Use the image-needle runner,
`needle_in_haystack_with_mask_image_needle.py`, for MM-NIAH image multiple
choice, identical-image, and VRAG-style image-query tasks.

Qwen3-VL MM-NIAH image multiple-choice causal example:

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

Qwen3-VL MM-NIAH text-needle example:

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

The original `--mask_topk` convention is preserved: positive values mask the
top-ranked heads from `--head_score_path`, `0` runs the baseline, and negative
values mask that many random heads. The examples use task-specific full
detection outputs from `results/detection/`; the curated four-task union files
under `configs/retriever/` are intended for retrieval runs rather than
task-specific causal masking. Prefill-plus-decoding masking is now the default;
pass `--no_mask_prefill` for decode-only masking. Use a task-specific
`--save_path` when running multiple causal tasks so their original-runner result
filenames do not collide. The original output layout is preserved under
`--save_path/<model_and_mask_name>/*_results.json`. Pass `--task_data_path` or
`--task_image_dir` only when not using the default `data/MMLongBench/` layout.

## MMDocIR Retrieval

MMRetHead provides page-level and layout-level MMDocIR runners. The default retriever is attention-based; `--scorer text_baseline` is available for dependency-light data and metric smoke tests.

Required data:

- `MMDocIR_annotations.jsonl`
- `MMDocIR_pages.parquet` for page retrieval
- `MMDocIR_layouts.parquet` for layout retrieval
- Extracted page/layout images via `--image-root`, or use `--materialize-images` to extract selected images from parquet for the run.

The commands below assume the default repo-local path
`data/MMDocIR_Evaluation_Dataset`. If you keep the dataset elsewhere, override
`--annotations`, `--pages-parquet` or `--layouts-parquet`, and `--image-root`.
For `--image-root`, pass the parent directory that contains the relevant
`page_images/` or `layout_images/` folder.

Run a cheap data validation pass first:

```bash
python experiments/mmdocir_retrieval/run_page_retrieval.py \
  --prepare-only \
  --results-dir results \
  --run-name page_prepare
```

If you did not extract `page_images/` or `layout_images/`, add
`--materialize-images` to the retrieval commands below. The runner will extract
the selected images from the parquet `image_binary` column into the results
directory and use them for that run.

### Page Retrieval

Qwen3-VL page retrieval:

```bash
python experiments/mmdocir_retrieval/run_page_retrieval.py \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --head-score-json configs/retriever/qwen3_vl_8b_instruct_128k_four_tasks_top5pct_union_detection_head_score.json \
  --head-top-k 112 \
  --window-size 4 \
  --results-dir results \
  --run-name qwen3vl_page_top112
```

Gemma 3 page retrieval:

```bash
python experiments/mmdocir_retrieval/run_page_retrieval.py \
  --model-name google/gemma-3-12b-it \
  --head-score-json configs/retriever/gemma_3_12b_it_128k_four_tasks_top5pct_union_detection_head_score.json \
  --head-top-k 100 \
  --window-size 4 \
  --results-dir results \
  --run-name gemma3_page_top100
```

### Layout Retrieval

```bash
python experiments/mmdocir_retrieval/run_layout_retrieval.py \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --head-score-json configs/retriever/qwen3_vl_8b_instruct_128k_four_tasks_top5pct_union_detection_head_score.json \
  --head-top-k 112 \
  --window-size 200 \
  --results-dir results \
  --run-name qwen3vl_layout_top112
```

Useful retrieval flags:

- `--materialize-images`: extract selected candidate images from parquet for the run.
- `--annotations`, `--pages-parquet`, `--layouts-parquet`, `--image-root`: override the default `data/MMDocIR_Evaluation_Dataset` paths.
- `--max-docs`, `--max-questions`: quick subsets.
- `--limit-pages-per-doc`, `--limit-layouts-per-doc`: candidate caps for debugging.
- `--window-size`: number of candidates scored per prompt.
- `--head-score-json`: detection JSON with `head_score_list`. If omitted, all heads are summed.
- `--head-top-k`: number of selected heads to use; `configs/retriever/index.json`
  lists the recommended value for each bundled head-score file.
- `--disable-null-calibration`: turn off N/A-query calibration.
- `--score-aggregation sum|mean`: aggregate token scores inside each candidate image span.
- `--artifact-mode full`: also write raw JSONL artifacts.
- `--quiet`: compact stdout for batch jobs.

## Outputs

Detection outputs:

```text
<output_dir>/<dataset>_<test>_in<input>_size<samples>_sampFalse_addnullTrue_<seed>.json
<output_dir>/<same_prefix>_activation_frequency.npz  # optional
```

`results/detection/index.json` indexes the mirrored full scored detection JSONs
included in this local copy. The mirrored metadata uses repo- or project-relative
paths instead of machine-local absolute paths. `configs/retriever/` stores
compact curated unions for retrieval; it is not the destination for full
detection reruns.

Page retrieval compact outputs:

```text
results/<run>__selected_manifest.json
results/<run>__config.json
results/<run>__per_query_topk.csv
results/<run>__metrics.json
results/<run>__official_page_metrics.json
results/<run>__official_page_domain_metrics.csv
```

Layout retrieval compact outputs:

```text
results/<run>__selected_manifest.json
results/<run>__config.json
results/<run>__per_query_topk.csv
results/<run>__layout_metrics.json
```

Causal masking original-runner outputs:

```text
<save_path>/<model_and_mask_name>/<model>_len_<context_length>_depth_<depth>_results.json
```

`--artifact-mode full` additionally writes raw selected-query/candidate JSONL and per-query score JSONL files.

## Citation

If you use this repository, please cite:

```bibtex
@misc{li2026mmrethead,
  title = {Can Retrieval Heads See Images? Multimodal Retrieval Heads in Long-Context Vision-Language Models},
  author = {Aaron Branson Cigres Li and Zhaowei Wang and Yu Zhao and Yiming Du and Haobo Li and Xiyu Ren and Ginny Wong and Simon See and Lishu Luo and Haodong Duan and Pasquale Minervini and Yangqiu Song},
  year = {2026},
  eprint = {2605.27243},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2605.27243},
  url = {https://arxiv.org/abs/2605.27243}
}
```

## Credits

This codebase builds on ideas and code structure from
[Retrieval_Head](https://github.com/nightdessert/Retrieval_Head). The retriever
component is also inspired by the query-focused retrieval-head approach in
[QRHead](https://github.com/princeton-pli/QRHead).

Generated `results/`, `runs/`, `logs/`, and `wandb/` folders are ignored by git.
