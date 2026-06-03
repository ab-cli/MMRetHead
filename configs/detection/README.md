# Detection Configs

These YAML files define the MM-NIAH detection task, context length, test file,
generation length, and sample count. They intentionally do not set the model or
data roots. The default roots are `data/MMLongBench/mmlb_data` and
`data/MMLongBench/mmlb_image`; pass path overrides only when your data lives
elsewhere.

Raw MM-NIAH data is not bundled in this repository. The configs point to files
relative to `--test_file_root`, and the loaders resolve all relative image paths
against `--image_file_root`. See the root `README.md` for official
MMLongBench download commands.

The bundled configs use `max_test_samples: 20` because they reproduce the full
scored detection JSONs shipped under `results/detection/`. For a larger
confirmatory rerun, override it:

```bash
--max_test_samples 128 --overwrite
```

## Required Data Layout

The bundled configs expect this MM-NIAH file layout:

```text
<test_file_root>/
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

<image_file_root>/
  mm-niah/
    ... image files referenced by image_list and choices_image ...
    text_needle_image/
      ... rendered positive-text needles used by text_needle_image configs ...
```

`test_files` in each YAML is joined to `--test_file_root`. Image fields in the
JSONL rows are joined to `--image_file_root`. The rendered-text needle loader
constructs filenames by lowercasing the positive text, removing punctuation,
replacing spaces with underscores, and appending `.png`; those generated PNGs
must already exist in `<image_file_root>/mm-niah/text_needle_image/`.

## Tasks

| Config prefix | Dataset key | Source JSONL | Evidence target | Answer format | Notes |
| --- | --- | --- | --- | --- | --- |
| `mm_niah_image_*` | `mm_niah_retrieval-image` | `NIAH/retrieval-image_test_K*_dep6.jsonl` | Image in the haystack `positive_ctxs` | Multiple choice, `A`/`B`/... | Upstream MM-NIAH image retrieval task. |
| `mm_niah_image_identical_*` | `identical_retrieval-image` | `NIAH/retrieval-image_test_K*_dep6.jsonl` | Inserted exact image in the document haystack | Binary `Yes`/`No` | Repo-constructed exact-image presence task; the current detection path is positive-presence only. |
| `mm_niah_text_*` | `mm_niah_retrieval-text` | `NIAH/retrieval-text_test_K*_dep6.jsonl` | Text span in `positive_ctxs` | Free-form text | Upstream MM-NIAH text retrieval task. |
| `mm_niah_text_needle_image_*` | `text_needle_image` | `NIAH/retrieval-text_test_K*_dep6.jsonl` | Rendered image of the positive text span | Free-form text | Repo-constructed rendered-text task; requires generated PNGs under `mm-niah/text_needle_image/`. |

Each task has `k8`, `k16`, `k32`, `k64`, and `k128` variants.

## Example

Run one Qwen3-VL-8B detection job from the `MMRetHead` repository root:

```bash
python -m mmrethead.eval \
  --config configs/detection/mm_niah_image_k128.yaml \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct
```

If `--output_dir` is omitted, the full scored JSON is written under:

```text
results/detection/<model_basename>/
```

Use the same config with a different `--model_name_or_path` to run the same
task/context condition on another supported VLM.
