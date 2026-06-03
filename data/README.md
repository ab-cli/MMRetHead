# Local Data

Put downloaded benchmark data here when running the release scripts from the
`MMRetHead` repository root. The raw datasets are ignored by git.

Expected layout:

```text
data/
  MMLongBench/
    mmlb_data/
      NIAH/
        retrieval-image_test_K128_dep6.jsonl
        retrieval-text_test_K128_dep6.jsonl
        ...
    mmlb_image/
      mm-niah/
        ...
        text_needle_image/
          ...
  MMDocIR_Evaluation_Dataset/
    MMDocIR_annotations.jsonl
    MMDocIR_pages.parquet
    MMDocIR_layouts.parquet
    page_images/
    layout_images/
```

The root README contains the download commands. These paths are the script
defaults, but every script still accepts explicit CLI path overrides.
