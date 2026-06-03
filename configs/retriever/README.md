# Retriever Configs

This folder contains compact retrieval-head configs that can be passed to
MMRetHead retrievers with `--head-score-json`. Each JSON stores a curated
`head_score_list` and the metadata needed to choose a matching retriever setup.

Use `index.json` for a machine-readable list of the bundled files and their
recommended `--head-top-k` values.

Naming:

- `*_four_tasks_top5pct_union_detection_head_score.json`: union of top-5% heads
  from the four MM-NIAH detection tasks at the named context length.

For MMDocIR page/layout retrieval, prefer a VLM four-task union file matching
the model and context you want to test.
