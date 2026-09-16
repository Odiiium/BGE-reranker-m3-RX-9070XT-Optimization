# bge-reranker-optimization

Reducing the inference latency of [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) on an AMD Radeon RX 9070 XT (RDNA4, `gfx1201`) with Triton kernels on ROCm. **The model weights stay unchanged**, so every optimized version must keep the baseline quality.

This repository holds the evaluation harness and the model versions. Each version is a git tag:

| Tag | Version | What it is |
|---|---|---|
| `v0.1-baseline` | V0 | Stock `transformers` model, fp16, PyTorch SDPA attention (AOTriton on ROCm), no compilation |

## Quick start

```bash
make baseline   # prepare missing data, evaluate V0 on the full suite (~23 min on RX 9070 XT, budget 45 min)
make smoke      # same pipeline on 5 queries per slice (~1 min)
make data       # only download datasets and build evaluation slices
make test       # unit tests (no GPU required)
```

`scripts/env.sh` activates the shared ROCm virtualenv at `../.venv` (override with `VENV_DIR`). It also exports `HSA_ENABLE_DXG_DETECTION=1`, which ROCm needs to see the GPU under WSL2.

### Requirements

- PyTorch and Triton ROCm builds, already installed in the venv. They are deliberately left out of `pyproject.toml` so pip cannot replace them with CUDA wheels.
- Python packages: `transformers>=5`, `huggingface_hub`, `pandas`, `pyarrow`, `pyyaml`, `tqdm`, `bm25s`, `PyStemmer`, `pytest`.
- About 1 GB of disk space for raw datasets and slices.

## Output of a run

Every evaluation writes `results/<timestamp>_<model>[_tag]/`:

| File | Content |
|---|---|
| `summary.md` | Tables: reranked quality, first-stage quality, gain, efficiency |
| `metrics.csv`, `metrics_wide.csv` | nDCG / MRR / Recall at @1, @5, @10 per dataset and stage |
| `efficiency.csv` | Per-query latency (p50/p95/p99), pairs/s, tokens/s, padding ratio, peak VRAM |
| `per_query/*.parquet` | Per-query metrics of both stages and timings |
| `scores/*.parquet` | Raw score of every candidate, for numerical-parity checks between versions |
| `run_info.json` | Model config, weights sha256, slice fingerprints, git commit, ROCm/torch/Triton versions, timing vs budget |

## Architecture

```
configs/
  datasets.yaml            slices: source, sampled queries, top_k, relevance threshold, pinned HF revisions
  eval.yaml                slices to run, metrics, cutoffs, time budget
  models/v0_baseline.yaml  model variant config
src/bgeopt/
  data/                    DatasetLoader (download -> slice -> parquet cache), BEIR / TREC DL / MIRACL sources, BM25
  metrics/                 NDCG, MRR, Recall + RetrievalEvaluator (trec_eval semantics)
  models/                  BaseReranker interface, registry, HFCrossEncoderReranker (V0)
  evaluation/              runner, report writer, CLI
scripts/                   prepare_data.py, evaluate_baseline.py, evaluate.py, env.sh
tests/                     metrics, slicing, end-to-end runner with a CPU reranker
```

Scoring one query has two stages, and each is timed separately:

- `prepare()` runs on the CPU: tokenization, truncation, length sorting, batching, padding.
- `predict()` runs on the device and returns host-side scores, so the timing includes GPU synchronization.

### Adding an optimized version

1. Subclass `BaseReranker`, or `HFCrossEncoderReranker` to reuse tokenization and loading. Override what the optimization changes:
   - `build_model()` for `torch.compile` or patched modules;
   - `forward_batch()` for custom kernels;
   - `prepare()` / `predict()` for unpadded varlen inputs.
2. Register it with `@register_reranker("my-key")` and add its module to `BUILTIN_IMPLEMENTATIONS` in `src/bgeopt/models/registry.py`.
3. Add `configs/models/v1_xxx.yaml` with `implementation: my-key`.
4. Run `python scripts/evaluate.py --model-config configs/models/v1_xxx.yaml`.

Data, metrics and reports are shared, so all versions are compared on identical inputs.

## Evaluation suite

The suite has 11 slices: ~5,000 queries and ~500,000 query–passage pairs. It is sized to use about 50–65% of the 45-minute budget for the slowest model (V0). Pair lengths range from ~30 tokens (Quora) up to the 512-token cap (ArguAna, SciFact), so latency is measured across short, medium and long sequences.

| Slice | Lang | Queries | Candidates | First stage | Judgments |
|---|---|---|---|---|---|
| trec-dl-2019 | en | 43 (all) | ≤100 | BM25 over the official top-1000 pool | 0–3, relevant ≥ 2 |
| trec-dl-2020 | en | 54 (all) | ≤100 | BM25 over the official top-1000 pool | 0–3, relevant ≥ 2 |
| scifact | en | 300 (all) | 100 | BM25, full corpus | 0–1 |
| nfcorpus | en | 323 (all) | 100 | BM25, full corpus | 1–2 |
| fiqa | en | 648 (all) | 100 | BM25, full corpus | 0–1 |
| trec-covid | en | 50 (all) | 100 | BM25, full corpus | 0–2 |
| scidocs | en | 1000 (all) | 100 | BM25, full corpus | 0–1 |
| arguana | en | 500 of 1406 | 100 | BM25, full corpus, query's own document excluded | 0–1 |
| quora | en | 400 of 10000 | 100 | BM25, full corpus, query's own document excluded | 0–1 |
| miracl-ru | ru | 997 (all) | 100 | MMTEB `MIRACLReranking` candidates | 0–1 |
| miracl-en | en | 705 (all) | 100 | MMTEB `MIRACLReranking` candidates | 0–1 |

How the slices are built:

- Queries without a relevant judgment are dropped.
- Sampling is deterministic, controlled by `seed` in the config.
- Any change to a slice definition rebuilds that slice automatically.
- The official TREC DL top-1000 files carry no scores, so BM25 is recomputed over the pooled passages of each year.

### Metrics

nDCG@k, MRR@k and Recall@k for k ∈ {1, 5, 10}. All cutoffs are computed from the same reranked lists: one inference pass, several metric passes.

The semantics follow `trec_eval` / `pytrec_eval`, the convention used by BEIR and MTEB:

- **nDCG** uses linear graded gain. IDCG is computed over all judged documents of the query.
- **MRR and Recall** use the slice's relevance threshold.
- **Score ties** are broken by `doc_id` in descending order.

The implementation matches `pytrec_eval` to within 1e-15 on randomized runs with ties.

## Environment notes

- **Use fp16.** fp32 matrix multiplication on `gfx1201` is 20–30× slower than fp16 with ROCm 7.2.
- **V0 sorts pairs by token length before batching.** This matches FlagEmbedding's `FlagReranker.compute_score`, so later versions are not credited with that speedup.
- **`TOKENIZERS_PARALLELISM=true`** makes the CPU stage ~4× faster. The pipeline never forks, so this is safe.
