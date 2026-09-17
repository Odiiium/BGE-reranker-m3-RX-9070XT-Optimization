# bge-reranker-optimization

Reducing the inference latency of [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) on an AMD Radeon RX 9070 XT (RDNA4, `gfx1201`) with Triton kernels on ROCm. **The model weights stay unchanged**, so every optimized version must keep the baseline quality.

This repository holds the evaluation harness and the model versions. Each version is a git tag:

| Tag | Version | What it is |
|---|---|---|
| `v0.1-baseline` | V0 | Stock `transformers` model, fp16, PyTorch SDPA attention (AOTriton on ROCm), no compilation |

## Quick start

```bash
make baseline         # quality: prepare missing data, evaluate V0 on the full suite (~23 min on RX 9070 XT)
make bench-baseline   # latency: controlled V0 benchmark, 3 sessions in separate processes (~12 min)
make compare A=results/quality/<baseline> B=results/quality/<candidate>   # or two results/latency runs
make smoke            # quality pipeline on 5 queries per slice (~1 min)
make bench-quick      # latency benchmark, quick profile: 2 sessions (~4 min)
make data             # only download datasets and build evaluation slices
make test             # unit tests (no GPU required)
```

Quality suite plus latency benchmark take ~35 minutes per version, within the 45-minute budget on one GPU.

`scripts/env.sh` activates the shared ROCm virtualenv at `../.venv` (override with `VENV_DIR`). It also exports `HSA_ENABLE_DXG_DETECTION=1`, which ROCm needs to see the GPU under WSL2.

Any model config value can be overridden from the command line, e.g. `python scripts/evaluate_baseline.py --override dtype=bfloat16 --max-queries 50 --tag bf16`.

### Requirements

- PyTorch and Triton ROCm builds, already installed in the venv. They are deliberately left out of `pyproject.toml` so pip cannot replace them with CUDA wheels.
- Python packages: `transformers>=5`, `huggingface_hub`, `pandas`, `pyarrow`, `pyyaml`, `tqdm`, `bm25s`, `PyStemmer`, `pytest`.
- About 1 GB of disk space for raw datasets and slices.

## Outputs

| Folder | Produced by | Main files |
|---|---|---|
| `results/quality/<timestamp>_<model>[_tag]/` | `evaluate*.py` | `summary.md`, `metrics.csv`, `efficiency.csv`, `per_query/*.parquet`, `scores/*.parquet`, `run_info.json` |
| `results/latency/<timestamp>_<model>[_tag]/` | `benchmark*.py` | `summary.md`, `latency.csv`, `samples.parquet`, `cells.parquet`, `sessions/s<k>/`, `run_info.json` |
| `results/compare/<baseline>__vs__<candidate>/` | `compare.py` | `summary.md`, `quality_deltas.csv` + `parity.csv` or `latency_speedups.csv`, `comparison.json` |

Quality run files:

| File | Content |
|---|---|
| `metrics.csv`, `metrics_wide.csv` | nDCG / MRR / Recall at @1, @5, @10 per dataset and stage |
| `efficiency.csv` | Per-query latency observed during the quality pass (p50/p95/p99), pairs/s, padding ratio, peak VRAM |
| `per_query/*.parquet` | Per-query metrics of both stages and timings |
| `scores/*.parquet` | Raw score of every candidate, used for numerical-parity checks between versions |
| `run_info.json` | Model config, weights sha256, slice fingerprints, git commit, ROCm/torch/Triton versions, timing vs budget |

Every run records `git describe`; a `-dirty` suffix means the working tree had uncommitted changes.

## Architecture

```
configs/
  datasets.yaml            slices: source, sampled queries, top_k, relevance threshold, pinned HF revisions
  eval.yaml                slices to run, metrics, cutoffs, time budget
  benchmark.yaml           latency benchmark profiles (default, quick)
  compare.yaml             equivalence margin, parity gates, bootstrap settings
  models/v0_baseline.yaml  model variant config
src/bgeopt/
  data/                    DatasetLoader (download -> slice -> parquet cache), BEIR / TREC DL / MIRACL sources, BM25
  metrics/                 NDCG, MRR, Recall + RetrievalEvaluator (trec_eval semantics)
  models/                  BaseReranker interface, registry, config overrides, HFCrossEncoderReranker (V0)
  evaluation/              quality runner, report writer, CLI
  benchmark/               latency profiles, session measurement, worker process, aggregation, orchestrator
  comparison/              quality equivalence + numerical parity, latency speedups, reports, CLI
  utils/                   environment capture, bootstrap / randomization statistics, paths
scripts/                   prepare_data, evaluate[_baseline], benchmark[_baseline], compare, env.sh
tests/                     metrics, slicing, runner, statistics, comparisons, benchmark sessions (CPU reranker)
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
4. Run `python scripts/evaluate.py --model-config configs/models/v1_xxx.yaml` and `python scripts/benchmark.py --model-config configs/models/v1_xxx.yaml`.
5. Compare both runs against the V0 runs with `scripts/compare.py`.

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

## Comparing versions

`scripts/compare.py BASELINE CANDIDATE` detects the run type and writes `results/compare/<baseline>__vs__<candidate>/`. A candidate evaluated on a subset (`--max-queries`) is compared on the common queries; the report lists the coverage.

### Quality runs

Two separate verdicts are reported.

**Quality** answers whether retrieval metrics changed. For every dataset × metric@k:

- Per-query differences are paired and bootstrapped (2000 resamples).
- A sign-flip randomization test gives the p-value.
- The metric is **equivalent** when the 90% CI of Δ lies within ±0.005 (TOST). Otherwise it is **worse** or **better** when the 95% CI excludes 0, and **inconclusive** in any other case.

Verdicts:

- **PASS**: every macro-averaged metric is equivalent or better.
- **FAIL**: a macro metric is worse, or a single dataset is worse by more than the margin.

**Numerical parity** checks how far raw scores drifted on identical (query, candidate) pairs:

- max and mean |Δlogit|;
- per-query Kendall τ-b, which handles ties (fp16 logits are quantized);
- top-k overlap and the share of queries with an identical top-k order.

Gates: median τ ≥ 0.99 and overlap@10 ≥ 0.98 on every dataset.

The gates were calibrated on V0 variants (50 queries per slice, 53,070 pairs):

| Candidate vs V0 | What changes | max \|Δlogit\| | mean \|Δlogit\| | min median τ | min overlap@10 | Parity |
|---|---|---|---|---|---|---|
| Same config, new process | nothing | 0 | 0 | 1.000 | 1.000 | PASS |
| `sort_by_length=false` | batch composition and padded shapes | 0.109 | ≤ 0.003 | 0.9991 | 0.998 | PASS |
| `attn_implementation=eager` | attention kernel (math path instead of AOTriton) | 0.094 | ≤ 0.003 | 0.9990 | 0.996 | PASS |
| `dtype=bfloat16` | numeric format | 0.480 | ≤ 0.029 | 0.9771 | 0.980 | FAIL |

What the calibration shows:

- **V0 is bitwise deterministic** across processes.
- **fp16 kernel or shape changes** drift by τ ≈ 0.999. The gates sit about 10× above that noise floor.
- **A precision change** such as bf16 is caught.

On 50-query subsets the metric CIs are often wider than ±0.005, so quality verdicts there tend to be INCONCLUSIVE. Use the full suite for release decisions.

### Latency runs

For every synthetic cell and replay dataset, the report gives:

- the p50 speedup (baseline / candidate) with a 95% CI;
- the p95 and p99 speedups;
- the end-to-end p50 speedup;
- a **faster** / **slower** flag when the CI excludes 1 and the effect is at least 3%.

The summary gives geometric-mean speedups over replay datasets and synthetic cells.

**Sessions are the statistical unit.** The CI is a Welch t-interval over per-session p50 values (log scale), not a bootstrap over individual calls. The reason is how this machine (WSL2) behaves:

- On light cells (below ~60 ms), each worker process lands in one of several discrete states. For example, quora runs at 44 ms in some processes and 54 ms in others.
- Calls inside one process are stable.
- Heavy cells are stable everywhere, e.g. 100 × 512 tokens: 421–425 ms.

A call-level bootstrap therefore reported false 10–20% "changes" in A/A tests.

Rules enforced by `compare.py`:

- **Profiles must match.** Warmup length and sample counts alone shift light cells by 10–20%.
- **Fewer than 2 sessions** in a run makes the report fall back to a call-level CI and warn that it is overconfident.
- **Separate invocations** are flagged. For release decisions, benchmark both versions in one interleaved invocation:
  `python scripts/benchmark.py --model-config configs/models/v0_baseline.yaml --model-config configs/models/v1_xxx.yaml`.

Validation, quick profile, one interleaved invocation of V0, V0-eager and V0:

| Comparison | Replay geo-mean p50 speedup | Cells flagged (of 27) |
|---|---|---|
| V0 vs V0 (A/A) | 0.99× | 0 |
| V0 vs V0 with eager attention (A/B, known slower kernel) | 0.72× | 18 slower, 0 faster |

## Latency benchmark

`scripts/benchmark.py` (and `benchmark_baseline.py`) measure latency separately from the quality pass:

- **Separate processes.** Every session runs in a fresh worker process. With several `--model-config`, sessions are interleaved (A B, B A, …).
- **Synthetic grid.** One query with N ∈ {1, 8, 32, 100} candidates of exactly L ∈ {64, 128, 256, 512} tokens, with no padding. This shows kernel and batch-shape behaviour, from the CPU-bound N=1 case to 100 × 512.
- **Replay.** 20 seeded real queries per slice, with 100 candidates each and real lengths and padding. Every session and every version sees the same queries.
- **Measurement.**
  - Each cell has its own warmup. The first call is kept as the cold per-shape cost (kernel selection, compilation).
  - The number of repetitions adapts to a time target.
  - Python GC is disabled while measuring.
  - Cell order is shuffled per session.
  - `predict` includes device synchronization.
- **Report.**
  - p50 with a bootstrap CI, p90, p95, p99, pairs/s and tokens/s;
  - session-to-session CV of p50, i.e. the measurement noise;
  - cold start per session (model load, first call, warmup);
  - peak VRAM.

Profiles live in `configs/benchmark.yaml`:

- `default`: 3 sessions, ~12 min for V0;
- `quick`: 2 sessions, ~4 min, for kernel development. Compare quick runs only with quick runs.

## Environment notes

- **Use fp16.** fp32 matrix multiplication on `gfx1201` is 20–30× slower than fp16 with ROCm 7.2.
- **V0 sorts pairs by token length before batching.** This matches FlagEmbedding's `FlagReranker.compute_score`, so later versions are not credited with that speedup.
- **`TOKENIZERS_PARALLELISM=true`** makes the CPU stage ~4× faster. The pipeline never forks, so this is safe.
