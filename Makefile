SHELL := /bin/bash
ENV := source scripts/env.sh &&

.PHONY: data baseline smoke bench-baseline bench-quick compare test

## Download datasets and build evaluation slices (idempotent)
data:
	$(ENV) python scripts/prepare_data.py

## Full V0 quality evaluation: prepares missing data, runs all slices, writes results/quality/<run>/
baseline:
	$(ENV) python scripts/evaluate_baseline.py

## Quick end-to-end check of the quality pipeline: 5 queries per slice
smoke:
	$(ENV) python scripts/evaluate_baseline.py --max-queries 5 --tag smoke

## V0 latency benchmark, default profile (3 sessions), writes results/latency/<run>/
bench-baseline:
	$(ENV) python scripts/benchmark_baseline.py

## V0 latency benchmark, quick profile (2 sessions, compare only with other quick runs)
bench-quick:
	$(ENV) python scripts/benchmark_baseline.py --profile quick --tag quick

## Compare two runs of the same type: make compare A=results/quality/<baseline> B=results/quality/<candidate>
compare:
	$(ENV) python scripts/compare.py $(A) $(B)

## Unit tests (no GPU needed)
test:
	$(ENV) python -m pytest -q
