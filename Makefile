SHELL := /bin/bash
ENV := source scripts/env.sh &&

.PHONY: data baseline smoke test

## Download datasets and build evaluation slices (idempotent)
data:
	$(ENV) python scripts/prepare_data.py

## Full V0 baseline evaluation: prepares missing data, runs all slices, writes results/<run>/
baseline:
	$(ENV) python scripts/evaluate_baseline.py

## Quick end-to-end check: 5 queries per slice
smoke:
	$(ENV) python scripts/evaluate_baseline.py --max-queries 5 --tag smoke

## Unit tests (no GPU needed)
test:
	$(ENV) python -m pytest -q
