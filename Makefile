.PHONY: install fetch features analyze squeeze all test lint fmt help

PYTHON := python
VENV := .venv
PIP := $(VENV)/bin/pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies into virtualenv
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e ".[dev]"

# ── Data pipeline ────────────────────────────────────────────────────────────
fetch:  ## Download raw FINRA short-sale and price data
	$(PYTHON) scripts/fetch_data.py

features:  ## Compute all features from raw data
	$(PYTHON) scripts/build_features.py

analyze:  ## Run IC / portfolio sorts / Fama-MacBeth analysis
	$(PYTHON) scripts/run_analysis.py

squeeze:  ## Train and evaluate the short-squeeze detector
	$(PYTHON) scripts/run_squeeze_model.py

all: fetch features analyze squeeze  ## Run the full pipeline end-to-end

# ── Quality ──────────────────────────────────────────────────────────────────
test:  ## Run the test suite with coverage
	$(VENV)/bin/pytest

lint:  ## Lint with ruff
	$(VENV)/bin/ruff check src/ tests/ scripts/

fmt:  ## Auto-format with black
	$(VENV)/bin/black src/ tests/ scripts/

# ── Notebooks ────────────────────────────────────────────────────────────────
nb:  ## Convert .py notebook stubs to .ipynb with jupytext
	for f in notebooks/*.py; do $(VENV)/bin/jupytext --to notebook $$f; done
