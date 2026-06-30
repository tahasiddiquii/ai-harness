.DEFAULT_GOAL := help
PYTHON ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help setup install dev lint fmt typecheck test eval demo serve clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv
	$(PYTHON) -m venv $(VENV)

install: ## Install the package (runtime deps only)
	$(BIN)/python -m pip install -U pip
	$(BIN)/python -m pip install -e .

dev: ## Install with dev + provider extras
	$(BIN)/python -m pip install -U pip
	$(BIN)/python -m pip install -e ".[dev,openai,anthropic]"

lint: ## Lint with ruff
	$(BIN)/ruff check .

fmt: ## Auto-format / auto-fix with ruff
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

typecheck: ## Type-check with mypy
	$(BIN)/mypy src

test: ## Run the test suite (offline, no keys)
	$(BIN)/pytest

eval: ## Run the evaluation harness + quality gate
	$(BIN)/python evals/run_evals.py

demo: ## Run the scripted demo (emits Langfuse traces when keys are set)
	$(BIN)/python scripts/demo.py

serve: ## Start the FastAPI server on :8000
	$(BIN)/ai-harness serve --reload

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
