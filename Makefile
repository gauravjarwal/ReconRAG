.DEFAULT_GOAL := help
PIPELINE ?= v1_hybrid

help: ## Show the commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

setup: ## Install everything, generate secrets, start services, migrate
	uv sync --extra ui
	uv run reconrag secrets
	$(MAKE) services
	uv run reconrag migrate

setup-phi: ## Install NER for PHI collections, then set PHI_NER_ENABLED=true
	uv sync --extra ui --extra phi
	uv pip install "en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"

services: ## Start Postgres and Redis
	docker compose -f docker-compose.dev.yml up -d --wait

dev: ## API, worker and UI, stopped together with Ctrl-C
	@trap 'kill 0' EXIT; \
	uv run reconrag worker & \
	uv run reconrag api & \
	BACKEND_URL=http://127.0.0.1:8000 BACKEND_API_KEY=$$(grep ^FRONTEND_API_KEY= .env | cut -d= -f2-) \
		uv run --extra ui --directory frontend streamlit run app.py --browser.gatherUsageStats=false --server.maxUploadSize=50

test: ## Unit tests, no network
	uv run pytest tests/unit -q

test-int: services ## Integration tests against real Postgres and Redis
	uv run pytest tests/integration -q

check: ## Lint, format, types and unit tests
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest -q

corpus: ## Rebuild the messy corpus
	uv run python -m evals.corpus.build

eval: ## Run all free evals, gate on the baseline, write the HTML report
	uv run python -m evals.run --label local
	uv run python -m evals.report --b-pipeline $(PIPELINE)
	uv run python -m evals.compare --gate --b-pipeline $(PIPELINE)

eval-ci: ## One pipeline, fail on regression (what CI runs)
	uv run python -m evals.run --pipelines $(PIPELINE) --label ci
	uv run python -m evals.compare --gate --b-pipeline $(PIPELINE)

eval-stub: ## Test the generation harness with no model and no cost
	uv run python -m evals.generation --per-category 2

eval-full: ## Live generation evals. Needs GEMINI_API_KEY and MAX_USD, for example MAX_USD=2
	@test -n "$(MAX_USD)" || (echo "Set MAX_USD. This spends real money."; exit 1)
	uv run python -m evals.generation --provider gemini --max-usd $(MAX_USD) --repeats 3 --judge --label full

judge-calibrate: ## Measure the judge on seeded errors. Needs GEMINI_API_KEY and MAX_USD
	@test -n "$(MAX_USD)" || (echo "Set MAX_USD. This spends real money."; exit 1)
	uv run python -m evals.generation --provider gemini --max-usd $(MAX_USD) --calibrate-judge

report: ## Compare the baseline with the latest run
	uv run python -m evals.report

label: ## Open the labelling page on http://127.0.0.1:8765
	uv run python -m evals.label

check-evals: ## Lint, types and tests for the eval code
	uv run ruff check evals
	uv run ruff format --check evals
	uv run mypy
	uv run pytest -q

.PHONY: help setup setup-phi services dev test test-int check corpus eval eval-ci eval-stub eval-full judge-calibrate report label check-evals
