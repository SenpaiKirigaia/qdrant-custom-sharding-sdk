.DEFAULT_GOAL := help

.PHONY: help install test test-integration test-e2e test-all test-min-client typecheck up down reset status logs load

help:
	@echo install: sync dev dependencies with uv
	@echo test: unit tests only
	@echo test-integration: pytest integration marker
	@echo test-e2e: pytest e2e marker
	@echo test-all: unit, integration, and e2e in one run
	@echo test-min-client: unit and integration with qdrant-client 1.12
	@echo typecheck: mypy and pyright reveal of client method signatures
	@echo up: start docker compose stack
	@echo down: stop stack, keep volumes
	@echo reset: stop stack and remove volumes
	@echo status: docker compose ps
	@echo logs: last 100 log lines
	@echo load: upsert, then migrate while readers run

install:
	uv sync

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

test-e2e:
	uv run pytest -m e2e

test-all:
	uv run pytest -m "integration or e2e or (not integration and not e2e)"

test-min-client:
	uv run --with "qdrant-client==1.12.*" pytest -q -m "not e2e and not typing" --client-floor=1.12

typecheck:
	uv run --with mypy mypy tests/typing/check_hover.py
	uv run --with pyright pyright tests/typing/check_hover.py
	uv run pytest -q -m typing -o addopts=

up:
	docker compose up -d

down:
	docker compose down

reset:
	docker compose down -v

status:
	docker compose ps

logs:
	docker compose logs --tail 100

load:
	uv run python -m tests.load.harness
