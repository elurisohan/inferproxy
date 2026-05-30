.PHONY: install dev test lint format

install:
	uv sync

dev:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run mypy app tests

format:
	uv run ruff check --fix .
	uv run black .
