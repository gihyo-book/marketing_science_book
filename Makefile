RUFF := uv run --locked ruff

.PHONY: lint lint-fix format format-check all

all: lint-fix format

lint:
	$(RUFF) check .

lint-fix:
	$(RUFF) check --fix .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .
