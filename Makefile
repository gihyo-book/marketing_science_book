RUFF := uvx ruff@0.16.4

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
