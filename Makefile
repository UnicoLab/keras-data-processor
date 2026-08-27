# ------------------------------------
# Customized func / var define
# ------------------------------------

HAS_POETRY := $(shell command -v poetry 2> /dev/null)

# Sets the package version from the latest git tag. This MUST stay a lazily
# expanded recipe command: as a `:=` shell assignment it ran `poetry version
# <tag>` while make was still parsing the file, so every single make target --
# `make help` included -- silently rewrote the version in pyproject.toml.
SET_VERSION_FROM_TAG = poetry version $$(git describe --tags --abbrev=0)

# ------------------------------------
# Test
# ------------------------------------

.PHONY: test
## Run ALL tests in parallel with optimizations
test:
	poetry run pytest -c pytest-fast.ini

.PHONY: test-fast
## Run only fast tests in parallel
test-fast:
	poetry run pytest -m "fast or unit" --maxfail=5

.PHONY: test-all
## Run ALL tests (unit, integration, layers, everything) with optimizations
test-all:
	poetry run pytest -c pytest-fast.ini --cov=kdp --cov-report=term-missing --cov-report=html

.PHONY: test-slow
## Run slow and integration tests
test-slow:
	poetry run pytest -m "slow or integration" --maxfail=3

.PHONY: test-unit
## Run unit tests only
test-unit:
	poetry run pytest -m "unit" --maxfail=10

.PHONY: test-integration
## Run integration tests only
test-integration:
	poetry run pytest -m "integration" --maxfail=3

.PHONY: test-layers
## Run layer-specific tests
test-layers:
	poetry run pytest -m "layers" --maxfail=5

.PHONY: test-processor
## Run processor-specific tests
test-processor:
	poetry run pytest -m "processor" --maxfail=3

.PHONY: test-time-series
## Run time series tests
test-time-series:
	poetry run pytest -m "time_series" --maxfail=3

.PHONY: test-sequential
## Run tests sequentially (no parallel execution)
test-sequential:
	poetry run pytest -n 0

.PHONY: test-verbose
## Run tests with verbose output
test-verbose:
	poetry run pytest -v --tb=long

.PHONY: test-benchmark
## Run performance benchmark tests
test-benchmark:
	poetry run pytest -m "performance" --benchmark-only

.PHONY: test-smoke
## Run a quick smoke test (fastest tests only)
test-smoke:
	poetry run pytest -m "fast" --maxfail=1 --tb=no -q

.PHONY: test-ultrafast
## Run tests with ultra-fast configuration (may be unstable)
test-ultrafast:
	poetry run pytest -c pytest-ultrafast.ini

.PHONY: test-fast-balanced
## Run tests with balanced fast configuration
test-fast-balanced:
	poetry run pytest -c pytest-fast.ini

.PHONY: test-micro
## Run only micro tests (fastest possible)
test-micro:
	poetry run pytest -m "micro" --maxfail=1 --tb=no -q --timeout=30

.PHONY: test-parallel-max
## Run tests with maximum parallelization
test-parallel-max:
	poetry run pytest -n logical --maxfail=3 --tb=no -q --timeout=90

.PHONY: unittests
## Run unittests (legacy command, use 'test' instead)
unittests: test

.PHONY: clean_tests
## Remove pytest cache and test artifacts
clean_tests:
	find . -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*junit_report.xml' -delete
	find . -type f -name '*.pyc' -delete
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml
	rm -f pytest.xml

.PHONY: coverage
## Generate coverage report
coverage:
	poetry run pytest --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# ------------------------------------
# Build package
# ------------------------------------

.PHONY: build_pkg
## Build the package using poetry
build_pkg:
	@echo "Start to build pkg"
ifdef HAS_POETRY
	@$(SET_VERSION_FROM_TAG)
	poetry build
else
	@echo "To build the package, you need to have poetry first"
	exit 1
endif

.PHONY: build
## Clean up cache from previous built, and build the package
build: clean_built build_pkg

.PHONY: clean_built
## Remove cache, built package, and docs directories after build or installation
clean_built:
	rm -rf dist
	find . -type f -name '*.py[co]' -delete
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# ------------------------------------
# Build doc
# ------------------------------------

.PHONY: generate_all_diagrams
## Generate all diagrams, organize them, and clean up in one command
generate_all_diagrams:
	@echo "Generating and organizing all diagrams in one step"
	./scripts/generate_all_diagrams.sh

.PHONY: generate_doc_content
## Generate documentation content from code and model architectures (DEPRECATED - use generate_all_diagrams instead)
generate_doc_content:
	@echo "NOTE: This target is deprecated. Please use 'make generate_all_diagrams' instead."
	@echo "Generating API documentation from docstrings"
	mkdir -p docs/generated/api
	poetry run python scripts/generate_docstring_docs.py
	@echo "Generating model architecture diagrams"
	mkdir -p docs/features/imgs/models
	poetry run python scripts/generate_model_diagrams.py
	@echo "Organizing documentation images"
	./scripts/organize_docs_images.sh
	@echo "Documentation content generation complete"

.PHONY: docs_deploy
## Build docs using mike
docs_deploy: generate_all_diagrams
	@echo "Starting to build docs"
	@echo "more info: https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/"
ifdef HAS_POETRY
	@$(SET_VERSION_FROM_TAG)
	poetry version -s | xargs -I {} sh -c 'echo Deploying version {} && mike deploy --push --update-aliases {} latest'
else
	@echo "To build the docs, you need to have poetry first"
	exit 1
endif

.PHONY: docs_version_list
## List available versions of the docs
docs_version_list:
	mike list

.PHONY: docs_version_serve
## Serve versioned docs
docs_version_serve:
	@echo "Start to serve versioned docs"
	mike serve

.PHONY: docs
## Create or Deploy MkDocs based documentation to GitHub pages.
deploy_doc: generate_all_diagrams
	mkdocs gh-deploy

.PHONY: serve_doc
## Test MkDocs based documentation locally.
serve_doc: generate_all_diagrams
	poetry run mkdocs serve

# ------------------------------------
# Clean All
# ------------------------------------

.PHONY: identify_unused_diagrams
## Identify potentially unused diagram files
identify_unused_diagrams:
	@echo "Scanning for potentially unused diagram files"
	poetry run python scripts/cleanup_unused_diagrams.py
	@echo "Scan complete. Check unused_diagrams_report.txt for details"

.PHONY: clean_old_diagrams
## Remove obsolete diagram directories and unused diagram files
clean_old_diagrams: identify_unused_diagrams
	@echo "Cleaning up old diagram directories"
	rm -rf docs/imgs/architectures
	@echo "Organizing documentation images"
	./scripts/organize_docs_images.sh
	@echo "Old diagram directories cleaned up"
	@echo "NOTE: Review unused_diagrams_report.txt to identify additional files to remove manually"

.PHONY: clean
## Remove cache, built package, and docs directories after build or installation
clean: clean_tests clean_built clean_old_diagrams
	@echo "Workspace cleaned"

# ------------------------------------
# Default
# ------------------------------------

.DEFAULT_GOAL := help

help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
