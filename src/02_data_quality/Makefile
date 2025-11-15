# Makefile for Data Audit Pipeline

# Default paths
DATA_PATH = ../../data/processed/romance_books_main_final.csv
AUDIT_OUTDIR = ./audit_outputs
PARSE_OUTDIR = ./parse_outputs

.PHONY: all install audit parse clean help test

all: audit parse

install:
	pip install -r requirements.txt

# Run complete data audit
audit:
	python data_audit/core/data_auditor.py \
		--data-path $(DATA_PATH) \
		--output-dir $(AUDIT_OUTDIR)

# Run list parsing
parse:
	python data_audit/parsing/list_parser.py \
		--data-path $(DATA_PATH) \
		--output-dir $(PARSE_OUTDIR)

# Run audit with verbose logging
audit-verbose:
	python data_audit/core/data_auditor.py \
		--data-path $(DATA_PATH) \
		--output-dir $(AUDIT_OUTDIR) \
		--verbose

# Run parsing with verbose logging
parse-verbose:
	python data_audit/parsing/list_parser.py \
		--data-path $(DATA_PATH) \
		--output-dir $(PARSE_OUTDIR) \
		--verbose

# Run audit only (without parsing)
audit-only: audit

# Run parsing only (without audit)
parse-only: parse

# Clean all output directories
clean:
	rm -rf $(AUDIT_OUTDIR) $(PARSE_OUTDIR)

# Run tests
test:
	python -m pytest tests/ -v

# Run tests with coverage
test-coverage:
	python -m pytest tests/ --cov=. --cov-report=html

# Format code
format:
	black .
	flake8 .

# Type checking
type-check:
	mypy .


# Generate documentation
docs:
	python -c "from data_audit.core.data_auditor import DataAuditor; help(DataAuditor)"

# Help
help:
	@echo "Available targets:"
	@echo "  all           - Run audit and parse (default)"
	@echo "  install       - Install dependencies"
	@echo "  audit         - Run data audit analysis"
	@echo "  parse         - Run list parsing"
	@echo "  audit-verbose - Run audit with verbose logging"
	@echo "  parse-verbose - Run parse with verbose logging"
	@echo "  audit-only    - Run audit only"
	@echo "  parse-only    - Run parse only"
	@echo "  clean         - Remove all output directories"
	@echo "  test          - Run test suite"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  format        - Format code with black and flake8"
	@echo "  type-check    - Run type checking with mypy"
	@echo "  docs          - Generate documentation"
	@echo "  help          - Show this help"
