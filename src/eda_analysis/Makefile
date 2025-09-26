# Makefile for EDA Analysis Pipeline

# Default paths
AUDIT_DIR = audit_outputs_full
PARSED_PARQUET = parse_outputs_full/parsed_books_20250926_002946.parquet
OUTDIR = shelf_norm_outputs

# Step 1: Run audit
audit:
	python 00_load_audit.py --data-path ../../data/processed/romance_books_main_final.csv --output-dir $(AUDIT_DIR)

# Step 1b: Parse lists
parse:
	python 01_parse_lists.py --data-path ../../data/processed/romance_books_main_final.csv --output-dir parse_outputs_full

# Step 2: Shelf normalization
shelf-norm:
	python shelf_normalize.py \
		--from-audit $(AUDIT_DIR) \
		--parsed-parquet $(PARSED_PARQUET) \
		--outdir $(OUTDIR)

# Run all steps in sequence
pipeline: audit parse shelf-norm

# Test the shelf normalization
test-shelf:
	python test_shelf_normalize.py

# Clean outputs
clean:
	rm -rf $(AUDIT_DIR) parse_outputs_full $(OUTDIR)

# Install dependencies
install:
	pip install -r requirements_audit.txt

# Help
help:
	@echo "Available targets:"
	@echo "  audit      - Run data audit (Step 1)"
	@echo "  parse      - Parse list fields (Step 1b)"
	@echo "  shelf-norm - Run shelf normalization (Step 2)"
	@echo "  pipeline   - Run all steps in sequence"
	@echo "  test-shelf - Test shelf normalization"
	@echo "  clean      - Remove all output directories"
	@echo "  install    - Install dependencies"
	@echo "  help       - Show this help"

.PHONY: audit parse shelf-norm pipeline test-shelf clean install help
