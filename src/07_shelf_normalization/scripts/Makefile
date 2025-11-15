# Makefile for Shelf Normalization Pipeline

# Default paths - adjust these based on your data locations
# Paths are relative to scripts/ directory (one level up from stage root)
AUDIT_DIR = ../../../data/processed/audit_outputs
PARSED_PARQUET = ../../../data/processed/parsed_books.parquet
NORMALIZE_OUTDIR = ../outputs/normalize
BRIDGE_OUTDIR = ../outputs/bridge
DIAGNOSTICS_OUTDIR = ../outputs/diagnostics

# Step 1: Run shelf normalization
normalize:
	python ../core/shelf_normalize.py \
		--from-audit $(AUDIT_DIR) \
		--parsed-parquet $(PARSED_PARQUET) \
		--outdir $(NORMALIZE_OUTDIR) \
		--n-top-shelves 100000 \
		--jw-threshold 0.94 \
		--j3-threshold 0.80 \
		--edit-max 1 \
		--min-seg-evidence 5 \
		--zipf-min 3.0

# Step 2: Bridge normalization with parsed data
bridge:
	python ../bridge/bridge_audit_normalize.py \
		--parsed $(PARSED_PARQUET) \
		--canon $(NORMALIZE_OUTDIR)/shelf_canonical.csv \
		--segments $(NORMALIZE_OUTDIR)/shelf_segments.csv \
		--noncontent $(NORMALIZE_OUTDIR)/noncontent_shelves.csv \
		--alias $(NORMALIZE_OUTDIR)/shelf_alias_candidates.csv \
		--out $(BRIDGE_OUTDIR)

# Step 3: Run diagnostics on normalization outputs
diagnostics:
	python ../diagnostics/diagnostics_explore.py \
		--in-dir $(NORMALIZE_OUTDIR) \
		--out-dir $(DIAGNOSTICS_OUTDIR) \
		--alias-dryrun-size 10000 \
		--alias-topk 200 \
		--verbose-logs

# Step 4: Validate bridge outputs
validate:
	python ../diagnostics/validate_bridge.py \
		--books $(BRIDGE_OUTDIR)/books_with_shelf_norm.parquet \
		--raw-long $(BRIDGE_OUTDIR)/shelves_raw_long.parquet \
		--canon-long $(BRIDGE_OUTDIR)/shelves_canon_long.parquet \
		--segments-long $(BRIDGE_OUTDIR)/segments_long.parquet \
		--canonical-csv $(NORMALIZE_OUTDIR)/shelf_canonical.csv \
		--segments-csv $(NORMALIZE_OUTDIR)/shelf_segments.csv \
		--noncontent-csv $(NORMALIZE_OUTDIR)/noncontent_shelves.csv \
		--alias-csv $(NORMALIZE_OUTDIR)/shelf_alias_candidates.csv \
		--out $(DIAGNOSTICS_OUTDIR)/bridge_qa \
		--print-examples \
		--verbose-logs

# Run complete pipeline
pipeline: normalize bridge diagnostics validate

# Install dependencies
install:
	pip install -r ../requirements.txt

# Create output directories
setup:
	mkdir -p $(NORMALIZE_OUTDIR) $(BRIDGE_OUTDIR) $(DIAGNOSTICS_OUTDIR)

# Clean outputs
clean:
	rm -rf $(NORMALIZE_OUTDIR) $(BRIDGE_OUTDIR) $(DIAGNOSTICS_OUTDIR)

# Help
help:
	@echo "Available targets:"
	@echo "  normalize   - Run shelf normalization (Step 1)"
	@echo "  bridge      - Bridge normalization with parsed data (Step 2)"
	@echo "  diagnostics - Run diagnostics on normalization outputs (Step 3)"
	@echo "  validate    - Validate bridge outputs (Step 4)"
	@echo "  pipeline    - Run complete pipeline (all steps)"
	@echo "  install     - Install dependencies"
	@echo "  setup       - Create output directories"
	@echo "  clean       - Remove all output directories"
	@echo "  help        - Show this help"

.PHONY: normalize bridge diagnostics validate pipeline install setup clean help
