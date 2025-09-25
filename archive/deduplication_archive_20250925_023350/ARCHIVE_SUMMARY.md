# Deduplication Module Archive Summary

**Archive Date**: September 25, 2025  
**Archive Location**: `archive/deduplication_archive_20250925_023350/`

## Overview

This archive contains unneeded code from the `src/deduplication/` module that has been superseded by newer, more comprehensive implementations or is no longer actively used.

## Archived Files

### 1. `post_cluster_diagnostics.py`
**Reason for Archival**: Functionality integrated into main pipeline
- **Original Purpose**: Standalone diagnostic tool for cluster quality assessment
- **Current Status**: All diagnostic functionality has been integrated into `dedupe_pipeline.py` via the `diagnose` command
- **Replacement**: Use `python dedupe_pipeline.py diagnose output_dir --threshold 0.6`
- **Key Functions Moved**:
  - `validate_samples_against_filtered()` → integrated into main pipeline
  - `rebuild_edge_samples_from_filtered()` → integrated into main pipeline
  - `cluster_cohesion_metrics()` → integrated into main pipeline
  - `flag_low_quality_clusters()` → integrated into main pipeline

### 2. `example_usage.py`
**Reason for Archival**: Demo/example code no longer needed
- **Original Purpose**: Example script demonstrating pipeline usage
- **Current Status**: Superseded by comprehensive documentation in `README_COMPLETE.md`
- **Replacement**: Refer to `README_COMPLETE.md` for complete usage examples and workflows

### 3. `README.md`
**Reason for Archival**: Superseded by more comprehensive documentation
- **Original Purpose**: Basic documentation for the enhanced dedupe pipeline
- **Current Status**: Superseded by `README_COMPLETE.md` which provides complete pipeline documentation
- **Replacement**: Use `README_COMPLETE.md` for all documentation needs

### 4. `graph_refine.py`
**Reason for Archival**: Functionality consolidated into main pipeline
- **Original Purpose**: Graph building and refinement algorithms with size-aware quality assessment
- **Current Status**: All functionality consolidated into `dedupe_pipeline.py`
- **Replacement**: Use `python dedupe_pipeline.py graph-build` and `python dedupe_pipeline.py size-aware-quality`
- **Key Functions Moved**:
  - `GraphBuildCfg` → consolidated into main pipeline
  - `build_clean_graph()` → consolidated into main pipeline
  - `QualityCfg` → consolidated into main pipeline
  - `size_aware_flags()` → consolidated into main pipeline

### 5. `graph_grow.py`
**Reason for Archival**: Functionality consolidated into main pipeline
- **Original Purpose**: Triangle-based cluster growth algorithms
- **Current Status**: All functionality consolidated into `dedupe_pipeline.py`
- **Replacement**: Use `python dedupe_pipeline.py refined-grow`
- **Key Functions Moved**:
  - `GraphGrowCfg` → consolidated into main pipeline
  - `grow_by_triangles()` → consolidated into main pipeline
  - `_grow_once()` → consolidated into main pipeline

### 6. `grow_runner.py`
**Reason for Archival**: Functionality consolidated into main pipeline
- **Original Purpose**: Growth pipeline runner with CLI interface
- **Current Status**: All functionality consolidated into `dedupe_pipeline.py`
- **Replacement**: Use `python dedupe_pipeline.py refined-grow`
- **Key Functions Moved**:
  - `refined_grow()` → consolidated as `cli_refined_grow` in main pipeline

## Current Active Components

The following files remain in `src/deduplication/` as the active, maintained codebase:

### Core Pipeline
- `dedupe_pipeline.py` - **Consolidated main pipeline** with all functionality:
  - Basic filtering and clustering
  - Graph building and refinement algorithms
  - Triangle-based cluster growth algorithms
  - Size-aware quality assessment
  - All CLI commands and diagnostics
- `mapping_export.py` - Frequency-based medoid export and normalization (kept separate due to distinct functionality)

### Documentation
- `README_COMPLETE.md` - Comprehensive documentation for the complete pipeline

### Module Files
- `__init__.py` - Module initialization

## Migration Notes

### For Users of `post_cluster_diagnostics.py`
- All diagnostic functionality is now available via the main pipeline
- Use `python dedupe_pipeline.py diagnose output_dir --threshold 0.6` instead
- The main pipeline provides the same validation and quality assessment features

### For Users of `example_usage.py`
- Refer to `README_COMPLETE.md` for comprehensive usage examples
- The complete workflow examples in the documentation are more up-to-date and comprehensive

### For Users of `README.md`
- Use `README_COMPLETE.md` for all documentation needs
- The complete documentation includes all features and provides better organization

## Archive Contents

```
archive/deduplication_archive_20250925_023350/
├── ARCHIVE_SUMMARY.md           # This file
├── post_cluster_diagnostics.py  # Standalone diagnostic tool (functionality integrated)
├── example_usage.py             # Demo/example script (superseded by documentation)
├── README.md                    # Basic documentation (superseded by README_COMPLETE.md)
├── graph_refine.py              # Graph building algorithms (consolidated into main pipeline)
├── graph_grow.py                # Triangle-based growth algorithms (consolidated into main pipeline)
└── grow_runner.py               # Growth pipeline runner (consolidated into main pipeline)
```

## Restoration

If any of these files need to be restored, they can be moved back from this archive directory. However, it is recommended to use the current active components as they provide better functionality and integration.

## Contact

For questions about this archive or the deduplication module, refer to the current documentation in `README_COMPLETE.md` or the active source code in `src/deduplication/`.
