# Replication Guide

This guide explains how to adapt the pipeline to work with other datasets.

## Dataset Requirements

### Required Data Format

Your dataset should include:

1. **Book Metadata** (CSV or JSON):
   - Book identifiers (ID, work_id)
   - Titles and authors
   - Publication information
   - Ratings and review counts
   - Genre classifications (optional)

2. **Review Data** (CSV or JSON):
   - Review identifiers
   - Review text
   - Ratings
   - Book identifiers (to link with metadata)

### Data Structure

The pipeline expects:
- Unique book identifiers
- Text reviews with associated book IDs
- Metadata fields for analysis

## Adaptation Steps

### 1. Data Integration Stage

**Location**: `src/01_data_integration/`

**Changes Needed**:
- Modify `final_csv_builder.py` to read your data format
- Update field mappings to match your dataset structure
- Adjust work-level aggregation logic if needed

**Key Files**:
- `final_csv_builder.py`: Main data integration script
- `run_builder.py`: Runner script

### 2. Data Quality Stage

**Location**: `src/02_data_quality/`

**Changes Needed**:
- Review quality thresholds in `pipeline_runner.py`
- Adjust outlier detection parameters if needed
- Update validation rules for your data types

**Key Files**:
- `pipeline_runner.py`: Main quality pipeline
- `step*.py`: Individual quality steps

### 3. Text Preprocessing Stage

**Location**: `src/03_text_preprocessing/`

**Changes Needed**:
- Adjust text cleaning rules for your data format
- Update genre categorization if using genres
- Modify normalization rules as needed

**Key Files**:
- `text_preprocessor.py`: Main preprocessing script

### 4. Review Extraction Stage

**Location**: `src/04_review_extraction/`

**Changes Needed**:
- Update data loading to match your review format
- Adjust language detection if needed
- Modify filtering criteria

**Key Files**:
- `extract_reviews.py`: Main extraction script

### 5. Topic Modeling Stage

**Location**: `src/05_topic_modeling/`

**Changes Needed**:
- Update data loading paths
- Adjust hyperparameter search space if needed
- Modify evaluation metrics as appropriate

**Key Files**:
- `BERTopic_OCTIS/bertopic_plus_octis.py`: Main topic modeling
- `bertopic_preparation/prepare_bertopic_input.py`: Input preparation

### 6. Configuration Updates

**Update Paths**:
- Modify data paths in scripts to point to your data
- Update output paths as needed
- Adjust file naming conventions

**Update Parameters**:
- Review stage-specific parameters
- Adjust thresholds and limits
- Update sample sizes if needed

## Testing Your Adaptation

### 1. Start Small

- Test with a small sample dataset first
- Verify each stage produces expected outputs
- Check data quality at each step

### 2. Validate Outputs

- Compare outputs with expected formats
- Check data quality metrics
- Verify topic modeling results make sense

### 3. Iterate

- Adjust parameters based on results
- Refine preprocessing steps
- Optimize for your specific dataset

## Common Adaptations

### Different Data Format

If your data is in a different format:
1. Create a data loader in `src/01_data_integration/`
2. Convert to expected format or modify pipeline to accept your format
3. Update field mappings throughout pipeline

### Different Languages

If working with non-English data:
1. Update language detection in `src/04_review_extraction/`
2. Install appropriate spaCy models
3. Adjust text preprocessing for your language

### Different Metadata Fields

If your metadata has different fields:
1. Update field mappings in `src/01_data_integration/`
2. Adjust quality checks in `src/02_data_quality/`
3. Update analysis scripts to use your fields

## Getting Help

- Review stage-specific README files
- Check example configurations
- Examine existing code for patterns
- Test incrementally and validate at each step

## Best Practices

1. **Keep Original Data**: Always preserve raw data
2. **Document Changes**: Note all adaptations made
3. **Version Control**: Use git to track changes
4. **Test Thoroughly**: Validate each stage before proceeding
5. **Maintain Structure**: Keep the stage-based organization

