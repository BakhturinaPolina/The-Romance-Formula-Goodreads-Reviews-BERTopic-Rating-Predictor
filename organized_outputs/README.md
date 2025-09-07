# Organized Outputs - Romance Novel NLP Research

This directory contains all outputs from the romance novel data cleaning and analysis pipeline, organized in a clear, structured format.

## 📁 Directory Structure

```
organized_outputs/
├── datasets/                    # All dataset files
│   ├── step_by_step/           # Datasets after each pipeline step
│   └── specialized_versions/   # Different versions of final datasets
├── logs/                       # All log files from pipeline execution
├── reports/                    # All analysis reports
│   ├── json/                   # JSON format reports
│   └── markdown/               # Markdown format reports
└── visualizations/             # All generated plots and charts
```

## 📊 Datasets

### Step-by-Step Pipeline Datasets
These datasets show the data transformation at each step of the cleaning pipeline:

1. **`01_raw_integrated_dataset.csv`** - Raw integrated data from JSON sources (119,678 records)
2. **`02_after_missing_values_treatment.pkl`** - After removing records with missing critical values (80,804 records)
3. **`03_after_duplicate_detection.pkl`** - After duplicate detection and resolution (80,804 records)
4. **`04_after_data_type_validation.pkl`** - After data type validation and correction (80,804 records)
5. **`05_after_outlier_treatment.pkl`** - After outlier detection and treatment (80,755 records)
6. **`06_after_data_type_optimization.parquet`** - After data type optimization for efficiency (80,755 records)
7. **`07_final_nlp_preprocessed.csv`** - Final dataset with NLP text preprocessing (80,755 records)

### Specialized Dataset Versions
Different versions of the final dataset for specific research needs:

- **`complete_dataset_all_columns.csv/.pkl`** - Complete final dataset with all 30 columns
- **`core_research_dataset_23_columns.csv/.pkl`** - Streamlined dataset with 23 essential research columns
- **`false_duplicates_similar_titles.csv/.pkl`** - Books with similar titles by different authors (21,105 records)

## 📋 Reports

### JSON Reports
- **Pipeline Execution Reports**: Step-by-step execution logs and statistics
- **Data Quality Reports**: Missing values, duplicates, outliers, validation results
- **Text Preprocessing Reports**: NLP processing statistics and validation
- **EDA Analysis Reports**: Exploratory data analysis results

### Markdown Reports
- **`dataset_documentation.md`** - Comprehensive dataset documentation
- **`DATA_QUALITY_ANALYSIS_AND_FIXES.md`** - Data quality analysis and fixes applied
- **`DETAILED_DATA_ANALYSIS_REPORT.md`** - Detailed analysis of the dataset
- **`PIPELINE_EXECUTION_ANALYSIS_REPORT.md`** - Pipeline execution analysis

## 📈 Visualizations

### Distribution Analysis
- **`figure_01_publication_year_histogram_cleaned.png`** - Publication year distribution
- **`figure_03_num_pages_median_histogram_cleaned.png`** - Page count distribution
- **`figure_05_ratings_count_sum_histogram_cleaned.png`** - Ratings count distribution
- **`figure_07_average_rating_weighted_mean_histogram_cleaned.png`** - Rating distribution

### Boxplot Analysis
- **`figure_02_publication_year_boxplot.png`** - Publication year boxplots
- **`figure_04_num_pages_median_boxplot.png`** - Page count boxplots
- **`figure_06_ratings_count_sum_boxplot.png`** - Ratings count boxplots
- **`figure_08_average_rating_weighted_mean_boxplot.png`** - Rating boxplots

### Categorical Analysis
- **`figure_09_genres_bar_chart.png`** - Genre distribution
- **`figure_10_in_series_bar_chart.png`** - Series participation
- **`figure_11_language_codes_en_bar_chart.png`** - Language distribution
- **`figure_14_series_classification_pie.png`** - Series classification

### Text Analysis
- **`figure_12_description_words_distribution.png`** - Description word count distribution
- **`figure_13_popular_shelves_tags_distribution.png`** - Popular shelves distribution

### Pipeline Comparison
- **`figure_1_before_after_distributions.png`** - Before/after cleaning comparison
- **`figure_2_cleaned_data_summary.png`** - Final cleaned data summary

## 📝 Logs

- **`data_quality_pipeline_*.log`** - Complete pipeline execution logs with timestamps

## 🔢 Data Reduction Summary

| **Stage** | **Records** | **Reduction** | **Cumulative Reduction** |
|-----------|-------------|---------------|-------------------------|
| Raw Integrated CSV | 119,678 | - | - |
| After Missing Values Treatment | 80,804 | -38,874 (-32.5%) | -32.5% |
| After Duplicate Detection | 80,804 | 0 (0%) | -32.5% |
| After Data Type Validation | 80,804 | 0 (0%) | -32.5% |
| After Outlier Treatment | 80,755 | -49 (-0.06%) | -32.52% |
| **Final Dataset** | **80,755** | **-38,923 (-32.52%)** | **-32.52%** |

## 🎯 Usage Recommendations

### For NLP Research
- Use **`core_research_dataset_23_columns.csv`** for efficient processing
- Use **`complete_dataset_all_columns.csv`** for comprehensive analysis

### For Data Quality Analysis
- Use **`false_duplicates_similar_titles.csv`** to validate duplicate detection
- Review step-by-step datasets to understand data transformation

### For Visualization
- All PNG files are publication-ready with consistent styling
- Use Antique color palette for academic presentations

### For Reproducibility
- All logs provide complete execution history
- JSON reports contain detailed statistics and parameters
- Markdown reports provide human-readable analysis

## 📚 File Formats

- **CSV**: Human-readable, compatible with most tools
- **PKL**: Python pickle format, preserves data types and is faster to load
- **Parquet**: Efficient columnar format, good for large datasets
- **JSON**: Structured reports with detailed statistics
- **PNG**: High-quality visualizations ready for publication

---

*This organization was created to provide clear access to all outputs from the romance novel NLP research pipeline. All files maintain their original timestamps and are organized by type and purpose for easy navigation.*
