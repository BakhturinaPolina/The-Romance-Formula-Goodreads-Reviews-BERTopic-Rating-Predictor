# Romance Novel NLP Research Project

## Overview

This research project explores the relationship between thematic characteristics of romance novels and their popularity among readers using Natural Language Processing (NLP) techniques. The study focuses on analyzing multiple forms of reader engagement to understand how thematic elements influence reader interest, satisfaction, and overall reception.

## Research Objectives

### Primary Research Question
How do thematic characteristics of romance novels relate to their popularity among readers?

### Specific Objectives
1. **Topic Modeling**: Extract themes from romance novels across different subgenres
2. **Review Analysis**: Analyze reader reviews to identify key themes and preferences
3. **Correlation Analysis**: Examine relationships between book themes and popularity metrics
4. **Comparative Analysis**: Compare author-intended themes with reader-perceived themes

### Target Subgenres

The research focuses on modern romantic novels in these subgenres:
- Contemporary Romance
- Historical Romance  
- Paranormal Romance
- Romantic Suspense
- Romantic Fantasy
- Science Fiction Romance

## Dataset

The project uses Goodreads metadata including:
- **Books**: 348MB of romance book metadata
- **Reviews**: 1.2GB of romance-specific reviews
- **Interactions**: 2.1GB of user-book interactions
- **Genres**: 23MB of genre classifications
- **Authors**: 17MB of author information
- **Series**: 27MB of series data
- **Works**: 72MB of work-level metadata

### Data Processing Approach

The dataset uses **work-level aggregation** to handle multiple editions of the same book:

- **Individual Edition Fields**: Removed from final dataset to avoid confusion
  - `average_rating` (individual edition rating)
  - `ratings_count` (individual edition ratings count)
  - `text_reviews_count` (individual edition reviews count)

- **Work-Level Aggregated Fields**: Used for all analysis
  - `average_rating_weighted_mean` (weighted average across all editions)
  - `ratings_count_sum` (total ratings across all editions)
  - `text_reviews_count_sum` (total reviews across all editions)

This approach ensures fair comparison between books regardless of how many editions they have.

## Research Phases

### Phase 1: Data Exploration and Understanding ✅ COMPLETE
**Objective**: Comprehensive understanding of dataset structure and quality

**Activities**:
- Systematic exploration of all 9 JSON files
- Field type identification and categorization
- Data quality assessment and documentation
- Relationship mapping between files
- Sample content analysis

**Deliverables**:
- Data structure documentation
- Quality assessment reports
- Field selection rationale
- Processing strategy

### Phase 2: Data Processing and Preparation ✅ COMPLETE
**Objective**: Create clean, analysis-ready datasets

**Activities**:
- Data cleaning and standardization
- Field selection and transformation
- Quality filtering and validation
- Subgenre classification
- Author balancing and deduplication

**Deliverables**:
- Cleaned datasets for analysis (119,678 romance novels)
- Processing documentation
- Quality metrics and validation
- Corpus creation for NLP analysis

### Phase 3: Data Quality Pipeline ✅ COMPLETE
**Objective**: Implement comprehensive data quality assurance

**Activities**:
- 6-step data cleaning and validation pipeline
- Outlier detection and treatment
- Data type optimization and persistence
- Final quality validation and certification

**Deliverables**:
- Quality-certified datasets ready for analysis
- Comprehensive quality validation reports
- Data type optimization for memory efficiency
- Pipeline documentation and reproducibility

### Phase 4: NLP Analysis and Topic Modeling 🔄 NEXT PHASE
**Objective**: Extract themes and topics from text content

**Activities**:
- Text preprocessing and normalization
- BERTopic modeling for books and reviews
- Topic interpretation and validation
- Theme extraction and categorization
- Cross-validation of topic models

**Deliverables**:
- Topic models for book descriptions and reviews
- Theme categorization and interpretation
- Correlation analysis with popularity metrics
- Research findings and insights

## Current Project Status

### ✅ **Completed**
- **Data Exploration**: Comprehensive understanding of all data sources
- **Data Processing**: Clean, analysis-ready datasets created
- **Data Cleaning**: Titles normalized, descriptions cleaned, series structured
- **Quality Validation**: Data quality assessed and documented
- **Data Quality Pipeline**: 6-step pipeline implemented and validated
- **Code Cleanup**: Unused code archived, working modules consolidated

### 🔄 **In Progress**
- **NLP Analysis Setup**: Preparing for topic modeling and theme extraction
- **Documentation Updates**: Consolidating project documentation

### 📋 **Next Steps**
- **Text Preprocessing**: Apply NLP preprocessing to quality-certified datasets
- **Topic Modeling**: Implement BERTopic for theme extraction
- **Subgenre Classification**: Parse popular shelves for standardized categories
- **Correlation Analysis**: Analyze theme-popularity relationships

## Project Structure

### Active Components
```
romance-novel-nlp-research/
├── src/                                          # Clean, focused source code
│   ├── data_quality/                             # Complete 6-step data quality pipeline
│   │   ├── pipeline_runner.py                    # Main pipeline runner
│   │   ├── step1_missing_values_cleaning.py      # Step 1: Missing Values Cleaning
│   │   ├── step2_duplicate_detection.py          # Step 2: Duplicate Detection
│   │   ├── step3_data_type_validation.py         # Step 3: Data Type Validation
│   │   ├── step4_outlier_detection.py            # Step 4: Outlier Detection
│   │   ├── step4_outlier_treatment.py            # Step 4: Outlier Treatment
│   │   ├── step5_data_type_optimization.py       # Step 5: Data Type Optimization
│   │   ├── step6_final_quality_validation.py     # Step 6: Final Quality Validation
│   │   └── outputs/                              # Pipeline execution outputs
│   │       ├── missing_values_cleaning/          # Step 1 outputs
│   │       ├── duplicate_detection/              # Step 2 outputs
│   │       ├── data_type_validation/             # Step 3 outputs
│   │       ├── outlier_detection/                # Step 4 outputs
│   │       ├── data_type_optimization/           # Step 5 outputs
│   │       └── final_quality_validation/         # Step 6 outputs
│   └── csv_building/                             # CSV generation module
│       ├── final_csv_builder.py                  # Main CSV builder
│       └── run_builder.py                        # Builder runner script
├── data/
│   ├── raw/                                      # Original Goodreads JSON files (9 files)
│   ├── intermediate/                              # Temporary processing outputs
│   └── processed/                                # Final cleaned datasets
├── logs/                                          # Current execution logs
├── docs/                                          # Project documentation
├── notebooks/                                     # Jupyter notebooks
└── archive/                                       # Archived development history
    ├── cleanup_20250105/                         # January 5, 2025 cleanup
    ├── cleanup_20250106/                         # January 6, 2025 cleanup
    ├── data_quality_archive_20250902/            # September 2025 data quality archive
    ├── data_quality_artifacts/                   # Development artifacts
    ├── pipeline_outputs_20250902/                # Archived pipeline outputs
    ├── processed_data_20250904/                  # Archived processed data
    └── unused_code/                              # Legacy and experimental code
```

### Key Files
- **Main Pipeline Runner**: `src/data_quality/pipeline_runner.py` (Complete 6-step pipeline)
- **CSV Builder**: `src/csv_building/final_csv_builder.py` (Dataset generation)
- **Builder Runner**: `src/csv_building/run_builder.py` (CSV generation runner)
- **Raw Data**: `data/raw/` (9 Goodreads JSON files)
- **Pipeline Outputs**: `src/data_quality/outputs/` (All pipeline step outputs)
- **Quality Reports**: Archived in `archive/processed_data_20250904/`

## Repository Maintenance

### Recent Cleanup Activities (January 2025)
The repository has undergone comprehensive cleanup to maintain a clean, focused codebase:

**January 6, 2025 Cleanup**:
- **Python Cache Files**: Removed `__pycache__` directories from `src/data_quality/` and `src/csv_building/`
- **Empty Log Files**: Archived 5 empty log files (0 bytes) that were never used
- **Duplicate Outputs**: Archived earlier versions of quality reports and unprocessed batch files
- **Development Tools**: Archived 3 diagnostic scripts used during development
- **Alternative Runners**: Archived 4 redundant runner scripts that duplicated core functionality

**January 5, 2025 Cleanup**:
- **Outdated Logs**: Archived 20+ log files from September 2025 (outdated)
- **Duplicate Outputs**: Archived 8 duplicate output files with older timestamps
- **Old Reports**: Archived 7 superseded analysis reports and summaries

**Archive Organization**:
- All development history preserved in `archive/` directory
- Clean separation between active code and development artifacts
- Easy recovery process for any archived files if needed
- Streamlined `src/` directory with only essential functionality

### Current Repository State
- **Clean Source Code**: No cache files or redundant scripts in active directories
- **Focused Structure**: Only core functionality remains in `src/` directory
- **Preserved History**: All development artifacts safely archived
- **Ready for Phase 4**: Clean codebase prepared for NLP analysis implementation

## Data Quality Pipeline

### Complete 6-Step Pipeline Architecture
The project implements a comprehensive data quality pipeline with all steps currently active:

1. **Step 1**: Missing Values Cleaning
   - **Implementation**: `step1_missing_values_cleaning.py`
   - **Outputs**: `src/data_quality/outputs/missing_values_cleaning/`
   - **Purpose**: Handle and document missing value patterns

2. **Step 2**: Duplicate Detection
   - **Implementation**: `step2_duplicate_detection.py`
   - **Outputs**: `src/data_quality/outputs/duplicate_detection/`
   - **Purpose**: Identify and resolve duplicate records

3. **Step 3**: Data Type Validation
   - **Implementation**: `step3_data_type_validation.py`
   - **Outputs**: `src/data_quality/outputs/data_type_validation/`
   - **Purpose**: Validate and standardize data types

4. **Step 4**: Outlier Detection & Treatment
   - **Detection**: `step4_outlier_detection.py`
   - **Treatment**: `step4_outlier_treatment.py`
   - **Outputs**: `src/data_quality/outputs/outlier_detection/`
   - **Purpose**: Identify and treat statistical outliers

5. **Step 5**: Data Type Optimization & Persistence
   - **Implementation**: `step5_data_type_optimization.py`
   - **Outputs**: `src/data_quality/outputs/data_type_optimization/`
   - **Purpose**: Optimize memory usage and create persistent formats

6. **Step 6**: Final Quality Validation & Certification
   - **Implementation**: `step6_final_quality_validation.py`
   - **Outputs**: `src/data_quality/outputs/final_quality_validation/`
   - **Purpose**: Final quality assessment and certification

### Pipeline Execution
- **Main Runner**: `src/data_quality/pipeline_runner.py` (Executes all 6 steps)
- **Individual Steps**: Each step can be run independently
- **Output Organization**: All outputs organized by step in `src/data_quality/outputs/`
- **Quality Reports**: Comprehensive reporting for each step

### Pipeline Benefits
- **Data Integrity**: Comprehensive validation and quality assurance
- **Memory Efficiency**: Optimized data types for large datasets
- **Reproducibility**: Documented pipeline with clear outputs
- **Quality Certification**: Final datasets meet quality standards
- **Modular Design**: Each step can be run independently or as part of full pipeline

## Working Code Architecture

### Data Quality Module
**Purpose**: Implement comprehensive data quality assurance pipeline

**Key Features**:
- **6-Step Pipeline**: Complete data cleaning and validation workflow
- **Quality Gates**: Automated quality threshold validation
- **Data Type Optimization**: Memory efficiency improvements
- **Comprehensive Reporting**: Detailed quality metrics and certification
- **Clean Architecture**: Single responsibility per step, well-documented

### CSV Building Module
**Purpose**: Create clean, analysis-ready romance novel datasets

**Key Features**:
- **Simplified Structure**: 19 columns (down from 24) for essential data only
- **High Performance**: Processes full dataset in ~15 minutes
- **Data Quality**: Comprehensive validation and reporting
- **Clean Architecture**: Single responsibility, well-documented
- **Work-Level Aggregation**: Handles multiple editions efficiently

**Output Structure (19 columns)**:
- **Core identifiers**: `work_id`, `book_id_list_en`
- **Book metadata**: `title`, `publication_year`, `language_codes_en`
- **Content**: `description`, `popular_shelves`, `genres`
- **Author info**: `author_id`, `author_name`, `author_average_rating`, `author_ratings_count`
- **Series data**: `series_id`, `series_title`, `series_works_count`
- **Metrics**: `ratings_count_sum`, `text_reviews_count_sum`, `average_rating_weighted_mean`

## Data Quality

### Current Dataset Characteristics
- **Size**: 119,678 romance novels
- **Coverage**: 2000-2020 publication years
- **Completeness**: 95%+ for core fields
- **Text Quality**: Clean descriptions with minimal HTML artifacts
- **Series Coverage**: 67% of books properly categorized
- **Structure**: Simplified 19-column format for research efficiency
- **Quality Status**: Quality-certified through 6-step pipeline

### Data Strengths
- Complete core metadata (title, author, publication year)
- Clean, normalized text content
- Comprehensive popularity metrics
- Structured series information
- Subgenre signals in popular shelves
- Work-level aggregation for fair comparison
- Quality-assured through systematic validation

## Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment with required packages
- Access to processed datasets

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd romance-novel-nlp-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Analysis
```bash
# Start Jupyter notebook
jupyter notebook

# Open notebooks/02_final_dataset_eda_and_cleaning.ipynb
# This notebook provides comprehensive EDA of the cleaned dataset
```

### Processing New Data
```bash
# Run the CSV builder
cd src/csv_building
python run_builder.py

# Choose processing mode:
# 1. Test with sample (recommended for first run)
# 2. Process full dataset
```

### Running Data Quality Pipeline
```bash
# Run complete 6-step pipeline
cd src/data_quality
python pipeline_runner.py

# Or run individual steps:
python step1_missing_values_cleaning.py
python step2_duplicate_detection.py
python step3_data_type_validation.py
python step4_outlier_detection.py
python step4_outlier_treatment.py
python step5_data_type_optimization.py
python step6_final_quality_validation.py
```

### Pipeline Outputs
All pipeline outputs are organized in `src/data_quality/outputs/`:
- Each step creates its own subdirectory
- Quality reports and processed data files are saved with timestamps
- Final optimized datasets are available in multiple formats (CSV, Parquet, Pickle)

## Documentation

### Current Documentation
- **README.md**: This file - project overview and current status
- **archive/ARCHIVE_SUMMARY.md**: Complete archive organization and cleanup history
- **archive/cleanup_20250106/CLEANUP_SUMMARY.md**: Latest cleanup details (January 2025)
- **src/data_quality/outputs/**: Pipeline execution reports and quality assessments
- **data/raw/README.md**: Raw data file descriptions and structure

### Code Documentation
- **Inline Documentation**: Comprehensive docstrings and comments
- **Module Structure**: Clear separation of concerns
- **Error Handling**: Graceful failure with detailed logging
- **Performance Notes**: Optimization strategies documented
- **Pipeline Documentation**: Step-by-step pipeline documentation

## Contributing

This is a research project focused on romance novel analysis. Contributions should align with the research objectives and maintain data quality standards.

**Development Guidelines**:
- Keep code focused and minimal
- Archive unused code regularly
- Maintain comprehensive documentation
- Follow established patterns and conventions
- Respect the 6-step data quality pipeline

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Data Source**: UCSD Goodreads Book Graph
- **Research Focus**: Romance novel thematic analysis and popularity correlation
- **Methodology**: NLP-based topic modeling and text analysis

---

**Last Updated**: January 2025  
**Status**: Phase 3 Complete, Phase 4 Ready  
**Code Status**: Clean, focused codebase with comprehensive data quality pipeline  
**Repository Status**: Streamlined after comprehensive cleanup (January 2025)  
**Next Milestone**: Topic modeling implementation and theme extraction