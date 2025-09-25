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
- **Complete Data Cleaning Pipeline**: Full pipeline from raw data to final analysis
- **NLP Text Preprocessing**: HTML cleaning, text normalization, genre categorization
- **EDA Analysis**: Comprehensive exploratory data analysis with statistical insights
- **Repository Organization**: All outputs centralized and organized
- **Repository Cleanup**: Comprehensive cleanup with all files safely archived
- **Dataset Specialization**: Multiple dataset versions created for different research needs

### 🔄 **In Progress**
- **Documentation Updates**: Finalizing project documentation

### 📋 **Next Steps**
- **Topic Modeling**: Implement BERTopic for theme extraction
- **Subgenre Classification**: Parse popular shelves for standardized categories
- **Correlation Analysis**: Analyze theme-popularity relationships
- **Research Paper**: Prepare findings for academic publication

## Project Structure

### Current Clean Repository Structure
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
│   │   ├── comprehensive_data_analysis.py        # Comprehensive data analysis
│   │   └── comprehensive_data_cleaner.py         # Comprehensive data cleaning
│   ├── csv_building/                             # CSV generation module
│   │   ├── final_csv_builder.py                  # Main CSV builder
│   │   └── run_builder.py                        # Builder runner script
│   ├── nlp_preprocessing/                        # NLP text preprocessing
│   │   ├── text_preprocessor.py                  # Main text preprocessor
│   │   ├── run_preprocessor.py                   # Preprocessor runner
│   │   └── test_preprocessor.py                  # Preprocessor tests
│   └── eda_analysis/                             # Exploratory data analysis
│       ├── eda_analysis_unusual_page_counts_notebook.ipynb  # EDA notebook
│       ├── logs/                                 # EDA execution logs
│       ├── outputs/                              # EDA outputs
│       └── README.md                             # EDA documentation
├── data/
│   ├── raw/                                      # Original Goodreads JSON files (9 files)
│   ├── intermediate/                              # Temporary processing outputs
│   └── processed/                                # Current datasets (9 files)
│       ├── final_books_2000_2020_en_enhanced_20250907_013708.csv
│       ├── romance_novels_text_preprocessed_20250907_015606.csv
│       ├── text_preprocessing_report_20250907_015613.json
│       ├── romance_books_main_cleaned.csv
│       ├── romance_books_main_final.csv
│       ├── romance_books_anthologies_collections.csv
│       ├── romance_books_short_form.csv
│       ├── romance_books_unexplainably_short.csv
│       └── romance_novels_main_cleaned_no_duplicates.csv
├── organized_outputs/                            # All organized outputs
│   ├── datasets/                                 # All dataset versions
│   │   ├── step_by_step/                         # 7 datasets showing pipeline progression
│   │   └── specialized_versions/                 # 9 specialized dataset versions
│   ├── logs/                                     # All pipeline execution logs
│   ├── reports/                                  # All analysis reports
│   │   ├── json/                                 # 20 JSON reports with statistics
│   │   └── markdown/                             # 5 Markdown analysis reports
│   ├── visualizations/                           # Publication-ready plots
│   └── README.md                                 # Complete output documentation
└── archive/                                      # Archived development history
    ├── cleanup_20250105/                         # January 5, 2025 cleanup
    ├── cleanup_20250106/                         # January 6, 2025 cleanup
    ├── cleanup_20250907/                         # September 7, 2025 comprehensive cleanup
    ├── data_quality_archive_20250902/            # September 2025 data quality archive
    ├── data_quality_artifacts/                   # Development artifacts
    ├── pipeline_outputs_20250902/                # Archived pipeline outputs
    ├── processed_data_20250904/                  # Archived processed data
    └── unused_code/                              # Legacy and experimental code
```

### Key Files
- **Main Pipeline Runner**: `src/data_quality/pipeline_runner.py` (Complete 6-step pipeline)
- **CSV Builder**: `src/csv_building/final_csv_builder.py` (Dataset generation)
- **NLP Preprocessor**: `src/nlp_preprocessing/text_preprocessor.py` (Text cleaning and normalization)
- **EDA Analysis**: `src/eda_analysis/eda_analysis_unusual_page_counts_notebook.ipynb` (Exploratory data analysis)
- **Comprehensive Analysis**: `src/data_quality/comprehensive_data_analysis.py` (Data analysis)
- **Comprehensive Cleaner**: `src/data_quality/comprehensive_data_cleaner.py` (Data cleaning)
- **Raw Data**: `data/raw/` (9 Goodreads JSON files)
- **Current Datasets**: `data/processed/` (9 current dataset files)
- **Organized Outputs**: `organized_outputs/` (All organized outputs and visualizations)
- **Archive**: `archive/` (All archived development history)

## Repository Maintenance

### Recent Cleanup Activities (September 2025)
The repository has undergone comprehensive cleanup and organization to maintain a clean, focused codebase:

**September 7, 2025 Comprehensive Cleanup**:
- **Repository Organization**: All outputs centralized in `organized_outputs/` directory
- **Duplicate Files**: Archived 50+ duplicate files now superseded by organized structure
- **Old Dataset Versions**: Archived 5 outdated dataset versions, keeping only current ones
- **Old Logs**: Archived 3 old log files superseded by organized logs
- **Old Reports**: Archived 10+ outdated reports superseded by organized reports
- **Data Quality Outputs**: Archived old pipeline outputs from `src/data_quality/outputs/`

**Previous Cleanup Activities (January 2025)**:
- **Python Cache Files**: Removed `__pycache__` directories from `src/data_quality/` and `src/csv_building/`
- **Empty Log Files**: Archived 5 empty log files (0 bytes) that were never used
- **Duplicate Outputs**: Archived earlier versions of quality reports and unprocessed batch files
- **Development Tools**: Archived 3 diagnostic scripts used during development
- **Alternative Runners**: Archived 4 redundant runner scripts that duplicated core functionality

**Archive Organization**:
- All development history preserved in `archive/` directory
- Clean separation between active code and development artifacts
- Easy recovery process for any archived files if needed
- Streamlined repository with only essential functionality

### Current Repository State
- **Clean Source Code**: No cache files or redundant scripts in active directories
- **Organized Outputs**: All outputs centralized in `organized_outputs/` with clear structure
- **Current Datasets**: Only 3 current dataset files in `data/processed/`
- **Preserved History**: All development artifacts safely archived
- **Ready for Research**: Clean, organized codebase prepared for advanced NLP analysis

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

### Complete Pipeline Results
- **Starting Records**: 119,678 romance novels (raw integrated data)
- **Final Records**: 80,755 romance novels (cleaned and preprocessed)
- **Total Reduction**: 38,923 records (32.52% reduction)
- **Data Quality**: High-quality, research-ready dataset
- **Text Processing**: HTML cleaned, normalized, and validated
- **Coverage**: 2000-2017 publication years (English editions)
- **Structure**: 30 columns (complete) or 23 columns (core research)
- **Quality Status**: Quality-certified through complete 6-step pipeline

### Dataset Versions Available
- **Complete Dataset**: All 30 columns with full metadata
- **Core Research Dataset**: 23 essential columns for efficient analysis
- **False Duplicates Dataset**: 21,105 records with similar titles by different authors
- **Step-by-Step Datasets**: 7 datasets showing pipeline progression
- **Specialized Versions**: 9 different dataset versions for specific research needs
  - Main cleaned datasets
  - Anthologies and collections
  - Short form content
  - Unexplainably short books
  - Sample datasets for testing

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

# Open src/eda_analysis/eda_analysis_unusual_page_counts_notebook.ipynb
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

### Running Comprehensive Analysis and Cleaning
```bash
# Run comprehensive data analysis
cd src/data_quality
python comprehensive_data_analysis.py

# Run comprehensive data cleaning
python comprehensive_data_cleaner.py
```

### Running NLP Text Preprocessing
```bash
# Run text preprocessing
cd src/nlp_preprocessing
python run_preprocessor.py

# Test preprocessing on sample data
python test_preprocessor.py
```

### Pipeline Outputs
All pipeline outputs are organized in `organized_outputs/`:
- **Datasets**: Step-by-step and specialized versions in `organized_outputs/datasets/`
- **Reports**: JSON and Markdown reports in `organized_outputs/reports/`
- **Logs**: Pipeline execution logs in `organized_outputs/logs/`
- **Visualizations**: Publication-ready plots in `organized_outputs/visualizations/`
- Final optimized datasets are available in multiple formats (CSV, Parquet, Pickle)

## Documentation

### Current Documentation
- **README.md**: This file - project overview and current status
- **organized_outputs/README.md**: Complete output documentation and organization
- **src/eda_analysis/README.md**: EDA analysis documentation
- **archive/ARCHIVE_SUMMARY.md**: Complete archive organization and cleanup history
- **archive/cleanup_20250106/CLEANUP_SUMMARY.md**: Latest cleanup details (January 2025)
- **organized_outputs/reports/**: Pipeline execution reports and quality assessments
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