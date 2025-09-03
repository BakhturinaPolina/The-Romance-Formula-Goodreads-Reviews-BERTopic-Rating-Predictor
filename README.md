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
│   ├── data_quality/                             # Data quality pipeline (Steps 4-6)
│   │   ├── outlier_detection_step4.py            # Step 4: Outlier Detection
│   │   ├── apply_outlier_treatment_step4.py      # Step 4: Outlier Treatment
│   │   ├── data_type_optimization_step5.py       # Step 5: Data Type Optimization
│   │   ├── final_quality_validation_step6.py     # Step 6: Final Quality Validation
│   │   ├── run_outlier_detection_step4.py        # Step 4 Runner
│   │   ├── README.md                             # Module documentation
│   │   └── README_STEP4_OUTLIER_DETECTION.md     # Step 4 details
│   └── csv_building/                             # Working CSV builder module
│       ├── final_csv_builder_working.py          # Active CSV builder (39KB)
│       └── run_working_builder.py                # Runner script (2.7KB)
├── data/
│   ├── raw/                                      # Original Goodreads JSON files
│   ├── intermediate/                              # Temporary processing outputs
│   └── processed/                                # Final cleaned datasets
│       └── README.md                             # Pipeline documentation
├── outputs/                                       # Current pipeline outputs
│   ├── final_quality_validation/                 # Step 6: Final quality validation
│   └── data_type_optimization/                   # Step 5: Data type optimization
├── logs/                                          # Execution logs
├── docs/                                          # Project documentation
├── notebooks/                                     # Jupyter notebooks
└── archive/                                       # Archived unused code and outputs
    ├── data_quality_archive_20250902/            # Archived data quality files
    ├── pipeline_outputs_20250902/                # Archived pipeline outputs
    └── unused_code/                              # Legacy and experimental code
```

### Key Files
- **Main Dataset**: `data/processed/cleaned_romance_novels_step*.pkl/csv`
- **Quality Pipeline**: `src/data_quality/` (Steps 4-6 implementation)
- **Working Builder**: `src/csv_building/final_csv_builder_working.py`
- **Runner Script**: `src/csv_building/run_working_builder.py`

## Data Quality Pipeline

### 6-Step Pipeline Architecture
The project implements a comprehensive data quality pipeline:

1. **Steps 1-3**: Initial Cleaning (Missing Values, Duplicates, Data Types)
   - Implemented elsewhere in the project
   - Outputs archived in `archive/pipeline_outputs_20250902/`

2. **Step 4**: Outlier Detection & Treatment
   - **Detection**: `OutlierDetectionReporter` class
   - **Treatment**: `OutlierTreatmentApplier` class
   - **Outputs**: Statistical reports and treated datasets

3. **Step 5**: Data Type Optimization & Persistence
   - **Optimization**: `DataTypeOptimizer` class
   - **Outputs**: Optimized datasets (parquet/pickle) and reports

4. **Step 6**: Final Quality Validation & Certification
   - **Validation**: `FinalQualityValidator` class
   - **Outputs**: Quality scores, validation reports, certification

### Pipeline Benefits
- **Data Integrity**: Comprehensive validation and quality assurance
- **Memory Efficiency**: Optimized data types for large datasets
- **Reproducibility**: Documented pipeline with clear outputs
- **Quality Certification**: Final datasets meet quality standards

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
# Run the working CSV builder
cd src/csv_building
python run_working_builder.py

# Choose processing mode:
# 1. Test with sample (recommended for first run)
# 2. Process full dataset
```

### Running Data Quality Pipeline
```bash
# Step 4: Outlier Detection
cd src/data_quality
python run_outlier_detection_step4.py

# Step 5: Data Type Optimization
python data_type_optimization_step5.py

# Step 6: Final Quality Validation
python final_quality_validation_step6.py
```

## Documentation

### Current Documentation
- **README.md**: This file - project overview and current status
- **.cursor/rules/folder-structure.mdc**: Detailed folder structure and architecture
- **src/data_quality/README.md**: Data quality module documentation
- **src/csv_building/README.md**: CSV building module documentation
- **notebooks/02_final_dataset_eda_and_cleaning.ipynb**: Updated EDA notebook

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

**Last Updated**: September 2025  
**Status**: Phase 3 Complete, Phase 4 Ready  
**Code Status**: Clean, focused codebase with comprehensive data quality pipeline  
**Next Milestone**: Topic modeling implementation and theme extraction