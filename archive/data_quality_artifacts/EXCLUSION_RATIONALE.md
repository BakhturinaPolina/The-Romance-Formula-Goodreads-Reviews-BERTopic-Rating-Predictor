# Dataset Exclusion Rationale

## Overview

This document provides the research transparency rationale for excluding certain books from the Romance Novel NLP Research dataset. The exclusions were made to ensure data quality and suitability for NLP analysis while maintaining statistical power.

## Exclusion Criteria

### 1. Missing `num_pages_median` (30.0% of dataset)
**Rationale**: Page count is a critical variable for understanding book length and genre characteristics.

**Impact**: 
- **Excluded**: 35,908 books
- **Reason**: Missing page count data prevents analysis of book length patterns
- **Research Impact**: Cannot analyze relationships between book length and popularity/themes
- **Alternative**: Could be imputed from similar books, but introduces uncertainty

**Decision**: Exclude to maintain data integrity and avoid imputation bias.

### 2. Missing Descriptions (5.2% of dataset)
**Rationale**: Book descriptions are essential for NLP analysis and theme extraction.

**Impact**:
- **Excluded**: 6,172 books
- **Reason**: No text content available for topic modeling
- **Research Impact**: Critical for RQ1 (Topic Modeling) and RQ4 (Author vs. Reader Themes)
- **Alternative**: None - text content cannot be imputed

**Decision**: Exclude as these books cannot contribute to text-based analysis.

### 3. Very Short Descriptions (<50 characters) (0.2% of dataset)
**Rationale**: Extremely short descriptions provide insufficient content for meaningful NLP analysis.

**Impact**:
- **Excluded**: 186 books
- **Reason**: Insufficient text length for topic modeling
- **Research Impact**: Would introduce noise in theme extraction
- **Alternative**: Could be included but would reduce analysis quality

**Decision**: Exclude to maintain NLP analysis quality standards.

## Combined Exclusion Results

### Dataset Impact
- **Original dataset**: 119,678 books
- **Final dataset**: 80,747 books
- **Retention rate**: 67.5%
- **Exclusion rate**: 32.5%

### Statistical Power Assessment
- **Sample size**: 80,747 books is sufficient for:
  - Robust topic modeling (RQ1)
  - Reliable correlation analysis (RQ3)
  - Author comparison studies (RQ4)
  - Subgenre analysis across multiple categories

### Coverage Analysis
- **Genre coverage**: 100% (all remaining books have genre data)
- **Author coverage**: 100% (all remaining books have author data)
- **Series coverage**: 66.9% (maintained from original dataset)
- **Text content**: 100% (all remaining books have usable descriptions)

## Research Transparency

### Documentation
- **Exclusion criteria**: Clearly defined and documented
- **Impact assessment**: Quantified effect on dataset size
- **Statistical justification**: Demonstrated adequacy of remaining sample
- **Alternative approaches**: Considered and documented

### Reproducibility
- **Exclusion logic**: Implemented in reproducible code
- **Data lineage**: Clear trace from original to final dataset
- **Quality metrics**: Documented for each exclusion category
- **Decision rationale**: Transparent and justified

### Limitations
- **Potential bias**: Exclusions may introduce systematic bias
- **Generalizability**: Results may not apply to excluded books
- **Missing data patterns**: Need to analyze if exclusions follow patterns

## Quality Assurance

### Data Integrity
- **No missing critical fields**: All remaining books have complete data
- **Consistent data types**: All variables pass validation
- **Text quality**: Descriptions meet minimum length requirements
- **Metadata completeness**: Author, genre, and series data available

### Analysis Readiness
- **NLP compatibility**: All books suitable for text analysis
- **Statistical validity**: Sufficient sample size for robust analysis
- **Variable coverage**: Complete data for all analysis variables
- **Quality standards**: Meets research quality requirements

## Conclusion

The exclusion of 32.5% of the original dataset was necessary to ensure data quality and suitability for NLP analysis. The remaining 80,747 books provide:

1. **Sufficient statistical power** for robust analysis
2. **Complete data quality** for all critical variables
3. **NLP-ready text content** for theme extraction
4. **Transparent exclusion rationale** for research integrity

These exclusions maintain the scientific rigor of the research while ensuring the dataset meets the quality standards required for reliable NLP analysis and theme extraction.

---

**Document Version**: 1.0  
**Date**: September 2025  
**Author**: AI Assistant  
**Review Status**: Pending  
**Next Review**: After Step 2 completion
