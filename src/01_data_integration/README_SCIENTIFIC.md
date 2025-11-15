# Stage 01: Data Integration - Scientific Documentation

## Research Objectives

- Integrate multiple Goodreads data sources into unified datasets
- Create representative samples for downstream analysis
- Handle work-level aggregation for multiple book editions

## Research Questions

1. How can multiple Goodreads data sources be effectively integrated?
2. What sampling strategy ensures representative datasets for analysis?
3. How should multiple editions of the same work be handled?

## Hypotheses

- **H1**: Work-level aggregation provides more accurate popularity metrics than edition-level data
- **H2**: Representative sampling preserves key demographic characteristics while enabling efficient analysis
- **H3**: Language filtering improves data quality for English-language analysis

## Dataset

- **Input**: Raw Goodreads JSON files (books, authors, series, genres, works, reviews)
- **Output**: Integrated CSV datasets with work-level aggregation
- **Sample**: Representative subdatasets (e.g., 6,000 books) with tier balancing

## Methodology

- **Work-Level Aggregation**: Groups multiple editions by work_id, aggregates metrics
- **Language Filtering**: Filters to English editions using regex patterns
- **Representative Sampling**: Stratified sampling across popularity tiers, preserving demographics
- **External Data Integration**: Extracts and integrates data from external sources

## Tools

- pandas: Data manipulation and integration
- json/gzip: Reading compressed JSON files
- datasets: External data source integration (Hugging Face)

## Statistical Tools

- Quantile-based tier definition
- Stratified sampling with demographic preservation
- Engagement-based prioritization metrics

## Results

- Integrated datasets with work-level aggregation
- Representative subdatasets created
- Language filtering applied
- External data sources integrated (if used)

