# Stage 05: Topic Modeling - Scientific Documentation

## Research Objectives

- Extract topics from reader reviews using BERTopic topic modeling
- Optimize topic modeling hyperparameters using OCTIS framework
- Evaluate topic quality and coherence
- Compare different embedding models for topic extraction

## Research Questions

1. What topics emerge from reader reviews?
2. What hyperparameters optimize topic modeling performance?
3. How do different embedding models affect topic extraction?
4. What is the optimal number of topics for the review corpus?

## Hypotheses

- **H1**: BERTopic effectively extracts meaningful topics from reviews
- **H2**: Hyperparameter optimization improves topic quality
- **H3**: Different embedding models produce different topic structures
- **H4**: Sentence-level analysis provides more granular topic extraction than review-level

## Dataset

- **Input**: Sentence-level review data (parquet format)
  - Prepared by `05_prepare_reviews_corpus_for_BERTopic/`
  - File: `data/processed/review_sentences_for_bertopic.parquet`
- **Output**: Topic models with extracted topics and document-topic assignments
- **Format**: OCTIS-compatible format for optimization

## Methodology

- **BERTopic**: Topic modeling using BERT-based embeddings
- **OCTIS Integration**: Optimization framework for hyperparameter tuning
- **Hyperparameter Optimization**: Bayesian optimization using OCTIS framework
- **Sentence-Level Analysis**: Topics extracted from individual sentences
- **Multiple Embeddings**: Testing various embedding models (sentence-transformers)

## Tools

- **BERTopic**: Topic modeling framework
- **OCTIS**: Topic modeling optimization framework
- **sentence-transformers**: Embedding models
- **spacy**: Sentence splitting (used in preparation stage)

## Statistical Tools

- Topic coherence metrics
- Topic diversity measures
- Hyperparameter search space optimization
- Model evaluation metrics

## Results

- Topic models with extracted topics
- Hyperparameter optimization results
- Topic quality evaluations
- Model comparison results

## Related Work

- **Data Preparation**: See `05_prepare_reviews_corpus_for_BERTopic/README_SCIENTIFIC.md`
  - Sentence segmentation methodology
  - Data coverage analysis
  - Corpus creation process
