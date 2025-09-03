# NLP Analysis Module

## Overview

The NLP Analysis module is designed to provide comprehensive natural language processing capabilities for the Romance Novel NLP Research project. This module will implement advanced text analysis, topic modeling, and correlation analysis to extract insights from the processed romance novel data.

**Status**: Placeholder module (currently empty `__init__.py`)

## Planned Components

### 🧠 Topic Modeling (`topic_modeling.py`)

**Purpose**: Implement BERTopic and other topic modeling techniques for analyzing review text and book descriptions.

**Planned Features**:
- **BERTopic Implementation**: Advanced topic modeling using BERT embeddings
- **Topic Visualization**: Interactive topic visualization and exploration
- **Topic Interpretation**: Automatic topic labeling and interpretation
- **Topic Evolution**: Analysis of topic trends over time
- **Cross-Dataset Topics**: Topic modeling across books and reviews

**Planned Usage**:
```python
from src.nlp_analysis.topic_modeling import BERTopicAnalyzer

# Initialize topic analyzer
analyzer = BERTopicAnalyzer(
    model_name="all-MiniLM-L6-v2",
    n_topics=20,
    min_topic_size=10
)

# Analyze review topics
review_topics = analyzer.analyze_reviews(reviews_data)

# Analyze book description topics
description_topics = analyzer.analyze_descriptions(books_data)

# Generate topic visualizations
analyzer.create_topic_visualizations()
```

**Expected Output**:
- Topic model artifacts and visualizations
- Topic-word distributions and coherence scores
- Topic-document assignments
- Topic evolution analysis
- Interactive topic exploration tools

### 📊 Correlation Analysis (`correlation_analysis.py`)

**Purpose**: Analyze correlations between text features, ratings, and metadata.

**Planned Features**:
- **Text-Rating Correlations**: Analyze relationships between review text and ratings
- **Feature Correlations**: Correlate text features with book metadata
- **Sentiment Analysis**: Sentiment correlation with ratings and popularity
- **Genre Correlations**: Text feature correlations across subgenres
- **Temporal Correlations**: Text feature evolution over time

**Planned Usage**:
```python
from src.nlp_analysis.correlation_analysis import CorrelationAnalyzer

# Initialize correlation analyzer
analyzer = CorrelationAnalyzer()

# Analyze text-rating correlations
text_rating_corr = analyzer.analyze_text_rating_correlations(
    reviews_data, books_data
)

# Analyze sentiment correlations
sentiment_corr = analyzer.analyze_sentiment_correlations(
    reviews_data, books_data
)

# Generate correlation visualizations
analyzer.create_correlation_plots()
```

**Expected Output**:
- Correlation matrices and heatmaps
- Statistical significance tests
- Feature importance rankings
- Correlation trend analysis
- Interactive correlation exploration

### 📝 Text Preprocessing (`text_preprocessing.py`)

**Purpose**: Comprehensive text preprocessing and feature extraction.

**Planned Features**:
- **Text Cleaning**: Remove noise, normalize text, handle special characters
- **Tokenization**: Advanced tokenization with custom romance-specific rules
- **Feature Extraction**: Extract linguistic features, readability scores
- **Text Normalization**: Standardize text formats and structures
- **Quality Assessment**: Assess text quality and completeness

**Planned Usage**:
```python
from src.nlp_analysis.text_preprocessing import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    lemmatize=True,
    remove_punctuation=True
)

# Preprocess review text
processed_reviews = preprocessor.preprocess_reviews(reviews_data)

# Extract text features
text_features = preprocessor.extract_features(processed_reviews)

# Assess text quality
quality_scores = preprocessor.assess_quality(processed_reviews)
```

**Expected Output**:
- Cleaned and normalized text data
- Extracted text features and metrics
- Text quality assessment scores
- Preprocessing pipeline artifacts
- Feature importance rankings

### 🎭 Sentiment Analysis (`sentiment_analysis.py`)

**Purpose**: Advanced sentiment analysis for romance novel reviews.

**Planned Features**:
- **Multi-Aspect Sentiment**: Analyze sentiment across different aspects
- **Romance-Specific Sentiment**: Custom sentiment models for romance content
- **Emotional Analysis**: Detect emotions beyond positive/negative
- **Sentiment Evolution**: Track sentiment changes over time
- **Cross-Cultural Sentiment**: Analyze sentiment across different cultures

**Planned Usage**:
```python
from src.nlp_analysis.sentiment_analysis import SentimentAnalyzer

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer(
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
    romance_specific=True
)

# Analyze review sentiment
sentiment_scores = analyzer.analyze_sentiment(reviews_data)

# Analyze emotional content
emotions = analyzer.analyze_emotions(reviews_data)

# Generate sentiment visualizations
analyzer.create_sentiment_visualizations()
```

**Expected Output**:
- Sentiment scores and classifications
- Emotional content analysis
- Sentiment trend analysis
- Sentiment correlation with ratings
- Interactive sentiment exploration

### 📈 Text Statistics (`text_statistics.py`)

**Purpose**: Comprehensive text statistics and linguistic analysis.

**Planned Features**:
- **Readability Metrics**: Calculate various readability scores
- **Linguistic Features**: Extract linguistic characteristics
- **Vocabulary Analysis**: Analyze vocabulary diversity and complexity
- **Stylometric Features**: Author and genre stylometric analysis
- **Text Complexity**: Assess text complexity and accessibility

**Planned Usage**:
```python
from src.nlp_analysis.text_statistics import TextStatistician

# Initialize text statistician
statistician = TextStatistician()

# Calculate readability metrics
readability_scores = statistician.calculate_readability(books_data)

# Analyze vocabulary
vocabulary_stats = statistician.analyze_vocabulary(reviews_data)

# Extract linguistic features
linguistic_features = statistician.extract_linguistic_features(
    reviews_data, books_data
)

# Generate statistical visualizations
statistician.create_statistical_plots()
```

**Expected Output**:
- Readability scores and metrics
- Vocabulary diversity statistics
- Linguistic feature matrices
- Stylometric analysis results
- Text complexity assessments

## Dependencies

### Planned External Dependencies
- **BERTopic**: Advanced topic modeling
- **Transformers**: Pre-trained language models
- **SentenceTransformers**: Sentence embeddings
- **NLTK**: Natural language processing toolkit
- **spaCy**: Advanced NLP processing
- **TextBlob**: Simple sentiment analysis
- **VADER**: Valence Aware Dictionary and sEntiment Reasoner
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static visualizations
- **Scikit-learn**: Machine learning utilities

### Internal Dependencies
- `src/utils/file_handlers.py`: File processing utilities
- `src/utils/lightweight_handlers.py`: Data processing utilities
- `config/settings.py`: Configuration and settings
- `config/logging_config.py`: Logging configuration

## Data Requirements

### Input Data
- **Processed Reviews**: Clean review text with metadata
- **Processed Books**: Book descriptions and metadata
- **Author Data**: Author information for stylometric analysis
- **Genre Data**: Subgenre classifications for cross-genre analysis

### Data Format
```python
# Expected review data format
reviews_data = {
    'review_id': str,
    'book_id': str,
    'user_id': str,
    'rating': int,
    'review_text': str,
    'review_date': datetime,
    'helpful_votes': int
}

# Expected book data format
books_data = {
    'book_id': str,
    'title': str,
    'author_id': str,
    'description': str,
    'publication_year': int,
    'average_rating': float,
    'subgenre': str
}
```

## Output Structure

### Analysis Results
```
outputs/
├── nlp_analysis/
│   ├── topic_models/
│   │   ├── bertopic_model.pkl
│   │   ├── topic_visualizations/
│   │   └── topic_reports/
│   ├── correlation_analysis/
│   │   ├── correlation_matrices/
│   │   ├── correlation_plots/
│   │   └── correlation_reports/
│   ├── sentiment_analysis/
│   │   ├── sentiment_scores.csv
│   │   ├── emotion_analysis.csv
│   │   └── sentiment_visualizations/
│   ├── text_statistics/
│   │   ├── readability_scores.csv
│   │   ├── vocabulary_stats.csv
│   │   └── linguistic_features.csv
│   └── reports/
│       ├── nlp_analysis_summary.html
│       ├── topic_modeling_report.html
│       └── correlation_analysis_report.html
```

### Visualization Outputs
- **Interactive Dashboards**: Web-based exploration tools
- **Static Plots**: Publication-ready visualizations
- **Topic Visualizations**: Interactive topic exploration
- **Correlation Heatmaps**: Feature correlation displays
- **Sentiment Plots**: Sentiment distribution and trend plots

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. **Text Preprocessing**: Implement comprehensive text preprocessing
2. **Basic Statistics**: Calculate fundamental text statistics
3. **Simple Sentiment**: Implement basic sentiment analysis
4. **Data Validation**: Ensure data quality for NLP analysis

### Phase 2: Topic Modeling (Weeks 3-4)
1. **BERTopic Implementation**: Set up BERTopic pipeline
2. **Topic Visualization**: Create interactive topic visualizations
3. **Topic Interpretation**: Implement automatic topic labeling
4. **Topic Analysis**: Analyze topic distributions and trends

### Phase 3: Advanced Analysis (Weeks 5-6)
1. **Correlation Analysis**: Implement comprehensive correlation analysis
2. **Advanced Sentiment**: Multi-aspect and emotion analysis
3. **Stylometric Analysis**: Author and genre stylometric features
4. **Cross-Dataset Analysis**: Integrate analysis across datasets

### Phase 4: Integration and Reporting (Weeks 7-8)
1. **Interactive Dashboards**: Create web-based exploration tools
2. **Automated Reporting**: Generate comprehensive analysis reports
3. **Performance Optimization**: Optimize for large datasets
4. **Documentation**: Complete documentation and usage examples

## Performance Considerations

### Scalability
- **Large Dataset Handling**: Process datasets with 100,000+ reviews
- **Memory Efficiency**: Optimize memory usage for large text corpora
- **Parallel Processing**: Implement parallel processing where possible
- **Caching**: Cache intermediate results for repeated analysis

### Optimization Strategies
- **Batch Processing**: Process data in manageable batches
- **Model Caching**: Cache pre-trained models and embeddings
- **Incremental Analysis**: Support incremental analysis for new data
- **Resource Monitoring**: Monitor memory and CPU usage

## Testing Strategy

### Test Coverage
- **Unit Tests**: Individual function and method tests
- **Integration Tests**: End-to-end NLP pipeline tests
- **Performance Tests**: Large dataset performance benchmarks
- **Quality Tests**: Output quality and accuracy validation

### Test Files
- `tests/nlp_analysis/test_topic_modeling.py`
- `tests/nlp_analysis/test_correlation_analysis.py`
- `tests/nlp_analysis/test_text_preprocessing.py`
- `tests/nlp_analysis/test_sentiment_analysis.py`
- `tests/nlp_analysis/test_text_statistics.py`

## Future Enhancements

### Advanced Features
1. **Multi-Language Support**: Analysis in multiple languages
2. **Deep Learning Models**: Custom neural network architectures
3. **Real-time Analysis**: Real-time text analysis capabilities
4. **API Integration**: REST API for NLP analysis services
5. **Cloud Deployment**: Cloud-based NLP analysis platform

### Research Extensions
1. **Novel Topic Models**: Custom topic modeling approaches
2. **Cross-Modal Analysis**: Integration with cover images and metadata
3. **Temporal Analysis**: Advanced time-series analysis of text trends
4. **Comparative Analysis**: Cross-genre and cross-author comparisons
5. **Predictive Modeling**: Predict ratings and popularity from text

## Contributing Guidelines

### Development Standards
1. **Code Quality**: Follow established coding standards and practices
2. **Documentation**: Comprehensive docstrings and usage examples
3. **Testing**: Maintain high test coverage for all components
4. **Performance**: Optimize for large dataset processing
5. **Modularity**: Design for reusability and maintainability

### Research Integration
1. **Academic Standards**: Follow academic research methodology
2. **Reproducibility**: Ensure all analyses are reproducible
3. **Validation**: Validate results against established benchmarks
4. **Reporting**: Generate publication-ready reports and visualizations
5. **Open Science**: Share code, data, and results openly
