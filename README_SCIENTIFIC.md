
## Introduction

### Context & Motivation

Romance is one of the largest and most lucrative fiction categories, with dense review activity on platforms like Goodreads. Existing theory (Radway, Ogas & Gaddam, Weinstein, Goralik) describes what should matter in romance—HEA (Happily Ever After), emotional intimacy, hero type, luxury, etc.—but large-scale empirical evidence is lacking on how readers themselves talk about these elements in reviews and how this discourse relates to ratings.

### Shift in Focus

Instead of analyzing novel texts, this study focuses purely on **reader discourse**: what readers praise, criticize, and discuss in reviews. Reviews are treated as a rich reception corpus that encodes readers' norms and expectations, offering direct insight into what readers value and penalize in romance novels.

### Aim & Thesis

This study bridges **computational text analysis of Goodreads reviews** with **romance genre theory** to empirically identify which reader-discussed themes distinguish highly-rated from poorly-rated books. Using BERTopic and semi-supervised labeling on Goodreads reviews, this study tests theory-driven claims about romance: it asks which reader-discussed themes (e.g., HEA satisfaction, love vs. 'smut-only', hero toxicity vs. protectiveness, luxury fantasy, triangles, heroine agency, escapist binge-reading) are associated with higher or lower book ratings.

### Research Approach

The study is twofold: (1) **Theme Discovery (Exploratory)**: Use BERTopic to discover latent themes and reader discourse patterns in romance reviews; (2) **Hypothesis Testing (Confirmatory)**: Test theory-driven claims by constructing per-book thematic indicators from discovered topics and assessing their association with average book ratings.

## Theoretical Framework

### 2.1 Psychological Appeal of Romance Reading

Romance reading serves compensatory and escapist functions (Radway, 1984). Ogas & Gaddam (2011) argue that women prefer narrative, relationship-centric erotic content over purely visual explicitness. These motives should surface explicitly in reviews as talk about "escape," "comfort read," "couldn't put it down," "emotional," "heartwarming," etc.

### 2.2 Formulaic Narratives and Genre Conventions

Romance follows a formulaic structure (Proppian functions, Radway's thirteen-step arc) with HEA as a non-negotiable convention. In reviews, these conventions appear as:

- **Praise**: "perfect HEA," "so satisfying," "loved the epilogue," "slow burn but worth it"
- **Complaints**: "no HEA," "cliffhanger," "non-ending," "too much drama, not enough payoff"

### 2.3 Key Elements of a "Good" Romance, Seen Through Reviews

**Hero archetype**: Strong but gentle, powerful but ultimately caring/protective. Reviews distinguish "cinnamon roll," "protective," "alpha but sweet" vs. "toxic," "red flags," "abusive," "controlling."

**Heroine archetype**: Relatable yet competent; not a doormat. Reviews praise "strong heroine," "spunky," "independent" vs. criticize "TSTL" (Too Stupid To Live), "doormat," "no backbone."

**Luxury & comfort fantasy**: Billionaire/royal tropes, lavish settings. Reviews discuss "billionaire fantasy," "rich AF," "over the top but fun" as escapist enhancement.

**Love vs. sex**: Emotional payoff vs. "just smut." Reviews frequently distinguish "steamy but with plot and feels" vs. "smut-only, no story."

### 2.4 Theory to Be Tested (in Reviews)

The main theoretical claims, as they should appear in review language:

- Readers value emotional intimacy and commitment signals more than sheer explicitness
- The hero's "dark" traits must be balanced by tenderness; reviewers will punish heroes labeled as "toxic"
- Luxury/wealth content functions as comfort fantasy but only works when paired with convincing love
- HEA is essential: unresolved or tragic endings should trigger strong negative review reactions
- Readers explicitly talk about triangles, heroine agency, and binge/escape when these are salient

## Research Objectives

### Objective 1 – Theme Discovery in Reviews
Develop a theory-driven thematic framework for romance as it appears in Goodreads reviews (commitment/HEA, tenderness vs. explicitness, hero type, triangles, agency, luxury, escapism), and map BERTopic topics to this framework.

### Objective 2 – Theme–Rating Relationships
Examine how the frequency and sentiment of these review themes vary between highly rated and poorly rated books (e.g., 4–5★ vs. 1–2★) and whether they predict average ratings.

### Objective 3 – Index Construction from Reviews
Build composite indicators from review discourse (e.g., Love-over-Sex in reviews; Hero Toxicity vs. Protectiveness; HEA/Ending Satisfaction; Escapism/Binge) and assess their association with ratings.

### Objective 4 – Reader Norms and Values
Use these patterns to infer reader norms: which tropes and outcomes are explicitly praised or punished in reviews, thereby offering an empirical, reader-centered portrait of "what makes a good romance" today.

## Research Questions

### Theory-Driven Questions

**RQ1 – Thematic Profile of Reader Discourse**: What major romance-related themes do readers emphasize in Goodreads reviews (HEA/ending, love vs. "smut," hero behavior, heroine agency, triangles, luxury, escapism, etc.)?

**RQ2 – Valence of Themes**: How are these themes evaluated in reviews? Which are predominantly praised (e.g., satisfying HEA, protective hero, strong heroine, escapist fun), and which are predominantly criticized (e.g., toxic hero, love triangle, "smut-only," unsatisfying ending)?

**RQ3 – Themes and Ratings**: How does the prominence of each theme in reviews relate to a book's average rating? Which review-discussed themes are reliably associated with higher versus lower ratings?

**RQ4 – Love vs. "Smut-only"**: Do books whose reviews stress emotional intimacy, chemistry, and commitment fare better than books whose reviews stress explicitness without emotional depth?

**RQ5 – Hero & Heroine Reception**: How do review-discussed portrayals of the hero (protective vs. toxic) and heroine (agency vs. doormat) correlate with ratings?

**RQ6 – Escapism, Luxury, and Satisfaction**: To what extent do escapist/binge-reading language and luxury/fantasy content in reviews predict higher ratings?

### Exploratory Questions

**RQ7 – Unexpected Themes**: What additional themes emerge from reader reviews that are not predicted by existing romance theory? How do these themes relate to ratings?

**RQ8 – Theme Interactions**: Are there interactions between themes (e.g., luxury theme with HEA praise) that predict ratings beyond individual theme effects?

**RQ9 – Review Patterns**: What patterns exist in reader engagement across different book characteristics (publication years, subgenres, series vs. standalone)?

## Hypotheses

All hypotheses are formulated in terms of review text, not novel text.

**H1 (HEA Praise → Higher Ratings)**: Books whose reviews frequently praise a satisfying HEA or ending ("perfect ending," "great HEA," "so satisfying") will have higher ratings; books whose reviews complain about endings ("cliffhanger," "no HEA," "rushed ending," "no closure") will have lower ratings.

**H2 (Love-over-Smut Hypothesis)**: Reviews emphasizing emotional connection, chemistry, and commitment ("slow burn," "emotional," "beautiful love story") predict higher ratings; reviews emphasizing "smut-only" or "all sex, no plot/feels" predict lower ratings.

**H3 (Protective vs. Toxic Hero Hypothesis)**: Books whose reviews describe the hero with positive relational terms ("protective," "sweet," "respectful," "cinnamon roll," "alpha but caring") have higher ratings; books whose reviews label the hero as "toxic," "abusive," "red flags," "controlling" have lower ratings.

**H4 (Triangle Complaint Hypothesis)**: Frequent mentions of love triangles or cheating in a critical tone ("hate triangles," "cheating trope," "too much drama") are associated with lower ratings.

**H5 (Heroine Agency Hypothesis)**: Books whose reviews praise the heroine as "strong," "independent," "not a doormat" have higher ratings than those where reviewers complain about a "TSTL" or "doormat" heroine.

**H6 (Escapism/Binge Hypothesis)**: Reviews that describe the book as an escape or binge ("couldn't put it down," "I devoured this," "perfect comfort read") are associated with higher ratings.

**H7 (Luxury as Enjoyed Fantasy Hypothesis)**: Books where reviews discuss luxury/fantasy elements (billionaire, rich lifestyle, glamorous settings) in a positive/gleeful way tend to have higher ratings; where such elements are criticized as shallow ("just money, no feelings"), ratings are not improved or may be lower.

**H8 (Justice/Punishment Hypothesis)**: Books whose reviews complain that villains or cheaters are not punished ("no consequences," "he got away with it") have lower ratings; books whose reviews praise satisfying justice ("got what they deserved," "karma") have higher ratings.

## Dataset

### Data Sources
The project uses Goodreads metadata from the UCSD Goodreads Book Graph:

- **Book Metadata**: Titles, authors, publication years, ratings, genres, series information
- **Reader Reviews**: Text reviews with ratings (~3.6M reviews, 1.2GB compressed)
- **User Interactions**: User-book interactions and reading patterns
- **Work-Level Aggregation**: Multiple editions aggregated to work-level for fair comparison

### Data Processing Approach

The dataset uses **work-level aggregation** to handle multiple editions of the same book:

- **Individual Edition Fields**: Removed from final dataset
  - `average_rating` (individual edition rating)
  - `ratings_count` (individual edition ratings count)
  - `text_reviews_count` (individual edition reviews count)

- **Work-Level Aggregated Fields**: Used for all analysis
  - `average_rating_weighted_mean` (weighted average across all editions)
  - `ratings_count_sum` (total ratings across all editions)
  - `text_reviews_count_sum` (total reviews across all editions)

This approach ensures fair comparison between books regardless of how many editions they have.

### Sample Characteristics
- **Main Dataset**: Processed book metadata with comprehensive quality assurance
- **Review Dataset**: English-language reviews extracted and filtered
- **Subdataset**: Representative 6,000-book sample for topic modeling analysis

## Methodology and Methods

### Research Pipeline

The research follows an 8-stage pipeline:

1. **Data Integration**: CSV building, data integration, representative sampling
2. **Data Quality**: 6-step quality assurance pipeline (missing values, duplicates, data types, outliers, optimization, validation)
3. **Text Preprocessing**: HTML cleaning, text normalization, genre categorization
4. **Review Extraction**: Language detection, filtering, and extraction of English reviews
5. **Prepare Reviews Corpus for BERTopic**: Sentence splitting and corpus creation from reviews
6. **Topic Modeling**: BERTopic analysis with hyperparameter optimization using OCTIS framework
7. **Shelf Normalization**: Normalization of user-generated shelf tags
8. **Corpus Analysis**: Statistical analysis of corpus characteristics

### Topic Modeling Methodology

- **Model**: BERTopic with various embedding models
- **Framework**: OCTIS (Optimization and Comparative Topic Modeling Infrastructure for Scholars)
- **Hyperparameter Optimization**: Bayesian optimization using OCTIS framework
- **Evaluation Metrics**: Topic coherence, diversity, and quality metrics
- **Sentence-Level Analysis**: Reviews split into sentences for granular topic extraction

### Semi-Supervised Topic Labeling

After BERTopic discovers latent themes, semi-supervised labeling is applied to map topics to the theoretical framework:

- **Domain Lexicons**: Use romance-specific vocabulary (e.g., "HEA," "toxic hero," "smut-only," "cinnamon roll") to identify relevant topics
- **Manual Inspection**: Review top words and exemplar reviews for each topic to interpret them in terms of romance theory (HEA, hero toxicity, triangles, etc.)
- **Weak Supervision**: Optionally refine with label functions and/or few-shot classifiers to group topics into broader thematic categories (e.g., commitment/HEA, explicitness, hero_good, hero_toxic, triangle, agency, escapism, luxury, justice)
- **Thematic Mapping**: Assign interpretable labels to topics based on their alignment with theoretical constructs

### Per-Book Theme Indicators

For each book, topic assignments are aggregated to compute theme indicators:

- **Proportion-Based Metrics**: Compute for each book the proportion of its reviews that belong to each theme
- **Composite Indices**: Derive simple indices such as:
  - Share of "HEA praise" vs. "ending complaint" reviews
  - Share of "smut-only" complaint reviews
  - Share of "protective hero" vs. "toxic hero" reviews
  - Share of "strong heroine" vs. "doormat" reviews
  - Share of "escapism/binge" reviews
  - Share of "luxury fantasy" reviews
  - Share of "justice/punishment" reviews

### Statistical Analysis

- **Descriptive Analysis**: Compare theme distributions between high- and low-rated books (4–5★ vs. 1–2★)
- **Regression Models**: Regress average book rating (and/or probability of 4–5★) on review theme indicators, controlling for number of ratings, year, etc.
- **Hypothesis Testing**: Test each hypothesis (H1–H8) by inspecting sign and significance of the relevant review-themed predictors
- **Heavy-Tail Analysis**: Clauset-Shalizi-Newman (2009) power-law fitting methodology
- **Overdispersion Testing**: Dean-Lawless and Cameron-Trivedi formal statistical tests
- **Quality Assurance**: Comprehensive validation with automated quality gates
- **Representative Sampling**: Stratified sampling preserving key demographic characteristics

## Tools

### Programming Languages and Frameworks
- **Python 3.8+**: Primary programming language
- **pandas**: Data manipulation and analysis
- **spaCy**: Natural language processing and sentence splitting
- **BERTopic**: Topic modeling framework
- **OCTIS**: Topic modeling optimization framework

### Data Processing Tools
- **pandas**: Data manipulation
- **pyarrow**: Parquet file handling
- **langdetect**: Language detection for review filtering

### Statistical Analysis
- **scipy**: Statistical tests and distributions
- **numpy**: Numerical computations
- **Custom implementations**: Power-law fitting, overdispersion tests

## Results

### Data Quality
- Comprehensive 6-step quality assurance pipeline implemented
- Missing values, duplicates, and outliers identified and treated
- Data type optimization for memory efficiency
- Final quality validation and certification completed

### Review Processing
- ~3.6M reviews processed from Goodreads dataset
- English-language reviews extracted and filtered
- Sentence-level dataset created for topic modeling (~8.7M sentences)
- Processing rate: ~100-110 reviews/second

### Topic Modeling
- BERTopic models trained with multiple embedding configurations
- Hyperparameter optimization completed
- Topic extraction and analysis in progress
- Results stored in structured format for further analysis

### Corpus Characteristics
- Representative 6,000-book subdataset created
- Balanced across popularity tiers (top, mid, thrash)
- Key demographic characteristics preserved
- Ready for statistical analysis and topic modeling

## Current Status

### Completed
- Data integration and quality assurance pipeline
- Text preprocessing and normalization
- Review extraction and filtering
- Sentence-level dataset preparation
- Topic modeling infrastructure setup
- Hyperparameter optimization framework

### In Progress
- Topic modeling analysis and interpretation
- Statistical analysis of topic distributions
- Correlation analysis between topics and metadata

## Outputs

All research outputs are organized in the `outputs/` directory:
- **Datasets**: Processed datasets at various pipeline stages
- **Reports**: Analysis reports in JSON and Markdown formats
- **Visualizations**: Publication-ready plots and charts
- **Logs**: Execution logs for reproducibility

## Reproducibility

The pipeline is designed for reproducibility:
- All code organized by research stage
- Configuration files for key parameters
- Comprehensive logging of all processing steps
- Documentation for each stage of the pipeline

See `docs/replication_guide.md` for instructions on adapting the pipeline to other datasets.

