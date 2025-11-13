"""
Generate comprehensive statistical analysis for corpus construction article section.

This script performs:
1. Full dataset EDA on romance_books_main_final.csv
2. Cross-corpus validation comparing full dataset vs 6,000-book subset
3. Statistical tests with appropriate methods
4. Publication-ready visualizations
5. Markdown and CSV reports
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.stats import chi2_contingency, mannwhitneyu, chisquare
from scipy.stats import normaltest
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "corpus_construction"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create output directories
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Antique color palette (from user preferences)
ANTIQUE_COLORS = [
    '#855C75FF', '#D9AF6BFF', '#AF6458FF', '#736F4CFF', 
    '#526A83FF', '#625377FF', '#68855CFF', '#9C9C5EFF', 
    '#A06177FF', '#8C785DFF', '#467378FF', '#7C7C7CFF'
]

# Set plotting style
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette(ANTIQUE_COLORS)


def canonicalize_genre(genre_str):
    """Enhanced canonicalization with historical romance merging (reused from create_subdataset_6000.py)"""
    if not isinstance(genre_str, str) or not genre_str.strip():
        return ""

    # Split by comma and process each genre
    genres = [g.strip().lower() for g in genre_str.split(',') if g.strip()]

    # Enhanced canonicalization with historical romance merging
    canonical_genres = []
    for genre in genres:
        # Normalize common variations
        if genre in ['sci fi', 'sci-fi', 'science fiction']:
            canonical_genres.append('science fiction')
        elif genre in ['ya', 'young adult']:
            canonical_genres.append('young adult')
        elif genre in ['historical fiction', 'historical', 'history']:
            canonical_genres.append('historical romance')
        elif genre in ['paranormal romance', 'paranormal']:
            canonical_genres.append('paranormal')
        elif genre in ['contemporary romance', 'contemporary']:
            canonical_genres.append('contemporary')
        elif genre in ['fantasy romance', 'fantasy']:
            canonical_genres.append('fantasy')
        elif genre in ['mystery romance', 'mystery']:
            canonical_genres.append('mystery')
        else:
            canonical_genres.append(genre)

    return ', '.join(sorted(set(canonical_genres)))


def compute_pop_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Compute popularity tiers based on quantiles of average_rating_weighted_mean."""
    q1 = df['average_rating_weighted_mean'].quantile(0.25)
    q3 = df['average_rating_weighted_mean'].quantile(0.75)
    
    def tierize(x, lo=q1, hi=q3):
        if pd.isna(x):
            return None
        if x < lo:
            return "thrash"
        if x > hi:
            return "top"
        return "mid"
    
    df = df.copy()
    df['pop_tier'] = df['average_rating_weighted_mean'].apply(tierize)
    return df


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare both datasets with consistent preprocessing."""
    logger.info("Loading datasets...")
    
    # Load full dataset
    full_path = DATA_PROCESSED / "romance_books_main_final.csv"
    if not full_path.exists():
        raise FileNotFoundError(f"Full dataset not found: {full_path}")
    
    logger.info(f"Loading full dataset from {full_path}")
    full_df = pd.read_csv(full_path)
    logger.info(f"Loaded {len(full_df):,} books from full dataset")
    
    # Load subset
    subset_path = DATA_PROCESSED / "romance_subdataset_6000.csv"
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset not found: {subset_path}")
    
    logger.info(f"Loading subset from {subset_path}")
    subset_df = pd.read_csv(subset_path)
    logger.info(f"Loaded {len(subset_df):,} books from subset")
    
    # Apply genre canonicalization to both
    logger.info("Applying genre canonicalization...")
    full_df['genres_str'] = full_df['genres_str'].apply(canonicalize_genre)
    subset_df['genres_str'] = subset_df['genres_str'].apply(canonicalize_genre)
    
    # Remove comics/graphic books from full dataset (matching subset preprocessing)
    full_df = full_df[~full_df['genres_str'].str.contains('comics|graphic', case=False, na=False)]
    logger.info(f"After removing comics/graphic: {len(full_df):,} books in full dataset")
    
    # Compute pop_tier for full dataset (subset already has it)
    full_df = compute_pop_tier(full_df)
    
    # Create series flag
    full_df['series_flag'] = (full_df['series_id'].astype(str) != 'stand_alone').astype(int)
    if 'series_flag' not in subset_df.columns:
        subset_df['series_flag'] = (subset_df['series_id'].astype(str) != 'stand_alone').astype(int)
    
    return full_df, subset_df


def compute_full_dataset_eda(full_df: pd.DataFrame) -> Dict:
    """Compute comprehensive EDA statistics for full dataset."""
    logger.info("Computing full dataset EDA...")
    
    results = {}
    
    # Basic info
    results['total_books'] = len(full_df)
    results['total_authors'] = full_df['author_id'].nunique()
    results['total_series'] = full_df[full_df['series_id'] != 'stand_alone']['series_id'].nunique()
    
    # Publication year
    year_col = 'publication_year'
    if year_col in full_df.columns:
        year_data = full_df[year_col].dropna()
        results['publication_year'] = {
            'mean': float(year_data.mean()),
            'median': float(year_data.median()),
            'std': float(year_data.std()),
            'min': int(year_data.min()),
            'max': int(year_data.max()),
            'q25': float(year_data.quantile(0.25)),
            'q75': float(year_data.quantile(0.75)),
            'distribution': year_data.value_counts().sort_index().to_dict()
        }
        # Decade distribution
        full_df['decade'] = (full_df[year_col] // 10) * 10
        results['decade_distribution'] = full_df['decade'].value_counts().sort_index().to_dict()
    
    # Genre distribution
    if 'genres_str' in full_df.columns:
        # Get all unique genres
        all_genres = []
        for genres in full_df['genres_str'].dropna():
            if isinstance(genres, str):
                all_genres.extend([g.strip() for g in genres.split(',')])
        
        genre_counts = pd.Series(all_genres).value_counts()
        results['genre_distribution'] = genre_counts.head(30).to_dict()
        results['total_unique_genres'] = len(genre_counts)
        results['top_20_genres'] = genre_counts.head(20).to_dict()
    
    # Ratings distribution
    rating_col = 'average_rating_weighted_mean'
    if rating_col in full_df.columns:
        rating_data = full_df[rating_col].dropna()
        results['ratings'] = {
            'mean': float(rating_data.mean()),
            'median': float(rating_data.median()),
            'std': float(rating_data.std()),
            'min': float(rating_data.min()),
            'max': float(rating_data.max()),
            'q25': float(rating_data.quantile(0.25)),
            'q75': float(rating_data.quantile(0.75)),
        }
        # Distribution shape
        try:
            stat, p = normaltest(rating_data)
            results['ratings']['normality_test'] = {'statistic': float(stat), 'p_value': float(p)}
        except:
            pass
    
    # Engagement metrics
    for col in ['ratings_count_sum', 'text_reviews_count_sum']:
        if col in full_df.columns:
            data = full_df[col].dropna()
            results[col] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75)),
            }
    
    # Page count
    if 'num_pages_median' in full_df.columns:
        pages_data = full_df['num_pages_median'].dropna()
        results['num_pages_median'] = {
            'mean': float(pages_data.mean()),
            'median': float(pages_data.median()),
            'std': float(pages_data.std()),
            'min': float(pages_data.min()),
            'max': float(pages_data.max()),
            'q25': float(pages_data.quantile(0.25)),
            'q75': float(pages_data.quantile(0.75)),
        }
    
    # Series vs standalone
    if 'series_flag' in full_df.columns:
        series_counts = full_df['series_flag'].value_counts()
        results['series_status'] = {
            'standalone': int(series_counts.get(0, 0)),
            'series': int(series_counts.get(1, 0)),
            'proportion_series': float(series_counts.get(1, 0) / len(full_df))
        }
    
    # Author metrics
    for col in ['author_average_rating', 'author_ratings_count']:
        if col in full_df.columns:
            data = full_df[col].dropna()
            results[col] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
            }
    
    # Missing data assessment
    missing_data = full_df.isnull().sum()
    missing_pct = (missing_data / len(full_df)) * 100
    results['missing_data'] = {
        'counts': missing_data[missing_data > 0].to_dict(),
        'percentages': missing_pct[missing_pct > 0].to_dict()
    }
    
    # Pop tier distribution
    if 'pop_tier' in full_df.columns:
        tier_counts = full_df['pop_tier'].value_counts()
        results['pop_tier_distribution'] = tier_counts.to_dict()
    
    logger.info("Full dataset EDA completed")
    return results


def compute_effect_size_categorical(contingency_table: pd.DataFrame) -> float:
    """Compute Cramér's V for categorical variables."""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    return cramers_v


def compute_effect_size_continuous(data1: pd.Series, data2: pd.Series) -> float:
    """Compute Cohen's d for continuous variables."""
    n1, n2 = len(data1), len(data2)
    var1, var2 = data1.var(ddof=1), data2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    cohens_d = (data1.mean() - data2.mean()) / pooled_std
    return cohens_d


def perform_cross_corpus_validation(full_df: pd.DataFrame, subset_df: pd.DataFrame) -> Dict:
    """Perform statistical tests comparing full dataset vs subset."""
    logger.info("Performing cross-corpus validation...")
    
    results = {}
    
    # Genre distribution comparison
    logger.info("Comparing genre distributions...")
    if 'genres_str' in full_df.columns and 'genres_str' in subset_df.columns:
        # Get all unique genres
        full_genres = []
        for genres in full_df['genres_str'].dropna():
            if isinstance(genres, str):
                full_genres.extend([g.strip() for g in genres.split(',')])
        
        subset_genres = []
        for genres in subset_df['genres_str'].dropna():
            if isinstance(genres, str):
                subset_genres.extend([g.strip() for g in genres.split(',')])
        
        # Get common genres (top 20 from full dataset)
        full_genre_counts = pd.Series(full_genres).value_counts()
        common_genres = full_genre_counts.head(20).index.tolist()
        
        # Create frequency vectors
        full_freq = [full_genres.count(g) for g in common_genres]
        subset_freq = [subset_genres.count(g) for g in common_genres]
        
        # Normalize to proportions
        full_total = sum(full_freq)
        subset_total = sum(subset_freq)
        full_prop = [f / full_total for f in full_freq]
        subset_prop = [s / subset_total for s in subset_freq]
        
        # Chi-square test
        try:
            chi2_stat, p_value = chisquare(subset_prop, full_prop)
            results['genre_distribution'] = {
                'test': 'chi-square',
                'statistic': float(chi2_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'full_genre_counts': dict(zip(common_genres, full_freq)),
                'subset_genre_counts': dict(zip(common_genres, subset_freq)),
                'full_proportions': dict(zip(common_genres, full_prop)),
                'subset_proportions': dict(zip(common_genres, subset_prop)),
            }
        except Exception as e:
            logger.warning(f"Genre chi-square test failed: {e}")
            results['genre_distribution'] = {'error': str(e)}
    
    # Publication year comparison
    logger.info("Comparing publication year distributions...")
    if 'publication_year' in full_df.columns and 'publication_year' in subset_df.columns:
        # Create decade bins for chi-square
        full_df_copy = full_df.copy()
        subset_df_copy = subset_df.copy()
        full_df_copy['decade'] = (full_df_copy['publication_year'] // 10) * 10
        subset_df_copy['decade'] = (subset_df_copy['publication_year'] // 10) * 10
        
        # Create contingency table properly
        full_decades = full_df_copy['decade'].value_counts().sort_index()
        subset_decades = subset_df_copy['decade'].value_counts().sort_index()
        
        # Get all decades
        all_decades = sorted(set(full_decades.index) | set(subset_decades.index))
        decade_contingency = pd.DataFrame({
            'full': [full_decades.get(d, 0) for d in all_decades],
            'subset': [subset_decades.get(d, 0) for d in all_decades]
        }, index=all_decades)
        
        if decade_contingency.shape[0] > 1 and decade_contingency.shape[1] > 1:
            try:
                chi2_stat, p_value, dof, expected = chi2_contingency(decade_contingency)
                cramers_v = compute_effect_size_categorical(decade_contingency)
                results['publication_year'] = {
                    'test': 'chi-square (decade bins)',
                    'statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'dof': int(dof),
                    'significant': p_value < 0.05,
                    'effect_size_cramers_v': float(cramers_v),
                    'full_distribution': full_decades.to_dict(),
                    'subset_distribution': subset_decades.to_dict(),
                }
            except Exception as e:
                logger.warning(f"Publication year chi-square test failed: {e}")
        
        # Also compare continuous year with Mann-Whitney U
        full_years = full_df['publication_year'].dropna()
        subset_years = subset_df['publication_year'].dropna()
        if len(full_years) > 0 and len(subset_years) > 0:
            try:
                stat, p_value = mannwhitneyu(full_years, subset_years, alternative='two-sided')
                cohens_d = compute_effect_size_continuous(full_years, subset_years)
                results['publication_year_continuous'] = {
                    'test': 'Mann-Whitney U',
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'effect_size_cohens_d': float(cohens_d),
                    'full_mean': float(full_years.mean()),
                    'subset_mean': float(subset_years.mean()),
                }
            except Exception as e:
                logger.warning(f"Publication year Mann-Whitney U test failed: {e}")
    
    # Series status comparison
    logger.info("Comparing series status...")
    if 'series_flag' in full_df.columns and 'series_flag' in subset_df.columns:
        series_contingency = pd.DataFrame({
            'full': full_df['series_flag'].value_counts().sort_index(),
            'subset': subset_df['series_flag'].value_counts().sort_index()
        }).fillna(0)
        
        if series_contingency.shape[0] > 1:
            try:
                chi2_stat, p_value, dof, expected = chi2_contingency(series_contingency)
                cramers_v = compute_effect_size_categorical(series_contingency)
                results['series_status'] = {
                    'test': 'chi-square',
                    'statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'dof': int(dof),
                    'significant': p_value < 0.05,
                    'effect_size_cramers_v': float(cramers_v),
                    'full_proportions': {
                        'standalone': float((full_df['series_flag'] == 0).sum() / len(full_df)),
                        'series': float((full_df['series_flag'] == 1).sum() / len(full_df))
                    },
                    'subset_proportions': {
                        'standalone': float((subset_df['series_flag'] == 0).sum() / len(subset_df)),
                        'series': float((subset_df['series_flag'] == 1).sum() / len(subset_df))
                    }
                }
            except Exception as e:
                logger.warning(f"Series status chi-square test failed: {e}")
    
    # Continuous variables: Ratings, engagement metrics, page counts
    continuous_vars = [
        'average_rating_weighted_mean',
        'ratings_count_sum',
        'text_reviews_count_sum',
        'num_pages_median',
        'author_average_rating',
        'author_ratings_count'
    ]
    
    for var in continuous_vars:
        if var in full_df.columns and var in subset_df.columns:
            logger.info(f"Comparing {var}...")
            full_data = full_df[var].dropna()
            subset_data = subset_df[var].dropna()
            
            if len(full_data) > 0 and len(subset_data) > 0:
                try:
                    stat, p_value = mannwhitneyu(full_data, subset_data, alternative='two-sided')
                    cohens_d = compute_effect_size_continuous(full_data, subset_data)
                    results[var] = {
                        'test': 'Mann-Whitney U',
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'effect_size_cohens_d': float(cohens_d),
                        'full_mean': float(full_data.mean()),
                        'full_median': float(full_data.median()),
                        'subset_mean': float(subset_data.mean()),
                        'subset_median': float(subset_data.median()),
                    }
                except Exception as e:
                    logger.warning(f"{var} Mann-Whitney U test failed: {e}")
    
    # Stratified analysis by pop_tier
    logger.info("Performing stratified analysis by pop_tier...")
    if 'pop_tier' in full_df.columns and 'pop_tier' in subset_df.columns:
        stratified_results = {}
        for tier in ['thrash', 'mid', 'top']:
            full_tier = full_df[full_df['pop_tier'] == tier]
            subset_tier = subset_df[subset_df['pop_tier'] == tier]
            
            if len(full_tier) > 0 and len(subset_tier) > 0:
                tier_results = {}
                for var in ['ratings_count_sum', 'text_reviews_count_sum', 'average_rating_weighted_mean']:
                    if var in full_tier.columns and var in subset_tier.columns:
                        full_data = full_tier[var].dropna()
                        subset_data = subset_tier[var].dropna()
                        if len(full_data) > 0 and len(subset_data) > 0:
                            try:
                                stat, p_value = mannwhitneyu(full_data, subset_data, alternative='two-sided')
                                tier_results[var] = {
                                    'p_value': float(p_value),
                                    'full_median': float(full_data.median()),
                                    'subset_median': float(subset_data.median()),
                                }
                            except:
                                pass
                stratified_results[tier] = tier_results
        results['stratified_by_tier'] = stratified_results
    
    logger.info("Cross-corpus validation completed")
    return results


def generate_visualizations(full_df: pd.DataFrame, subset_df: pd.DataFrame, 
                           eda_results: Dict, validation_results: Dict):
    """Generate all visualizations for the report."""
    logger.info("Generating visualizations...")
    
    # 1. Publication year histograms
    if 'publication_year' in full_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Full dataset
        axes[0, 0].hist(full_df['publication_year'].dropna(), bins=18, 
                       color=ANTIQUE_COLORS[0], alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Full Dataset: Publication Year Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Publication Year')
        axes[0, 0].set_ylabel('Number of Books')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Subset
        axes[0, 1].hist(subset_df['publication_year'].dropna(), bins=18,
                       color=ANTIQUE_COLORS[1], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Subset (6,000 books): Publication Year Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Publication Year')
        axes[0, 1].set_ylabel('Number of Books')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overlay comparison
        axes[1, 0].hist(full_df['publication_year'].dropna(), bins=18, 
                       color=ANTIQUE_COLORS[0], alpha=0.5, label='Full Dataset', edgecolor='black')
        axes[1, 0].hist(subset_df['publication_year'].dropna(), bins=18,
                       color=ANTIQUE_COLORS[1], alpha=0.5, label='Subset', edgecolor='black')
        axes[1, 0].set_title('Publication Year: Full vs Subset Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Publication Year')
        axes[1, 0].set_ylabel('Number of Books')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # By tier (subset only)
        if 'pop_tier' in subset_df.columns:
            for tier, color in zip(['thrash', 'mid', 'top'], ANTIQUE_COLORS[:3]):
                tier_data = subset_df[subset_df['pop_tier'] == tier]['publication_year'].dropna()
                axes[1, 1].hist(tier_data, bins=18, alpha=0.6, label=tier.capitalize(), 
                               color=color, edgecolor='black')
            axes[1, 1].set_title('Subset: Publication Year by Popularity Tier', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Publication Year')
            axes[1, 1].set_ylabel('Number of Books')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'publication_year_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved publication_year_distributions.png")
    
    # 2. Genre frequency bar charts
    if 'genres_str' in full_df.columns:
        # Get top genres
        all_genres_full = []
        for genres in full_df['genres_str'].dropna():
            if isinstance(genres, str):
                all_genres_full.extend([g.strip() for g in genres.split(',')])
        
        all_genres_subset = []
        for genres in subset_df['genres_str'].dropna():
            if isinstance(genres, str):
                all_genres_subset.extend([g.strip() for g in genres.split(',')])
        
        full_genre_counts = pd.Series(all_genres_full).value_counts().head(20)
        subset_genre_counts = pd.Series(all_genres_subset).value_counts().head(20)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Full dataset
        axes[0].barh(range(len(full_genre_counts)), full_genre_counts.values, 
                    color=ANTIQUE_COLORS[0], alpha=0.7, edgecolor='black')
        axes[0].set_yticks(range(len(full_genre_counts)))
        axes[0].set_yticklabels(full_genre_counts.index, fontsize=9)
        axes[0].set_xlabel('Frequency', fontweight='bold')
        axes[0].set_title('Full Dataset: Top 20 Genres', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].invert_yaxis()
        
        # Subset
        axes[1].barh(range(len(subset_genre_counts)), subset_genre_counts.values,
                     color=ANTIQUE_COLORS[1], alpha=0.7, edgecolor='black')
        axes[1].set_yticks(range(len(subset_genre_counts)))
        axes[1].set_yticklabels(subset_genre_counts.index, fontsize=9)
        axes[1].set_xlabel('Frequency', fontweight='bold')
        axes[1].set_title('Subset (6,000 books): Top 20 Genres', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'genre_frequency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved genre_frequency_comparison.png")
    
    # 3. Ratings distribution
    if 'average_rating_weighted_mean' in full_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Full dataset
        axes[0, 0].hist(full_df['average_rating_weighted_mean'].dropna(), bins=50,
                       color=ANTIQUE_COLORS[0], alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Full Dataset: Average Rating Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Average Rating (weighted mean)')
        axes[0, 0].set_ylabel('Number of Books')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Subset
        axes[0, 1].hist(subset_df['average_rating_weighted_mean'].dropna(), bins=50,
                       color=ANTIQUE_COLORS[1], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Subset: Average Rating Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Average Rating (weighted mean)')
        axes[0, 1].set_ylabel('Number of Books')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overlay
        axes[1, 0].hist(full_df['average_rating_weighted_mean'].dropna(), bins=50,
                       color=ANTIQUE_COLORS[0], alpha=0.5, label='Full Dataset', edgecolor='black')
        axes[1, 0].hist(subset_df['average_rating_weighted_mean'].dropna(), bins=50,
                       color=ANTIQUE_COLORS[1], alpha=0.5, label='Subset', edgecolor='black')
        axes[1, 0].set_title('Average Rating: Full vs Subset Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Average Rating (weighted mean)')
        axes[1, 0].set_ylabel('Number of Books')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # By tier (subset)
        if 'pop_tier' in subset_df.columns:
            for tier, color in zip(['thrash', 'mid', 'top'], ANTIQUE_COLORS[:3]):
                tier_data = subset_df[subset_df['pop_tier'] == tier]['average_rating_weighted_mean'].dropna()
                axes[1, 1].hist(tier_data, bins=30, alpha=0.6, label=tier.capitalize(),
                               color=color, edgecolor='black')
            axes[1, 1].set_title('Subset: Average Rating by Popularity Tier', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Average Rating (weighted mean)')
            axes[1, 1].set_ylabel('Number of Books')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ratings_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved ratings_distributions.png")
    
    # 4. Engagement metrics boxplots
    engagement_vars = ['ratings_count_sum', 'text_reviews_count_sum']
    for var in engagement_vars:
        if var in full_df.columns and var in subset_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Full vs Subset comparison
            data_to_plot = [
                full_df[var].dropna(),
                subset_df[var].dropna()
            ]
            axes[0].boxplot(data_to_plot, labels=['Full Dataset', 'Subset'], patch_artist=True)
            for patch, color in zip(axes[0].artists, ANTIQUE_COLORS[:2]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[0].set_ylabel(var.replace('_', ' ').title(), fontweight='bold')
            axes[0].set_title(f'{var.replace("_", " ").title()}: Full vs Subset', fontsize=12, fontweight='bold')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # By tier (subset)
            if 'pop_tier' in subset_df.columns:
                tier_data = [subset_df[subset_df['pop_tier'] == tier][var].dropna() 
                            for tier in ['thrash', 'mid', 'top']]
                bp = axes[1].boxplot(tier_data, labels=['Thrash', 'Mid', 'Top'], patch_artist=True)
                for patch, color in zip(bp['boxes'], ANTIQUE_COLORS[:3]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                axes[1].set_ylabel(var.replace('_', ' ').title(), fontweight='bold')
                axes[1].set_title(f'Subset: {var.replace("_", " ").title()} by Tier', fontsize=12, fontweight='bold')
                axes[1].set_yscale('log')
                axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            safe_var_name = var.replace('_', '_')
            plt.savefig(FIGURES_DIR / f'{safe_var_name}_boxplots.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved {safe_var_name}_boxplots.png")
    
    # 5. Page count distributions
    if 'num_pages_median' in full_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Full vs Subset
        axes[0].hist(full_df['num_pages_median'].dropna(), bins=50, 
                    color=ANTIQUE_COLORS[0], alpha=0.5, label='Full Dataset', edgecolor='black')
        axes[0].hist(subset_df['num_pages_median'].dropna(), bins=50,
                    color=ANTIQUE_COLORS[1], alpha=0.5, label='Subset', edgecolor='black')
        axes[0].set_title('Page Count: Full vs Subset Comparison', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Median Page Count')
        axes[0].set_ylabel('Number of Books')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # By tier (subset)
        if 'pop_tier' in subset_df.columns:
            for tier, color in zip(['thrash', 'mid', 'top'], ANTIQUE_COLORS[:3]):
                tier_data = subset_df[subset_df['pop_tier'] == tier]['num_pages_median'].dropna()
                axes[1].hist(tier_data, bins=30, alpha=0.6, label=tier.capitalize(),
                           color=color, edgecolor='black')
            axes[1].set_title('Subset: Page Count by Popularity Tier', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Median Page Count')
            axes[1].set_ylabel('Number of Books')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'page_count_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved page_count_distributions.png")
    
    # 6. Series status
    if 'series_flag' in full_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Full dataset
        series_counts_full = full_df['series_flag'].value_counts()
        axes[0].pie(series_counts_full.values, labels=['Standalone', 'Series'], 
                   autopct='%1.1f%%', colors=ANTIQUE_COLORS[:2], startangle=90)
        axes[0].set_title('Full Dataset: Series vs Standalone', fontsize=12, fontweight='bold')
        
        # Subset
        series_counts_subset = subset_df['series_flag'].value_counts()
        axes[1].pie(series_counts_subset.values, labels=['Standalone', 'Series'],
                   autopct='%1.1f%%', colors=ANTIQUE_COLORS[:2], startangle=90)
        axes[1].set_title('Subset: Series vs Standalone', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'series_status_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved series_status_comparison.png")
    
    logger.info("All visualizations generated")


def export_reports(full_df: pd.DataFrame, subset_df: pd.DataFrame,
                   eda_results: Dict, validation_results: Dict):
    """Export markdown report and CSV tables."""
    logger.info("Exporting reports...")
    
    # 1. Full dataset summary statistics CSV
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()
    summary_stats = full_df[numeric_cols].describe().T
    summary_stats['missing_count'] = full_df[numeric_cols].isnull().sum()
    summary_stats['missing_pct'] = (summary_stats['missing_count'] / len(full_df)) * 100
    summary_stats.to_csv(REPORTS_DIR / 'full_dataset_summary_statistics.csv')
    logger.info("Saved full_dataset_summary_statistics.csv")
    
    # 2. Genre distributions
    if 'genres_str' in full_df.columns:
        all_genres_full = []
        for genres in full_df['genres_str'].dropna():
            if isinstance(genres, str):
                all_genres_full.extend([g.strip() for g in genres.split(',')])
        
        genre_counts_full = pd.Series(all_genres_full).value_counts()
        genre_df_full = pd.DataFrame({
            'genre': genre_counts_full.index,
            'frequency': genre_counts_full.values,
            'percentage': (genre_counts_full.values / len(full_df)) * 100
        })
        genre_df_full.to_csv(REPORTS_DIR / 'genre_distribution_full.csv', index=False)
        logger.info("Saved genre_distribution_full.csv")
        
        all_genres_subset = []
        for genres in subset_df['genres_str'].dropna():
            if isinstance(genres, str):
                all_genres_subset.extend([g.strip() for g in genres.split(',')])
        
        genre_counts_subset = pd.Series(all_genres_subset).value_counts()
        genre_df_subset = pd.DataFrame({
            'genre': genre_counts_subset.index,
            'frequency': genre_counts_subset.values,
            'percentage': (genre_counts_subset.values / len(subset_df)) * 100
        })
        genre_df_subset.to_csv(REPORTS_DIR / 'genre_distribution_subset.csv', index=False)
        logger.info("Saved genre_distribution_subset.csv")
    
    # 3. Publication year distribution
    if 'publication_year' in full_df.columns:
        year_dist_full = full_df['publication_year'].value_counts().sort_index()
        year_dist_subset = subset_df['publication_year'].value_counts().sort_index()
        year_df = pd.DataFrame({
            'year': year_dist_full.index,
            'full_dataset_count': year_dist_full.values,
            'subset_count': year_dist_subset.reindex(year_dist_full.index, fill_value=0).values
        })
        year_df['full_dataset_pct'] = (year_df['full_dataset_count'] / len(full_df)) * 100
        year_df['subset_pct'] = (year_df['subset_count'] / len(subset_df)) * 100
        year_df.to_csv(REPORTS_DIR / 'publication_year_distribution.csv', index=False)
        logger.info("Saved publication_year_distribution.csv")
    
    # 4. Cross-corpus comparison table
    comparison_data = []
    key_vars = [
        'average_rating_weighted_mean',
        'ratings_count_sum',
        'text_reviews_count_sum',
        'num_pages_median',
        'author_average_rating',
        'author_ratings_count'
    ]
    
    for var in key_vars:
        if var in full_df.columns and var in subset_df.columns:
            full_data = full_df[var].dropna()
            subset_data = subset_df[var].dropna()
            comparison_data.append({
                'variable': var,
                'full_mean': float(full_data.mean()),
                'full_median': float(full_data.median()),
                'full_std': float(full_data.std()),
                'subset_mean': float(subset_data.mean()),
                'subset_median': float(subset_data.median()),
                'subset_std': float(subset_data.std()),
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(REPORTS_DIR / 'cross_corpus_comparison.csv', index=False)
    logger.info("Saved cross_corpus_comparison.csv")
    
    # 5. Statistical test results
    test_results = []
    for test_name, test_data in validation_results.items():
        if isinstance(test_data, dict) and 'test' in test_data:
            test_results.append({
                'test_name': test_name,
                'test_type': test_data.get('test', ''),
                'statistic': test_data.get('statistic', np.nan),
                'p_value': test_data.get('p_value', np.nan),
                'significant': test_data.get('significant', False),
                'effect_size': test_data.get('effect_size_cramers_v', test_data.get('effect_size_cohens_d', np.nan)),
            })
    
    if test_results:
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(REPORTS_DIR / 'statistical_test_results.csv', index=False)
        logger.info("Saved statistical_test_results.csv")
    
    # 6. Markdown report
    markdown_content = generate_markdown_report(full_df, subset_df, eda_results, validation_results)
    with open(REPORTS_DIR / 'CORPUS_STATISTICS_REPORT.md', 'w') as f:
        f.write(markdown_content)
    logger.info("Saved CORPUS_STATISTICS_REPORT.md")
    
    logger.info("All reports exported")


def generate_markdown_report(full_df: pd.DataFrame, subset_df: pd.DataFrame,
                            eda_results: Dict, validation_results: Dict) -> str:
    """Generate comprehensive markdown report."""
    
    md = []
    md.append("# Corpus Construction Statistical Analysis Report\n")
    md.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("---\n\n")
    
    # Executive Summary
    md.append("## Executive Summary\n\n")
    md.append(f"This report presents comprehensive statistical analysis of the romance novel corpus construction, ")
    md.append(f"including exploratory data analysis (EDA) on the full dataset ({eda_results['total_books']:,} books) ")
    md.append(f"and cross-corpus validation comparing the full dataset to the 6,000-book research subset.\n\n")
    
    # Full Dataset Characteristics
    md.append("## Full Dataset Characteristics\n\n")
    md.append(f"### Overview\n\n")
    md.append(f"- **Total Books:** {eda_results['total_books']:,}\n")
    md.append(f"- **Total Authors:** {eda_results['total_authors']:,}\n")
    md.append(f"- **Total Series:** {eda_results['total_series']:,}\n\n")
    
    if 'publication_year' in eda_results:
        py = eda_results['publication_year']
        md.append(f"### Publication Year\n\n")
        md.append(f"- **Range:** {py['min']} - {py['max']}\n")
        md.append(f"- **Mean:** {py['mean']:.1f}\n")
        md.append(f"- **Median:** {py['median']:.1f}\n")
        md.append(f"- **Standard Deviation:** {py['std']:.1f}\n")
        md.append(f"- **Quartiles:** Q1={py['q25']:.1f}, Q3={py['q75']:.1f}\n\n")
    
    if 'ratings' in eda_results:
        r = eda_results['ratings']
        md.append(f"### Average Rating (Weighted Mean)\n\n")
        md.append(f"- **Mean:** {r['mean']:.3f}\n")
        md.append(f"- **Median:** {r['median']:.3f}\n")
        md.append(f"- **Range:** {r['min']:.3f} - {r['max']:.3f}\n")
        md.append(f"- **Quartiles:** Q1={r['q25']:.3f}, Q3={r['q75']:.3f}\n\n")
    
    if 'ratings_count_sum' in eda_results:
        rc = eda_results['ratings_count_sum']
        md.append(f"### Ratings Count (Sum across editions)\n\n")
        md.append(f"- **Mean:** {rc['mean']:,.0f}\n")
        md.append(f"- **Median:** {rc['median']:,.0f}\n")
        md.append(f"- **Range:** {rc['min']:,.0f} - {rc['max']:,.0f}\n\n")
    
    if 'text_reviews_count_sum' in eda_results:
        trc = eda_results['text_reviews_count_sum']
        md.append(f"### Text Reviews Count (Sum across editions)\n\n")
        md.append(f"- **Mean:** {trc['mean']:,.0f}\n")
        md.append(f"- **Median:** {trc['median']:,.0f}\n")
        md.append(f"- **Range:** {trc['min']:,.0f} - {trc['max']:,.0f}\n\n")
    
    if 'num_pages_median' in eda_results:
        npc = eda_results['num_pages_median']
        md.append(f"### Page Count (Median)\n\n")
        md.append(f"- **Mean:** {npc['mean']:.1f}\n")
        md.append(f"- **Median:** {npc['median']:.1f}\n")
        md.append(f"- **Range:** {npc['min']:.0f} - {npc['max']:.0f}\n\n")
    
    if 'series_status' in eda_results:
        ss = eda_results['series_status']
        md.append(f"### Series Status\n\n")
        md.append(f"- **Standalone:** {ss['standalone']:,} ({ss['proportion_series']*100:.1f}%)\n")
        md.append(f"- **Series:** {ss['series']:,} ({100-ss['proportion_series']*100:.1f}%)\n\n")
    
    if 'genre_distribution' in eda_results:
        md.append(f"### Genre Distribution\n\n")
        md.append(f"- **Total Unique Genres:** {eda_results['total_unique_genres']}\n")
        md.append(f"**Top 10 Genres:**\n\n")
        for i, (genre, count) in enumerate(list(eda_results['top_20_genres'].items())[:10], 1):
            md.append(f"{i}. {genre}: {count:,} occurrences\n")
        md.append("\n")
    
    # Cross-Corpus Validation
    md.append("## Cross-Corpus Validation\n\n")
    md.append("This section presents statistical comparisons between the full dataset and the 6,000-book subset ")
    md.append("to validate the representativeness of the research corpus.\n\n")
    
    # Statistical Tests
    md.append("### Statistical Test Results\n\n")
    md.append("| Test | Test Type | Statistic | p-value | Significant | Effect Size |\n")
    md.append("|------|-----------|-----------|--------|------------|-------------|\n")
    
    for test_name, test_data in validation_results.items():
        if isinstance(test_data, dict) and 'test' in test_data:
            test_type = test_data.get('test', '')
            stat = test_data.get('statistic', np.nan)
            p_val = test_data.get('p_value', np.nan)
            sig = 'Yes' if test_data.get('significant', False) else 'No'
            effect_size = test_data.get('effect_size_cramers_v', test_data.get('effect_size_cohens_d', np.nan))
            
            md.append(f"| {test_name} | {test_type} | {stat:.4f} | {p_val:.4f} | {sig} | {effect_size:.4f} |\n")
    
    md.append("\n")
    
    # Key Findings
    md.append("### Key Findings\n\n")
    
    # Check genre distribution
    if 'genre_distribution' in validation_results:
        gd = validation_results['genre_distribution']
        if gd.get('significant', False):
            md.append("- ⚠️ **Genre distribution differs significantly** between full dataset and subset (p < 0.05)\n")
        else:
            md.append("- ✅ **Genre distribution is statistically similar** between full dataset and subset (p >= 0.05)\n")
    
    # Check publication year
    if 'publication_year' in validation_results:
        py_test = validation_results['publication_year']
        if py_test.get('significant', False):
            md.append("- ⚠️ **Publication year distribution differs significantly** (p < 0.05)\n")
        else:
            md.append("- ✅ **Publication year distribution is statistically similar** (p >= 0.05)\n")
    
    # Check series status
    if 'series_status' in validation_results:
        ss_test = validation_results['series_status']
        if ss_test.get('significant', False):
            md.append("- ⚠️ **Series status distribution differs significantly** (p < 0.05)\n")
        else:
            md.append("- ✅ **Series status distribution is statistically similar** (p >= 0.05)\n")
    
    # Continuous variables
    continuous_vars = ['average_rating_weighted_mean', 'ratings_count_sum', 'text_reviews_count_sum', 
                      'num_pages_median']
    for var in continuous_vars:
        if var in validation_results:
            v_test = validation_results[var]
            if v_test.get('significant', False):
                md.append(f"- ⚠️ **{var.replace('_', ' ').title()} differs significantly** (p={v_test.get('p_value', 0):.4f})\n")
            else:
                md.append(f"- ✅ **{var.replace('_', ' ').title()} is statistically similar** (p={v_test.get('p_value', 1):.4f})\n")
    
    md.append("\n")
    
    # Visualizations
    md.append("## Visualizations\n\n")
    md.append("All visualizations are saved in the `figures/` directory:\n\n")
    md.append("- `publication_year_distributions.png`: Publication year distributions for full dataset, subset, and comparison\n")
    md.append("- `genre_frequency_comparison.png`: Top 20 genre frequencies for full dataset and subset\n")
    md.append("- `ratings_distributions.png`: Average rating distributions for full dataset, subset, and by tier\n")
    md.append("- `ratings_count_sum_boxplots.png`: Boxplots comparing ratings counts\n")
    md.append("- `text_reviews_count_sum_boxplots.png`: Boxplots comparing review counts\n")
    md.append("- `page_count_distributions.png`: Page count distributions for full dataset and subset\n")
    md.append("- `series_status_comparison.png`: Series vs standalone proportions\n\n")
    
    # Data Files
    md.append("## Data Files\n\n")
    md.append("All CSV tables are saved in the `reports/corpus_construction/` directory:\n\n")
    md.append("- `full_dataset_summary_statistics.csv`: Descriptive statistics for all numeric variables\n")
    md.append("- `genre_distribution_full.csv`: Genre frequencies in full dataset\n")
    md.append("- `genre_distribution_subset.csv`: Genre frequencies in subset\n")
    md.append("- `cross_corpus_comparison.csv`: Side-by-side comparison of key metrics\n")
    md.append("- `statistical_test_results.csv`: All statistical test results\n")
    md.append("- `publication_year_distribution.csv`: Year counts for full and subset\n\n")
    
    return ''.join(md)


def main():
    """Main execution function."""
    logger.info("Starting corpus construction statistical analysis...")
    
    try:
        # Load datasets
        full_df, subset_df = load_datasets()
        
        # Compute full dataset EDA
        eda_results = compute_full_dataset_eda(full_df)
        
        # Perform cross-corpus validation
        validation_results = perform_cross_corpus_validation(full_df, subset_df)
        
        # Generate visualizations
        generate_visualizations(full_df, subset_df, eda_results, validation_results)
        
        # Export reports
        export_reports(full_df, subset_df, eda_results, validation_results)
        
        logger.info("Analysis complete! All outputs saved to reports/corpus_construction/")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

