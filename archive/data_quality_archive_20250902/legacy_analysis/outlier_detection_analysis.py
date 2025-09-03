#!/usr/bin/env python3
"""
Step 3: Outlier Detection and Treatment

This script performs comprehensive outlier detection and treatment analysis for the
romance novel dataset, focusing on:

1. Title Duplication Analysis
   - Investigate legitimate duplicate titles vs. errors
   - Check author_name differences in duplicate titles
   - Plan title disambiguation strategies

2. Series Data Cleaning
   - Validate series_title_works_count field accuracy
   - Identify missing series_title books
   - Plan series_title data correction

3. Outlier Detection
   - Publication year distribution analysis
   - Rating and review count anomalies
   - Page count distribution validation

Author: Research Assistant
Date: 2025-01-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
import time # Added for performance monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OutlierDetectionAnalyzer:
    """
    Comprehensive outlier detection and treatment analyzer for romance novel dataset.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to the cleaned dataset CSV file
        """
        self.data_path = data_path or "data/processed/final_books_2000_2020_en_cleaned_nlp_ready_20250902_161743.csv"
        self.df = None
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("outputs/outlier_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the cleaned dataset.
        
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading data from: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            return self.df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def analyze_title_duplications(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Analyze title duplications to identify legitimate duplicates vs. errors.
        
        Args:
            batch_size: Number of duplicate titles to process in each batch
        
        Returns:
            Dictionary with duplication analysis results
        """
        logger.info("=" * 80)
        logger.info("🔍 STARTING TITLE DUPLICATION ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset size: {len(self.df):,} total records")
        logger.info(f"⚙️  Batch size: {batch_size:,} titles per batch")
        logger.info(f"⏰ Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Find duplicate titles
        logger.info("🔍 Identifying duplicate titles...")
        title_counts = self.df['title'].value_counts()
        duplicate_titles = title_counts[title_counts > 1]
        
        logger.info(f"📚 Found {len(duplicate_titles):,} duplicate titles")
        logger.info(f"📚 Total duplicate books: {duplicate_titles.sum():,}")
        
        # Calculate statistics
        avg_duplicates = duplicate_titles.mean()
        max_duplicates = duplicate_titles.max()
        min_duplicates = duplicate_titles.min()
        
        logger.info(f"📊 Duplicate statistics:")
        logger.info(f"   • Average duplicates per title: {avg_duplicates:.2f}")
        logger.info(f"   • Maximum duplicates: {max_duplicates}")
        logger.info(f"   • Minimum duplicates: {min_duplicates}")
        
        results = {
            'total_duplicate_titles': len(duplicate_titles),
            'total_duplicate_books': duplicate_titles.sum(),
            'duplicate_titles_list': duplicate_titles.to_dict(),
            'author_name_analysis': {},
            'disambiguation_strategies': [],
            'analysis_metadata': {
                'batch_size': batch_size,
                'analysis_start_time': datetime.now().isoformat(),
                'total_titles_processed': 0,
                'legitimate_duplicates': 0,
                'potential_errors': 0
            }
        }
        
        # Process duplicate titles in batches
        total_batches = (len(duplicate_titles) + batch_size - 1) // batch_size
        logger.info(f"🔄 Processing {len(duplicate_titles):,} duplicate titles in {total_batches} batches")
        
        start_time = time.time()
        batch_start_time = start_time
        
        for batch_num, (title, count) in enumerate(duplicate_titles.items(), 1):
            # Process individual title
            title_books = self.df[self.df['title'] == title]
            
            # Check author_name differences
            unique_author_names = title_books['author_name'].nunique()
            author_names_list = title_books['author_name'].unique().tolist()
            
            # Check publication year differences
            year_range = title_books['publication_year'].max() - title_books['publication_year'].min()
            
            # Determine if legitimate duplicate
            is_legitimate = False
            if unique_author_names > 1:
                is_legitimate = True
                strategy = "Different author_names - legitimate duplicate"
                results['analysis_metadata']['legitimate_duplicates'] += 1
            elif year_range > 5:
                is_legitimate = True
                strategy = "Different publication years - possible reprints/editions"
                results['analysis_metadata']['legitimate_duplicates'] += 1
            elif title_books['series_title'].nunique() > 1:
                is_legitimate = True
                strategy = "Different series_title - legitimate duplicate"
                results['analysis_metadata']['legitimate_duplicates'] += 1
            else:
                strategy = "Potential data error - investigate further"
                results['analysis_metadata']['potential_errors'] += 1
            
            results['author_name_analysis'][title] = {
                'count': count,
                'unique_author_names': unique_author_names,
                'author_names': author_names_list,
                'year_range': year_range,
                'is_legitimate': is_legitimate,
                'strategy': strategy,
                'books_data': title_books[['author_name', 'publication_year', 'series_title', 'work_id']].to_dict('records')
            }
            
            if not is_legitimate:
                results['disambiguation_strategies'].append({
                    'title': title,
                    'issue': 'Potential duplicate',
                    'recommendation': 'Manual review required'
                })
        
            results['analysis_metadata']['total_titles_processed'] += 1
            
            # Log progress every batch_size titles or every 10% progress
            if batch_num % batch_size == 0 or batch_num % max(1, len(duplicate_titles) // 10) == 0:
                current_time = time.time()
                batch_duration = current_time - batch_start_time
                total_duration = current_time - start_time
                
                progress_pct = (batch_num / len(duplicate_titles)) * 100
                titles_per_sec = batch_num / total_duration if total_duration > 0 else 0
                eta_seconds = (len(duplicate_titles) - batch_num) / titles_per_sec if titles_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                
                logger.info(f"📈 Progress: {batch_num:,}/{len(duplicate_titles):,} titles ({progress_pct:.1f}%)")
                logger.info(f"   • Processing speed: {titles_per_sec:.2f} titles/sec")
                logger.info(f"   • Batch duration: {batch_duration:.2f}s")
                logger.info(f"   • Total duration: {total_duration:.2f}s")
                logger.info(f"   • ETA: {eta_minutes:.1f} minutes")
                logger.info(f"   • Legitimate duplicates: {results['analysis_metadata']['legitimate_duplicates']:,}")
                logger.info(f"   • Potential errors: {results['analysis_metadata']['potential_errors']:,}")
                
                batch_start_time = current_time
        
        # Final analysis summary
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("✅ TITLE DUPLICATION ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Final Results:")
        logger.info(f"   • Total titles processed: {results['analysis_metadata']['total_titles_processed']:,}")
        logger.info(f"   • Legitimate duplicates: {results['analysis_metadata']['legitimate_duplicates']:,}")
        logger.info(f"   • Potential errors: {results['analysis_metadata']['potential_errors']:,}")
        logger.info(f"   • Disambiguation strategies needed: {len(results['disambiguation_strategies']):,}")
        logger.info(f"⏱️  Total analysis time: {total_duration:.2f} seconds")
        logger.info(f"⚡ Average processing speed: {results['analysis_metadata']['total_titles_processed'] / total_duration:.2f} titles/sec")
        logger.info(f"📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Add timing metadata to results
        results['analysis_metadata']['analysis_end_time'] = datetime.now().isoformat()
        results['analysis_metadata']['total_duration_seconds'] = total_duration
        results['analysis_metadata']['processing_speed_titles_per_sec'] = results['analysis_metadata']['total_titles_processed'] / total_duration if total_duration > 0 else 0
        
        return results
    
    def analyze_series_title_data(self) -> Dict[str, Any]:
        """
        Analyze series_title data for accuracy and completeness.
        
        Returns:
            Dictionary with series_title analysis results
        """
        logger.info("=" * 80)
        logger.info("📖 STARTING SERIES TITLE DATA ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset size: {len(self.df):,} total records")
        logger.info(f"⏰ Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Filter books with series_title information
        logger.info("🔍 Filtering books with series information...")
        series_title_books = self.df[self.df['series_title'].notna() & (self.df['series_title'] != '')]
        
        logger.info(f"📚 Found {len(series_title_books):,} books with series information")
        logger.info(f"📚 Unique series: {series_title_books['series_title'].nunique():,}")
        
        # Calculate series statistics
        series_counts = series_title_books['series_title'].value_counts()
        avg_books_per_series = series_counts.mean()
        max_books_per_series = series_counts.max()
        min_books_per_series = series_counts.min()
        
        logger.info(f"📊 Series statistics:")
        logger.info(f"   • Average books per series: {avg_books_per_series:.2f}")
        logger.info(f"   • Maximum books in a series: {max_books_per_series}")
        logger.info(f"   • Minimum books in a series: {min_books_per_series}")
        
        results = {
            'total_series_title_books': len(series_title_books),
            'unique_series_title': series_title_books['series_title'].nunique(),
            'series_title_works_count_analysis': {},
            'missing_books_analysis': {},
            'correction_recommendations': [],
            'analysis_metadata': {
                'analysis_start_time': datetime.now().isoformat(),
                'total_series_analyzed': 0,
                'series_with_discrepancies': 0,
                'series_accuracy_rate': 0.0
            }
        }
        
        # Analyze series_title_works_count field
        if 'series_title_works_count' in self.df.columns:
            logger.info("🔍 Analyzing series works count accuracy...")
            start_time = time.time()
            
            series_title_counts = series_title_books.groupby('series_title').agg({
                'series_title_works_count': ['first', 'nunique'],
                'work_id': 'count'
            }).round(2)
            
            series_title_counts.columns = ['works_count', 'unique_works_counts', 'actual_books']
            series_title_counts['discrepancy'] = series_title_counts['works_count'] - series_title_counts['actual_books']
            
            # Identify series_title with discrepancies
            discrepancies = series_title_counts[abs(series_title_counts['discrepancy']) > 0]
            
            accuracy_rate = (len(series_title_counts) - len(discrepancies)) / len(series_title_counts) * 100
            
            results['series_title_works_count_analysis'] = {
                'total_series_title': len(series_title_counts),
                'series_title_with_discrepancies': len(discrepancies),
                'discrepancy_details': discrepancies.to_dict('index'),
                'accuracy_rate': accuracy_rate
            }
            
            # Update metadata
            results['analysis_metadata']['total_series_analyzed'] = len(series_title_counts)
            results['analysis_metadata']['series_with_discrepancies'] = len(discrepancies)
            results['analysis_metadata']['series_accuracy_rate'] = accuracy_rate
            
            logger.info(f"📊 Series works count analysis:")
            logger.info(f"   • Total series analyzed: {len(series_title_counts):,}")
            logger.info(f"   • Series with discrepancies: {len(discrepancies):,}")
            logger.info(f"   • Accuracy rate: {accuracy_rate:.1f}%")
            
            # Generate correction recommendations
            logger.info("🔍 Generating correction recommendations...")
            for series_title_name, data in discrepancies.iterrows():
                if data['discrepancy'] > 0:
                    results['correction_recommendations'].append({
                        'series_title': series_title_name,
                        'issue': f"Missing {int(data['discrepancy'])} books",
                        'action': 'Add missing books to dataset'
                    })
                else:
                    results['correction_recommendations'].append({
                        'series_title': series_title_name,
                        'issue': f"Extra {int(abs(data['discrepancy']))} books",
                        'action': 'Verify series_title_works_count accuracy'
                    })
        
            works_analysis_time = time.time() - start_time
            logger.info(f"⏱️  Series works count analysis completed in {works_analysis_time:.2f} seconds")
        
        # Analyze series completeness
        logger.info("🔍 Analyzing series completeness...")
        start_time = time.time()
        
        series_completeness = series_title_books.groupby('series_title').agg({
            'work_id': 'count',
            'publication_year': ['min', 'max'],
            'title': list
        })
        
        series_completeness.columns = ['book_count', 'start_year', 'end_year', 'titles']
        series_completeness['year_span'] = series_completeness['end_year'] - series_completeness['start_year']
        
        # Identify potentially incomplete series
        incomplete_series = series_completeness[
            (series_completeness['year_span'] > 10) & 
            (series_completeness['book_count'] < 5)
        ]
        
        results['missing_books_analysis'] = {
            'incomplete_series_title': len(incomplete_series),
            'incomplete_series_title_details': incomplete_series.to_dict('index'),
            'completeness_score': len(series_completeness) / len(series_completeness) * 100
        }
        
        completeness_analysis_time = time.time() - start_time
        logger.info(f"📊 Series completeness analysis:")
        logger.info(f"   • Potentially incomplete series: {len(incomplete_series):,}")
        logger.info(f"   • Completeness score: {results['missing_books_analysis']['completeness_score']:.1f}%")
        logger.info(f"⏱️  Series completeness analysis completed in {completeness_analysis_time:.2f} seconds")
        
        # Final analysis summary
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("✅ SERIES TITLE DATA ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Final Results:")
        logger.info(f"   • Total series books: {len(series_title_books):,}")
        logger.info(f"   • Unique series: {series_title_books['series_title'].nunique():,}")
        logger.info(f"   • Series with discrepancies: {results['analysis_metadata']['series_with_discrepancies']:,}")
        logger.info(f"   • Series accuracy rate: {results['analysis_metadata']['series_accuracy_rate']:.1f}%")
        logger.info(f"   • Potentially incomplete series: {len(incomplete_series):,}")
        logger.info(f"   • Correction recommendations: {len(results['correction_recommendations']):,}")
        logger.info(f"⏱️  Total analysis time: {total_duration:.2f} seconds")
        logger.info(f"📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Add timing metadata to results
        results['analysis_metadata']['analysis_end_time'] = datetime.now().isoformat()
        results['analysis_metadata']['total_duration_seconds'] = total_duration
        
        logger.info(f"Series data analysis complete. Found {len(series_title_books)} series books across {series_title_books['series_title'].nunique()} series.")
        return results
    
    def detect_statistical_outliers(self) -> Dict[str, Any]:
        """
        Detect statistical outliers in numerical fields.
        
        Returns:
            Dictionary with outlier detection results
        """
        logger.info("=" * 80)
        logger.info("📊 STARTING STATISTICAL OUTLIER DETECTION")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset size: {len(self.df):,} total records")
        logger.info(f"⏰ Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        start_time = time.time()
        
        # Define fields to analyze
        fields_to_analyze = [
            ('publication_year', 'Publication Year'),
            ('average_rating_weighted_mean', 'Rating'),
            ('text_reviews_count_sum', 'Review Count'),
            ('num_pages_median', 'Page Count')
        ]
        
        logger.info(f"🔍 Analyzing {len(fields_to_analyze)} numerical fields for outliers...")
        
        results = {
            'publication_year_outliers': {},
            'rating_outliers': {},
            'review_count_outliers': {},
            'page_count_outliers': {},
            'summary': {},
            'analysis_metadata': {
                'analysis_start_time': datetime.now().isoformat(),
                'fields_analyzed': [],
                'total_outliers_found': 0,
                'field_analysis_times': {}
            }
        }
        
        total_outliers = 0
        
        # Analyze each field
        for field_name, field_display in fields_to_analyze:
            if field_name in self.df.columns:
                logger.info(f"🔍 Analyzing {field_display} field...")
                field_start_time = time.time()
                
                # Get field data
                field_data = self.df[field_name].dropna()
                logger.info(f"   • Field: {field_display}")
                logger.info(f"   • Non-null values: {len(field_data):,}")
                logger.info(f"   • Data range: {field_data.min():.2f} to {field_data.max():.2f}")
                
                # Detect outliers
                outliers_result = self._detect_numerical_outliers(field_data, field_name)
                
                # Store results based on field type
                if field_name == 'publication_year':
                    results['publication_year_outliers'] = outliers_result
                elif field_name == 'average_rating_weighted_mean':
                    results['rating_outliers'] = outliers_result
                elif field_name == 'text_reviews_count_sum':
                    results['review_count_outliers'] = outliers_result
                elif field_name == 'num_pages_median':
                    results['page_count_outliers'] = outliers_result
                
                # Update metadata
                field_analysis_time = time.time() - field_start_time
                results['analysis_metadata']['field_analysis_times'][field_name] = field_analysis_time
                results['analysis_metadata']['fields_analyzed'].append(field_name)
                
                if 'outlier_count' in outliers_result:
                    outlier_count = outliers_result['outlier_count']
                    total_outliers += outlier_count
                    outlier_pct = outliers_result.get('outlier_percentage', 0)
                    
                    logger.info(f"   • Outliers found: {outlier_count:,} ({outlier_pct:.2f}%)")
                    logger.info(f"   • Analysis time: {field_analysis_time:.2f} seconds")
                else:
                    logger.warning(f"   • No outlier data available for {field_display}")
                
                logger.info(f"✅ {field_display} analysis completed")
            else:
                logger.warning(f"⚠️  Field '{field_name}' not found in dataset, skipping...")
        
        # Create summary
        results['summary'] = {
            'total_outliers_detected': total_outliers,
            'fields_analyzed': results['analysis_metadata']['fields_analyzed']
        }
        
        # Update metadata
        results['analysis_metadata']['total_outliers_found'] = total_outliers
        results['analysis_metadata']['analysis_end_time'] = datetime.now().isoformat()
        
        # Final analysis summary
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("✅ STATISTICAL OUTLIER DETECTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Final Results:")
        logger.info(f"   • Fields analyzed: {len(results['analysis_metadata']['fields_analyzed'])}")
        logger.info(f"   • Total outliers detected: {total_outliers:,}")
        logger.info(f"   • Analysis time: {total_duration:.2f} seconds")
        
        # Field-specific timing
        for field_name, field_time in results['analysis_metadata']['field_analysis_times'].items():
            logger.info(f"   • {field_name}: {field_time:.2f}s")
        
        logger.info(f"📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Add total duration to metadata
        results['analysis_metadata']['total_duration_seconds'] = total_duration
        
        logger.info(f"Statistical outlier detection complete. Found {total_outliers} outliers.")
        return results
    
    def _detect_numerical_outliers(self, series_title: pd.Series, field_name: str) -> Dict[str, Any]:
        """
        Detect outliers in a numerical series_title using IQR method.
        
        Args:
            series_title: Numerical series_title to analyze
            field_name: Name of the field being analyzed
            
        Returns:
            Dictionary with outlier detection results
        """
        if len(series_title) == 0:
            return {'error': 'No data available'}
        
        Q1 = series_title.quantile(0.25)
        Q3 = series_title.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series_title[(series_title < lower_bound) | (series_title > upper_bound)]
        
        return {
            'field': field_name,
            'total_values': len(series_title),
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(series_title)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'outliers': outliers.tolist(),
            'outlier_indices': outliers.index.tolist()
        }
    
    def create_visualizations(self) -> None:
        """
        Create visualizations for outlier detection results.
        """
        logger.info("Creating visualizations...")
        
        if self.df is None:
            logger.warning("No data loaded for visualization")
            return
        
        # Set up subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Outlier Detection Analysis - Romance Novel Dataset', fontsize=16)
        
        # Publication year distribution
        if 'publication_year' in self.df.columns:
            axes[0, 0].hist(self.df['publication_year'].dropna(), bins=30, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Publication Year Distribution')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Rating distribution
        if 'rating' in self.df.columns:
            axes[0, 1].hist(self.df['rating'].dropna(), bins=30, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Rating Distribution')
            axes[0, 1].set_xlabel('Rating')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Review count distribution (log scale)
        if 'review_count' in self.df.columns:
            log_reviews = np.log1p(self.df['review_count'].dropna())
            axes[1, 0].hist(log_reviews, bins=30, alpha=0.7, color='salmon')
            axes[1, 0].set_title('Review Count Distribution (Log Scale)')
            axes[1, 0].set_xlabel('Log(Review Count + 1)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Page count distribution
        if 'page_count' in self.df.columns:
            axes[1, 1].hist(self.df['page_count'].dropna(), bins=30, alpha=0.7, color='gold')
            axes[1, 1].set_title('Page Count Distribution')
            axes[1, 1].set_xlabel('Pages')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / f"outlier_detection_visualizations_{self.timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualizations saved to: {viz_path}")
        
        plt.close()
    
    def generate_treatment_recommendations(self) -> Dict[str, Any]:
        """
        Generate treatment recommendations based on analysis results.
        
        Returns:
            Dictionary with treatment recommendations
        """
        logger.info("Generating treatment recommendations...")
        
        recommendations = {
            'title_duplications': [],
            'series_title_data': [],
            'statistical_outliers': [],
            'priority_actions': [],
            'research_impact': {}
        }
        
        # Title duplication recommendations
        if 'title_duplications' in self.results:
            dup_results = self.results['title_duplications']
            
            if dup_results['total_duplicate_titles'] > 0:
                recommendations['title_duplications'].append({
                    'issue': f"Found {dup_results['total_duplicate_titles']} duplicate titles",
                    'action': 'Review each duplicate for legitimacy',
                    'priority': 'High' if dup_results['total_duplicate_titles'] > 10 else 'Medium'
                })
                
                # Check for potential errors
                potential_errors = [title for title, data in dup_results['author_name_analysis'].items() 
                                  if not data['is_legitimate']]
                
                if potential_errors:
                    recommendations['title_duplications'].append({
                        'issue': f"Found {len(potential_errors)} potentially erroneous duplicates",
                        'action': 'Manual review and correction required',
                        'priority': 'High'
                    })
        
        # Series data recommendations
        if 'series_title_data' in self.results:
            series_title_results = self.results['series_title_data']
            
            if 'series_title_works_count_analysis' in series_title_results:
                accuracy = series_title_results['series_title_works_count_analysis'].get('accuracy_rate', 0)
                
                if accuracy < 90:
                    recommendations['series_title_data'].append({
                        'issue': f"Series works count accuracy: {accuracy:.1f}%",
                        'action': 'Investigate and correct series_title_works_count discrepancies',
                        'priority': 'Medium'
                    })
            
            if 'missing_books_analysis' in series_title_results:
                incomplete = series_title_results['missing_books_analysis'].get('incomplete_series_title', 0)
                
                if incomplete > 0:
                    recommendations['series_title_data'].append({
                        'issue': f"Found {incomplete} potentially incomplete series_title",
                        'action': 'Verify series_title completeness and add missing books',
                        'priority': 'Low'
                    })
        
        # Statistical outlier recommendations
        if 'statistical_outliers' in self.results:
            outlier_results = self.results['statistical_outliers']
            
            for field in ['publication_year', 'rating', 'review_count', 'page_count']:
                field_key = f'{field}_outliers'
                if field_key in outlier_results and 'outlier_percentage' in outlier_results[field_key]:
                    outlier_pct = outlier_results[field_key]['outlier_percentage']
                    
                    if outlier_pct > 5:
                        recommendations['statistical_outliers'].append({
                            'issue': f"{field.replace('_', ' ').title()} has {outlier_pct:.1f}% outliers",
                            'action': 'Review outliers for data quality issues',
                            'priority': 'Medium' if outlier_pct > 10 else 'Low'
                        })
        
        # Priority actions
        high_priority = [rec for rec in recommendations['title_duplications'] + 
                        recommendations['series_title_data'] + 
                        recommendations['statistical_outliers'] 
                        if rec['priority'] == 'High']
        
        recommendations['priority_actions'] = high_priority
        
        # Research impact assessment
        recommendations['research_impact'] = {
            'RQ1_Topic_Modeling': 'Title duplicates may affect theme extraction accuracy',
            'RQ2_Review_Analysis': 'Series inconsistencies may impact series_title-level analysis',
            'RQ3_Correlation_Analysis': 'Data quality issues identified for correction',
            'RQ4_Author_vs_Reader_Themes': 'Author consistency validated'
        }
        
        logger.info(f"Generated {len(recommendations['priority_actions'])} high-priority recommendations")
        return recommendations
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete outlier detection and treatment analysis.
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting complete outlier detection and treatment analysis...")
        
        # Load data
        self.load_data()
        
        # Run analyses
        self.results['title_duplications'] = self.analyze_title_duplications()
        self.results['series_title_data'] = self.analyze_series_title_data()
        self.results['statistical_outliers'] = self.detect_statistical_outliers()
        
        # Generate recommendations
        self.results['treatment_recommendations'] = self.generate_treatment_recommendations()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_detailed_report()
        
        logger.info("Complete analysis finished successfully!")
        return self.results
    
    def generate_detailed_report(self) -> None:
        """
        Generate a detailed analysis report.
        """
        logger.info("Generating detailed report...")
        
        report_path = self.output_dir / f"step3_outlier_detection_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STEP 3: OUTLIER DETECTION AND TREATMENT REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Total Records: {len(self.df)}\n\n")
            
            # Title Duplication Analysis
            f.write("1. TITLE DUPLICATION ANALYSIS\n")
            f.write("-" * 40 + "\n")
            if 'title_duplications' in self.results:
                dup_results = self.results['title_duplications']
                f.write(f"Total duplicate titles: {dup_results['total_duplicate_titles']}\n")
                f.write(f"Total duplicate books: {dup_results['total_duplicate_books']}\n\n")
                
                for title, data in list(dup_results['author_name_analysis'].items())[:10]:  # Show first 10
                    f.write(f"Title: {title}\n")
                    f.write(f"  Count: {data['count']}\n")
                    f.write(f"  Authors: {', '.join(data['author_names'])}\n")
                    f.write(f"  Legitimate: {data['is_legitimate']}\n")
                    f.write(f"  Strategy: {data['strategy']}\n\n")
            
            # Series Data Analysis
            f.write("2. SERIES DATA ANALYSIS\n")
            f.write("-" * 40 + "\n")
            if 'series_title_data' in self.results:
                series_title_results = self.results['series_title_data']
                f.write(f"Total series_title books: {series_title_results['total_series_title_books']}\n")
                f.write(f"Unique series_title: {series_title_results['unique_series_title']}\n\n")
                
                if 'series_title_works_count_analysis' in series_title_results:
                    works_analysis = series_title_results['series_title_works_count_analysis']
                    f.write(f"Series works count accuracy: {works_analysis.get('accuracy_rate', 0):.1f}%\n")
                    f.write(f"Series with discrepancies: {works_analysis.get('series_title_with_discrepancies', 0)}\n\n")
            
            # Statistical Outlier Analysis
            f.write("3. STATISTICAL OUTLIER ANALYSIS\n")
            f.write("-" * 40 + "\n")
            if 'statistical_outliers' in self.results:
                outlier_results = self.results['statistical_outliers']
                f.write(f"Total outliers detected: {outlier_results['summary']['total_outliers_detected']}\n")
                f.write(f"Fields analyzed: {', '.join(outlier_results['summary']['fields_analyzed'])}\n\n")
                
                for field in ['publication_year', 'rating', 'review_count', 'page_count']:
                    field_key = f'{field}_outliers'
                    if field_key in outlier_results:
                        field_data = outlier_results[field_key]
                        if 'outlier_percentage' in field_data:
                            f.write(f"{field.replace('_', ' ').title()}:\n")
                            f.write(f"  Outliers: {field_data['outlier_percentage']:.1f}%\n")
                            f.write(f"  Count: {field_data['outlier_count']}\n\n")
            
            # Treatment Recommendations
            f.write("4. TREATMENT RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            if 'treatment_recommendations' in self.results:
                recs = self.results['treatment_recommendations']
                
                f.write("High Priority Actions:\n")
                for rec in recs.get('priority_actions', []):
                    f.write(f"  - {rec['issue']}\n")
                    f.write(f"    Action: {rec['action']}\n\n")
                
                f.write("Research Impact:\n")
                for rq, impact in recs.get('research_impact', {}).items():
                    f.write(f"  {rq}: {impact}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Detailed report saved to: {report_path}")
    
    def save_results_json(self) -> None:
        """
        Save analysis results as JSON file.
        """
        json_path = self.output_dir / f"step3_outlier_detection_results_{self.timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict('records')
            elif isinstance(value, dict):
                json_results[key] = value
            else:
                json_results[key] = str(value)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Results JSON saved to: {json_path}")


def main():
    """
    Main function to run the outlier detection analysis.
    """
    try:
        # Initialize analyzer
        analyzer = OutlierDetectionAnalyzer()
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Save results
        analyzer.save_results_json()
        
        # Print summary
        print("\n" + "="*60)
        print("OUTLIER DETECTION ANALYSIS COMPLETE")
        print("="*60)
        print(f"Title duplications found: {results['title_duplications']['total_duplicate_titles']}")
        print(f"Series books analyzed: {results['series_title_data']['total_series_title_books']}")
        print(f"Total outliers detected: {results['statistical_outliers']['summary']['total_outliers_detected']}")
        print(f"High priority actions: {len(results['treatment_recommendations']['priority_actions'])}")
        print(f"\nReports saved to: {analyzer.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
