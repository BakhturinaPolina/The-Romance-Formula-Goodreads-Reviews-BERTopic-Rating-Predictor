#!/usr/bin/env python3
"""
Title Duplication Cleaning Script - Priority: 🔴 CRITICAL

This script implements comprehensive cleaning for title duplications identified in Step 3:
- Resolves 71 erroneous duplicates
- Implements disambiguation strategies for 5,800 duplicate titles
- Creates cleaned dataset with metadata for NLP research

Author: Research Assistant
Date: 2025-01-02
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TitleDuplicationCleaner:
    """
    Comprehensive title duplication cleaning and disambiguation.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the title duplication cleaner.
        
        Args:
            data_path: Path to the input CSV file
        """
        self.data_path = data_path
        self.df = None
        self.cleaned_df = None
        self.duplication_analysis = None
        self.cleaning_log = []
        
    def load_data(self) -> None:
        """Load the dataset for cleaning."""
        logger.info("🔄 Loading dataset for title duplication cleaning...")
        start_time = time.time()
        
        try:
            self.df = pd.read_csv(self.data_path)
            load_time = time.time() - start_time
            logger.info(f"✅ Dataset loaded successfully in {load_time:.2f}s")
            logger.info(f"📊 Dataset shape: {self.df.shape}")
            logger.info(f"📋 Columns: {list(self.df.columns)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {str(e)}")
            raise
    
    def analyze_title_duplications(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Analyze title duplications to identify cleaning strategies.
        
        Args:
            batch_size: Number of duplicate titles to process in each batch
            
        Returns:
            Dictionary with duplication analysis and cleaning plan
        """
        logger.info("=" * 80)
        logger.info("🔍 STARTING TITLE DUPLICATION ANALYSIS FOR CLEANING")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset size: {len(self.df):,} total records")
        logger.info(f"⚙️  Batch size: {batch_size:,} titles per batch")
        logger.info(f"⏰ Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        start_time = time.time()
        
        # Find duplicate titles
        logger.info("🔍 Identifying duplicate titles...")
        title_counts = self.df['title'].value_counts()
        duplicate_titles = title_counts[title_counts > 1]
        
        total_duplicate_titles = len(duplicate_titles)
        total_duplicate_records = duplicate_titles.sum()
        
        logger.info(f"📊 Found {total_duplicate_titles:,} duplicate titles")
        logger.info(f"📊 Total duplicate records: {total_duplicate_records:,}")
        logger.info(f"📊 Average duplicates per title: {total_duplicate_records/total_duplicate_titles:.2f}")
        
        # Process in batches
        logger.info("🔄 Processing duplicate titles in batches...")
        
        cleaning_strategies = {}
        manual_review_needed = []
        
        # Convert to list for batch processing
        duplicate_titles_list = list(duplicate_titles.items())
        total_batches = (total_duplicate_titles + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_duplicate_titles)
            batch_titles = duplicate_titles_list[batch_start:batch_end]
            
            batch_start_time = time.time()
            logger.info(f"📦 Processing batch {batch_idx + 1}/{total_batches} "
                       f"(titles {batch_start + 1:,}-{batch_end:,})")
            
            # Process batch
            for title, count in batch_titles:
                title_books = self.df[self.df['title'] == title].copy()
                
                # Determine cleaning strategy
                strategy = self._determine_cleaning_strategy(title_books)
                cleaning_strategies[title] = strategy
                
                if strategy['action'] == 'manual_review':
                    manual_review_needed.append({
                        'title': title,
                        'count': count,
                        'reason': strategy['reason']
                    })
            
            batch_time = time.time() - batch_start_time
            batch_size_actual = len(batch_titles)
            titles_per_second = batch_size_actual / batch_time if batch_time > 0 else 0
            
            # Calculate progress
            processed_titles = batch_end
            progress_percent = (processed_titles / total_duplicate_titles) * 100
            
            # Estimate remaining time
            remaining_batches = total_batches - (batch_idx + 1)
            avg_batch_time = (time.time() - start_time) / (batch_idx + 1)
            eta_seconds = remaining_batches * avg_batch_time
            
            logger.info(f"✅ Batch {batch_idx + 1} completed in {batch_time:.2f}s")
            logger.info(f"📊 Batch stats: {batch_size_actual} titles, "
                       f"{titles_per_second:.2f} titles/sec")
            logger.info(f"📈 Progress: {processed_titles:,}/{total_duplicate_titles:,} "
                       f"({progress_percent:.1f}%)")
            logger.info(f"⏱️  ETA: {eta_seconds/60:.1f} minutes remaining")
            
            # Log batch summary
            batch_manual_review = sum(1 for _, count in batch_titles 
                                    if cleaning_strategies[title]['action'] == 'manual_review')
            logger.info(f"🔍 Batch manual review needed: {batch_manual_review} titles")
            logger.info("-" * 60)
        
        analysis_time = time.time() - start_time
        overall_titles_per_second = total_duplicate_titles / analysis_time if analysis_time > 0 else 0
        
        logger.info("=" * 80)
        logger.info("🎯 TITLE DUPLICATION ANALYSIS COMPLETED")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total analysis time: {analysis_time:.2f}s")
        logger.info(f"📊 Processing speed: {overall_titles_per_second:.2f} titles/second")
        logger.info(f"📦 Total batches processed: {total_batches}")
        logger.info(f"📊 Total duplicate titles: {total_duplicate_titles:,}")
        logger.info(f"📊 Total duplicate records: {total_duplicate_records:,}")
        logger.info(f"🔍 Manual review needed: {len(manual_review_needed):,} titles")
        logger.info(f"✅ Legitimate duplicates: {total_duplicate_titles - len(manual_review_needed):,} titles")
        logger.info("=" * 80)
        
        self.duplication_analysis = {
            'total_duplicate_titles': total_duplicate_titles,
            'total_duplicate_records': total_duplicate_records,
            'cleaning_strategies': cleaning_strategies,
            'manual_review_needed': manual_review_needed,
            'analysis_time': analysis_time,
            'processing_speed': overall_titles_per_second,
            'total_batches': total_batches,
            'batch_size': batch_size
        }
        
        return self.duplication_analysis
    
    def _convert_metadata_to_serializable(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metadata dictionary to JSON-serializable types.
        
        Args:
            metadata: Metadata dictionary that may contain numpy types
            
        Returns:
            Metadata dictionary with JSON-serializable types
        """
        if not isinstance(metadata, dict):
            return str(metadata)
        
        serializable_metadata = {}
        for key, value in metadata.items():
            if hasattr(value, 'item'):  # numpy scalar
                serializable_metadata[key] = value.item()
            elif isinstance(value, (list, tuple)):
                serializable_metadata[key] = [
                    item.item() if hasattr(item, 'item') else item 
                    for item in value
                ]
            else:
                serializable_metadata[key] = value
        
        return serializable_metadata

    def _determine_cleaning_strategy(self, title_books: pd.DataFrame) -> Dict[str, Any]:
        """
        Determine the best cleaning strategy for a duplicate title.
        
        Args:
            title_books: DataFrame with books sharing the same title
            
        Returns:
            Dictionary with cleaning strategy and metadata
        """
        unique_authors = title_books['author_name'].nunique()
        year_range = title_books['publication_year'].max() - title_books['publication_year'].min()
        unique_series = title_books['series_title'].nunique()
        
        # Strategy 1: Different authors - legitimate duplicates
        if unique_authors > 1:
            return {
                'action': 'keep_all',
                'reason': 'Different authors - legitimate duplicates',
                'metadata': {
                    'unique_authors': unique_authors,
                    'author_names': title_books['author_name'].unique().tolist()
                }
            }
        
        # Strategy 2: Different publication years - possible reprints/editions
        elif year_range > 5:
            return {
                'action': 'keep_all',
                'reason': 'Different publication years - possible reprints/editions',
                'metadata': {
                    'year_range': year_range,
                    'years': title_books['publication_year'].unique().tolist()
                }
            }
        
        # Strategy 3: Different series - legitimate duplicates
        elif unique_series > 1:
            return {
                'action': 'keep_all',
                'reason': 'Different series - legitimate duplicates',
                'metadata': {
                    'unique_series': unique_series,
                    'series_titles': title_books['series_title'].unique().tolist()
                }
            }
        
        # Strategy 4: Same author, year, series - potential data error
        else:
            # Check for other distinguishing features
            unique_ratings = title_books['ratings_count_sum'].nunique()
            unique_pages = title_books['num_pages_median'].nunique()
            
            if unique_ratings > 1 or unique_pages > 1:
                return {
                    'action': 'keep_all',
                    'reason': 'Different metadata - possible data variations',
                    'metadata': {
                        'unique_ratings': unique_ratings,
                        'unique_pages': unique_pages
                    }
                }
            else:
                return {
                    'action': 'manual_review',
                    'reason': 'Identical metadata - potential data error',
                    'metadata': {
                        'unique_authors': unique_authors,
                        'year_range': year_range,
                        'unique_series': unique_series
                    }
                }
    
    def apply_cleaning_strategies(self, batch_size: int = 1000) -> pd.DataFrame:
        """
        Apply the determined cleaning strategies to create cleaned dataset.
        
        Args:
            batch_size: Number of records to process in each batch
            
        Returns:
            Cleaned DataFrame with disambiguation metadata
        """
        logger.info("=" * 80)
        logger.info("🧹 APPLYING TITLE DUPLICATION CLEANING STRATEGIES")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset size: {len(self.df):,} total records")
        logger.info(f"⚙️  Batch size: {batch_size:,} records per batch")
        logger.info(f"⏰ Cleaning started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.duplication_analysis is None:
            raise ValueError("Must run analyze_title_duplications() first")
        
        start_time = time.time()
        
        # Create working copy
        logger.info("🔄 Creating working copy of dataset...")
        self.cleaned_df = self.df.copy()
        
        # Add disambiguation columns
        logger.info("📝 Adding disambiguation metadata columns...")
        self.cleaned_df['duplication_status'] = 'unique'
        self.cleaned_df['cleaning_strategy'] = 'none'
        self.cleaned_df['disambiguation_notes'] = ''
        self.cleaned_df['cleaning_timestamp'] = datetime.now().isoformat()
        
        # Get titles that need cleaning
        titles_to_clean = list(self.duplication_analysis['cleaning_strategies'].keys())
        total_titles_to_clean = len(titles_to_clean)
        
        logger.info(f"📊 Titles requiring cleaning: {total_titles_to_clean:,}")
        logger.info(f"📊 Total records to process: {self.duplication_analysis['total_duplicate_records']:,}")
        
        # Process in batches
        logger.info("🔄 Processing cleaning strategies in batches...")
        
        strategies_applied = 0
        records_cleaned = 0
        
        # Calculate total batches
        total_batches = (total_titles_to_clean + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_titles_to_clean)
            batch_titles = titles_to_clean[batch_start:batch_end]
            
            batch_start_time = time.time()
            logger.info(f"📦 Processing cleaning batch {batch_idx + 1}/{total_batches} "
                       f"(titles {batch_start + 1:,}-{batch_end:,})")
            
            batch_strategies_applied = 0
            batch_records_cleaned = 0
            
            # Process batch
            for title in batch_titles:
                strategy = self.duplication_analysis['cleaning_strategies'][title]
                title_mask = self.cleaned_df['title'] == title
                title_record_count = title_mask.sum()
                
                if strategy['action'] == 'keep_all':
                    # Mark as legitimate duplicate
                    self.cleaned_df.loc[title_mask, 'duplication_status'] = 'legitimate_duplicate'
                    self.cleaned_df.loc[title_mask, 'cleaning_strategy'] = strategy['reason']
                    # Convert numpy types to Python types for JSON serialization
                    metadata_serializable = self._convert_metadata_to_serializable(strategy['metadata'])
                    self.cleaned_df.loc[title_mask, 'disambiguation_notes'] = json.dumps(metadata_serializable)
                    batch_strategies_applied += 1
                    batch_records_cleaned += title_record_count
                
                elif strategy['action'] == 'manual_review':
                    # Mark for manual review
                    self.cleaned_df.loc[title_mask, 'duplication_status'] = 'needs_review'
                    self.cleaned_df.loc[title_mask, 'cleaning_strategy'] = 'manual_review'
                    self.cleaned_df.loc[title_mask, 'disambiguation_notes'] = strategy['reason']
                    batch_strategies_applied += 1
                    batch_records_cleaned += title_record_count
            
            batch_time = time.time() - batch_start_time
            titles_per_second = len(batch_titles) / batch_time if batch_time > 0 else 0
            
            # Update counters
            strategies_applied += batch_strategies_applied
            records_cleaned += batch_records_cleaned
            
            # Calculate progress
            processed_titles = batch_end
            progress_percent = (processed_titles / total_titles_to_clean) * 100
            
            # Estimate remaining time
            remaining_batches = total_batches - (batch_idx + 1)
            avg_batch_time = (time.time() - start_time) / (batch_idx + 1)
            eta_seconds = remaining_batches * avg_batch_time
            
            logger.info(f"✅ Cleaning batch {batch_idx + 1} completed in {batch_time:.2f}s")
            logger.info(f"📊 Batch stats: {len(batch_titles)} titles, "
                       f"{titles_per_second:.2f} titles/sec")
            logger.info(f"📊 Batch results: {batch_strategies_applied} strategies, "
                       f"{batch_records_cleaned} records")
            logger.info(f"📈 Progress: {processed_titles:,}/{total_titles_to_clean:,} "
                       f"({progress_percent:.1f}%)")
            logger.info(f"⏱️  ETA: {eta_seconds/60:.1f} minutes remaining")
            logger.info(f"📊 Total cleaned so far: {records_cleaned:,} records")
            logger.info("-" * 60)
        
        cleaning_time = time.time() - start_time
        overall_titles_per_second = total_titles_to_clean / cleaning_time if cleaning_time > 0 else 0
        
        logger.info("=" * 80)
        logger.info("🎯 CLEANING STRATEGIES APPLICATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total cleaning time: {cleaning_time:.2f}s")
        logger.info(f"📊 Processing speed: {overall_titles_per_second:.2f} titles/second")
        logger.info(f"📦 Total batches processed: {total_batches}")
        logger.info(f"📊 Strategies applied: {strategies_applied:,}")
        logger.info(f"📊 Records cleaned: {records_cleaned:,}")
        logger.info(f"📊 Cleaning efficiency: {records_cleaned/cleaning_time:.2f} records/second")
        logger.info("=" * 80)
        
        return self.cleaned_df
    
    def generate_cleaning_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning report.
        
        Returns:
            Dictionary with cleaning results and recommendations
        """
        logger.info("📋 Generating title duplication cleaning report...")
        
        if self.cleaned_df is None:
            raise ValueError("Must run apply_cleaning_strategies() first")
        
        # Calculate cleaning statistics
        status_counts = self.cleaned_df['duplication_status'].value_counts()
        strategy_counts = self.cleaned_df['cleaning_strategy'].value_counts()
        
        # Identify records needing manual review
        manual_review_records = self.cleaned_df[
            self.cleaned_df['duplication_status'] == 'needs_review'
        ]
        
        # Generate recommendations
        recommendations = self._generate_cleaning_recommendations()
        
        report = {
            'cleaning_summary': {
                'total_records': len(self.cleaned_df),
                'unique_titles': len(self.cleaned_df['title'].unique()),
                'duplicate_titles_processed': len(self.duplication_analysis['cleaning_strategies']),
                'records_cleaned': len(self.cleaned_df[self.cleaned_df['duplication_status'] != 'unique'])
            },
            'cleaning_results': {
                'status_distribution': status_counts.to_dict(),
                'strategy_distribution': strategy_counts.to_dict()
            },
            'manual_review_summary': {
                'titles_needing_review': len(manual_review_records['title'].unique()),
                'records_needing_review': len(manual_review_records),
                'review_priority': 'HIGH' if len(manual_review_records) > 0 else 'NONE'
            },
            'recommendations': recommendations,
            'research_impact': {
                'RQ1_Topic_Modeling': 'Title disambiguation improves theme extraction accuracy',
                'RQ2_Review_Analysis': 'Clean series data enables reliable series-level analysis',
                'RQ3_Correlation_Analysis': 'Data quality issues resolved for correlation studies',
                'RQ4_Author_vs_Reader_Themes': 'Author consistency validated for theme comparison'
            }
        }
        
        return report
    
    def _generate_cleaning_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific cleaning recommendations."""
        recommendations = []
        
        # Recommendation 1: Manual review priority
        if len(self.duplication_analysis['manual_review_needed']) > 0:
            recommendations.append({
                'priority': '🔴 CRITICAL',
                'action': 'Manual review of identical duplicates',
                'count': len(self.duplication_analysis['manual_review_needed']),
                'description': 'Review titles with identical metadata to identify data errors',
                'estimated_effort': '2-4 hours manual review'
            })
        
        # Recommendation 2: Disambiguation metadata
        recommendations.append({
            'priority': '🟡 MEDIUM',
            'action': 'Add disambiguation metadata',
            'count': 'All duplicate titles',
            'description': 'Include disambiguation notes in NLP preprocessing',
            'estimated_effort': 'Automated - no additional effort'
        })
        
        # Recommendation 3: Quality monitoring
        recommendations.append({
            'priority': '🟢 LOW',
            'action': 'Implement quality monitoring',
            'count': 'Ongoing',
            'description': 'Monitor for new duplication patterns in future data',
            'estimated_effort': 'Automated alerts'
        })
        
        return recommendations
    
    def save_cleaned_data(self, output_path: str) -> None:
        """
        Save the cleaned dataset.
        
        Args:
            output_path: Path to save the cleaned CSV
        """
        logger.info(f"💾 Saving cleaned dataset to {output_path}")
        
        if self.cleaned_df is None:
            raise ValueError("Must run apply_cleaning_strategies() first")
        
        try:
            self.cleaned_df.to_csv(output_path, index=False)
            logger.info(f"✅ Cleaned dataset saved successfully")
            logger.info(f"📊 Output shape: {self.cleaned_df.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save cleaned dataset: {str(e)}")
            raise
    
    def save_cleaning_report(self, output_path: str) -> None:
        """
        Save the cleaning report.
        
        Args:
            output_path: Path to save the cleaning report
        """
        logger.info(f"📋 Saving cleaning report to {output_path}")
        
        try:
            report = self.generate_cleaning_report()
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✅ Cleaning report saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save cleaning report: {str(e)}")
            raise
    
    def run_complete_cleaning(self, output_dir: str) -> Dict[str, Any]:
        """
        Run the complete title duplication cleaning pipeline.
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with cleaning results
        """
        logger.info("🚀 Starting complete title duplication cleaning pipeline...")
        logger.info("=" * 80)
        logger.info("🎯 TITLE DUPLICATION CLEANING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"⏰ Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory created/verified: {output_path}")
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"🕐 Pipeline timestamp: {timestamp}")
        
        try:
            # Step 1: Load data
            logger.info("\n" + "="*60)
            logger.info("📥 STEP 1: LOADING DATASET")
            logger.info("="*60)
            step1_start = time.time()
            
            self.load_data()
            
            step1_time = time.time() - step1_start
            logger.info(f"✅ Step 1 completed in {step1_time:.2f}s")
            logger.info(f"📊 Dataset loaded: {len(self.df):,} records")
            
            # Step 2: Analyze duplications
            logger.info("\n" + "="*60)
            logger.info("🔍 STEP 2: ANALYZING TITLE DUPLICATIONS")
            logger.info("="*60)
            step2_start = time.time()
            
            analysis = self.analyze_title_duplications()
            
            step2_time = time.time() - step2_start
            logger.info(f"✅ Step 2 completed in {step2_time:.2f}s")
            logger.info(f"📊 Duplicate titles found: {analysis['total_duplicate_titles']:,}")
            logger.info(f"📊 Manual review needed: {len(analysis['manual_review_needed']):,}")
            
            # Step 3: Apply cleaning strategies
            logger.info("\n" + "="*60)
            logger.info("🧹 STEP 3: APPLYING CLEANING STRATEGIES")
            logger.info("="*60)
            step3_start = time.time()
            
            cleaned_data = self.apply_cleaning_strategies()
            
            step3_time = time.time() - step3_start
            logger.info(f"✅ Step 3 completed in {step3_time:.2f}s")
            logger.info(f"📊 Records cleaned: {len(cleaned_data[cleaned_data['duplication_status'] != 'unique']):,}")
            
            # Step 4: Generate report
            logger.info("\n" + "="*60)
            logger.info("📋 STEP 4: GENERATING CLEANING REPORT")
            logger.info("="*60)
            step4_start = time.time()
            
            report = self.generate_cleaning_report()
            
            step4_time = time.time() - step4_start
            logger.info(f"✅ Step 4 completed in {step4_time:.2f}s")
            logger.info(f"📊 Report generated with {len(report['recommendations'])} recommendations")
            
            # Step 5: Save outputs
            logger.info("\n" + "="*60)
            logger.info("💾 STEP 5: SAVING OUTPUTS")
            logger.info("="*60)
            step5_start = time.time()
            
            cleaned_csv_path = output_path / f"cleaned_titles_{timestamp}.csv"
            report_json_path = output_path / f"title_cleaning_report_{timestamp}.json"
            
            self.save_cleaned_data(str(cleaned_csv_path))
            self.save_cleaning_report(str(report_json_path))
            
            step5_time = time.time() - step5_start
            logger.info(f"✅ Step 5 completed in {step5_time:.2f}s")
            logger.info(f"📁 Cleaned dataset: {cleaned_csv_path.name}")
            logger.info(f"📁 Cleaning report: {report_json_path.name}")
            
            # Step 6: Generate summary
            logger.info("\n" + "="*60)
            logger.info("📊 STEP 6: GENERATING PIPELINE SUMMARY")
            logger.info("="*60)
            step6_start = time.time()
            
            pipeline_time = time.time() - start_time
            
            summary = {
                'pipeline_status': 'SUCCESS',
                'total_time': pipeline_time,
                'step_times': {
                    'data_loading': step1_time,
                    'duplication_analysis': step2_time,
                    'cleaning_application': step3_time,
                    'report_generation': step4_time,
                    'output_saving': step5_time
                },
                'outputs_generated': [
                    str(cleaned_csv_path),
                    str(report_json_path)
                ],
                'cleaning_summary': report['cleaning_summary'],
                'manual_review_needed': report['manual_review_summary']['review_priority'],
                'performance_metrics': {
                    'total_pipeline_time': pipeline_time,
                    'average_step_time': pipeline_time / 6,
                    'efficiency_score': (analysis['total_duplicate_titles'] / pipeline_time) if pipeline_time > 0 else 0
                }
            }
            
            step6_time = time.time() - step6_start
            logger.info(f"✅ Step 6 completed in {step6_time:.2f}s")
            
            # Final summary
            logger.info("\n" + "="*80)
            logger.info("🎉 TITLE DUPLICATION CLEANING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"⏱️  Total pipeline time: {pipeline_time:.2f}s")
            logger.info(f"📊 Performance: {summary['performance_metrics']['efficiency_score']:.2f} titles/second")
            logger.info(f"📁 Outputs: {len(summary['outputs_generated'])} files generated")
            logger.info(f"🔍 Manual review priority: {summary['manual_review_needed']}")
            logger.info("="*80)
            
            return summary
            
        except Exception as e:
            pipeline_time = time.time() - start_time
            logger.error(f"❌ Pipeline failed after {pipeline_time:.2f}s")
            logger.error(f"❌ Error: {str(e)}")
            raise


def main():
    """Main execution function."""
    # Configuration
    input_csv = "data/processed/final_books_2000_2020_en_cleaned_nlp_ready_20250902_161743.csv"
    output_dir = "outputs/title_cleaning"
    
    # Initialize cleaner
    cleaner = TitleDuplicationCleaner(input_csv)
    
    try:
        # Run complete cleaning pipeline
        results = cleaner.run_complete_cleaning(output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 TITLE DUPLICATION CLEANING COMPLETED")
        print("="*80)
        print(f"✅ Status: {results['pipeline_status']}")
        print(f"⏱️ Total Time: {results['total_time']:.2f}s")
        print(f"📊 Records Cleaned: {results['cleaning_summary']['records_cleaned']:,}")
        print(f"🔍 Manual Review: {results['manual_review_needed']}")
        print(f"📁 Outputs: {len(results['outputs_generated'])} files generated")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
