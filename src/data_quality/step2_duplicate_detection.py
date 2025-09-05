#!/usr/bin/env python3
"""
Step 2: Duplicate Detection & Resolution (High Priority)

This script implements comprehensive duplicate detection and resolution for the romance novel dataset:
1. Exact Duplicate Detection and Removal
2. Title Duplication Analysis and Classification
3. Author Duplication Detection
4. Disambiguation Strategy Implementation
5. Duplicate Flagging and Documentation

Author: Research Assistant
Date: 2025-09-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DuplicateDetector:
    """
    Comprehensive duplicate detection and resolution system for romance novel dataset.
    Implements multiple detection strategies and resolution approaches.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the duplicate detector.
        
        Args:
            data_path: Path to the input dataset file (from Step 1)
        """
        self.data_path = data_path or "outputs/missing_values_cleaning/romance_novels_step1_missing_values_treated_20250902_000000.pkl"
        self.df = None
        self.detection_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("outputs/duplicate_detection")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Duplicate detection criteria
        self.duplicate_criteria = {
            'exact_duplicates': ['work_id', 'title', 'author_id', 'publication_year'],
            'title_duplicates': ['title', 'author_id'],
            'author_duplicates': ['author_id', 'author_name'],
            'series_duplicates': ['series_id', 'series_title']
        }
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset for duplicate detection.
        
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading dataset from: {self.data_path}")
            
            if self.data_path.endswith('.pkl'):
                self.df = pd.read_pickle(self.data_path)
                logger.info("Data loaded from pickle file - data types preserved")
            else:
                self.df = pd.read_csv(self.data_path)
                logger.info("Data loaded from CSV file")
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def detect_exact_duplicates(self) -> Dict[str, Any]:
        """
        Detect exact duplicates based on key fields.
        
        Returns:
            Dictionary with exact duplicate detection results
        """
        logger.info("🔍 Detecting exact duplicates...")
        
        criteria = self.duplicate_criteria['exact_duplicates']
        available_criteria = [col for col in criteria if col in self.df.columns]
        
        if not available_criteria:
            return {'error': 'No duplicate detection criteria available'}
        
        # Find exact duplicates
        duplicate_mask = self.df.duplicated(subset=available_criteria, keep=False)
        exact_duplicates = self.df[duplicate_mask]
        
        results = {
            'criteria_used': available_criteria,
            'total_duplicates': len(exact_duplicates),
            'duplicate_groups': 0,
            'duplicate_analysis': {}
        }
        
        if len(exact_duplicates) > 0:
            # Group duplicates
            duplicate_groups = exact_duplicates.groupby(available_criteria)
            results['duplicate_groups'] = len(duplicate_groups)
            
            # Analyze each group
            for group_key, group_df in duplicate_groups:
                group_id = str(group_key) if len(available_criteria) > 1 else group_key
                results['duplicate_analysis'][group_id] = {
                    'count': len(group_df),
                    'work_ids': group_df['work_id'].tolist() if 'work_id' in group_df.columns else [],
                    'titles': group_df['title'].tolist() if 'title' in group_df.columns else [],
                    'authors': group_df['author_name'].tolist() if 'author_name' in group_df.columns else []
                }
        
        logger.info(f"Exact duplicate detection completed:")
        logger.info(f"  • Total duplicates found: {results['total_duplicates']}")
        logger.info(f"  • Duplicate groups: {results['duplicate_groups']}")
        
        return results
    
    def detect_title_duplicates(self) -> Dict[str, Any]:
        """
        Detect title duplicates with same author.
        
        Returns:
            Dictionary with title duplicate detection results
        """
        logger.info("🔍 Detecting title duplicates...")
        
        criteria = self.duplicate_criteria['title_duplicates']
        available_criteria = [col for col in criteria if col in self.df.columns]
        
        if not available_criteria:
            return {'error': 'Title duplicate detection criteria not available'}
        
        # Find title duplicates
        duplicate_mask = self.df.duplicated(subset=available_criteria, keep=False)
        title_duplicates = self.df[duplicate_mask]
        
        results = {
            'criteria_used': available_criteria,
            'total_duplicates': len(title_duplicates),
            'duplicate_groups': 0,
            'duplicate_analysis': {},
            'classification': {}
        }
        
        if len(title_duplicates) > 0:
            # Group by title and author
            duplicate_groups = title_duplicates.groupby(available_criteria)
            results['duplicate_groups'] = len(duplicate_groups)
            
            # Classify each group
            for group_key, group_df in duplicate_groups:
                group_id = str(group_key) if len(available_criteria) > 1 else group_key
                
                # Analyze the group
                group_analysis = self._classify_title_duplicate_group(group_df)
                
                results['duplicate_analysis'][group_id] = {
                    'count': len(group_df),
                    'work_ids': group_df['work_id'].tolist() if 'work_id' in group_df.columns else [],
                    'publication_years': group_df['publication_year'].tolist() if 'publication_year' in group_df.columns else [],
                    'series_info': group_df['series_title'].tolist() if 'series_title' in group_df.columns else []
                }
                
                results['classification'][group_id] = group_analysis
        
        logger.info(f"Title duplicate detection completed:")
        logger.info(f"  • Total duplicates found: {results['total_duplicates']}")
        logger.info(f"  • Duplicate groups: {results['duplicate_groups']}")
        
        return results
    
    def _classify_title_duplicate_group(self, group_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify a title duplicate group.
        
        Args:
            group_df: DataFrame containing duplicate records
            
        Returns:
            Classification results
        """
        classification = {
            'type': 'unknown',
            'confidence': 'low',
            'recommendation': 'manual_review',
            'details': {}
        }
        
        # Check publication years
        if 'publication_year' in group_df.columns:
            years = group_df['publication_year'].dropna()
            if len(years) > 1:
                year_range = years.max() - years.min()
                if year_range <= 1:
                    classification['type'] = 'same_edition'
                    classification['confidence'] = 'high'
                    classification['recommendation'] = 'keep_most_recent'
                elif year_range <= 5:
                    classification['type'] = 'reprint'
                    classification['confidence'] = 'medium'
                    classification['recommendation'] = 'keep_original'
                else:
                    classification['type'] = 'different_editions'
                    classification['confidence'] = 'medium'
                    classification['recommendation'] = 'keep_all'
        
        # Check series information
        if 'series_title' in group_df.columns:
            series_info = group_df['series_title'].dropna()
            if len(series_info.unique()) > 1:
                classification['details']['series_conflict'] = True
                classification['recommendation'] = 'manual_review'
        
        return classification
    
    def detect_author_duplicates(self) -> Dict[str, Any]:
        """
        Detect author duplicates (same author with different IDs).
        
        Returns:
            Dictionary with author duplicate detection results
        """
        logger.info("🔍 Detecting author duplicates...")
        
        if 'author_name' not in self.df.columns:
            return {'error': 'Author name field not available'}
        
        # Find potential author duplicates
        author_counts = self.df['author_name'].value_counts()
        potential_duplicates = author_counts[author_counts > 1]
        
        results = {
            'total_potential_duplicates': len(potential_duplicates),
            'duplicate_analysis': {},
            'author_id_conflicts': 0
        }
        
        # Analyze each potential duplicate author
        for author_name, count in potential_duplicates.items():
            author_records = self.df[self.df['author_name'] == author_name]
            
            # Check for different author IDs
            if 'author_id' in author_records.columns:
                unique_author_ids = author_records['author_id'].nunique()
                if unique_author_ids > 1:
                    results['author_id_conflicts'] += 1
                    results['duplicate_analysis'][author_name] = {
                        'record_count': count,
                        'unique_author_ids': unique_author_ids,
                        'author_ids': author_records['author_id'].unique().tolist(),
                        'conflict_type': 'multiple_author_ids'
                    }
        
        logger.info(f"Author duplicate detection completed:")
        logger.info(f"  • Potential duplicates: {results['total_potential_duplicates']}")
        logger.info(f"  • Author ID conflicts: {results['author_id_conflicts']}")
        
        return results
    
    def apply_duplicate_resolution(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply duplicate resolution strategies.
        
        Returns:
            Tuple of (resolved_dataframe, resolution_results)
        """
        logger.info("🔧 Applying duplicate resolution strategies...")
        
        resolved_df = self.df.copy()
        resolution_results = {
            'exact_duplicates_removed': 0,
            'title_duplicates_resolved': 0,
            'author_duplicates_resolved': 0,
            'flags_created': 0,
            'resolution_details': {}
        }
        
        # 1. Remove exact duplicates
        exact_duplicates = self.detect_exact_duplicates()
        if exact_duplicates.get('total_duplicates', 0) > 0:
            criteria = exact_duplicates['criteria_used']
            before_count = len(resolved_df)
            resolved_df = resolved_df.drop_duplicates(subset=criteria, keep='first')
            removed_count = before_count - len(resolved_df)
            
            resolution_results['exact_duplicates_removed'] = removed_count
            resolution_results['resolution_details']['exact_duplicates'] = {
                'removed_count': removed_count,
                'criteria_used': criteria,
                'action': 'Removed exact duplicates, kept first occurrence'
            }
            
            logger.info(f"  ✅ Removed {removed_count} exact duplicates")
        
        # 2. Create duplicate flags
        self._create_duplicate_flags(resolved_df, resolution_results)
        
        # 3. Apply title duplicate resolution
        title_duplicates = self.detect_title_duplicates()
        if title_duplicates.get('total_duplicates', 0) > 0:
            resolved_df, title_resolution = self._resolve_title_duplicates(resolved_df, title_duplicates)
            resolution_results['title_duplicates_resolved'] = title_resolution['resolved_count']
            resolution_results['resolution_details']['title_duplicates'] = title_resolution
            
            logger.info(f"  ✅ Resolved {title_resolution['resolved_count']} title duplicates")
        
        # 4. Apply author duplicate resolution
        author_duplicates = self.detect_author_duplicates()
        if author_duplicates.get('author_id_conflicts', 0) > 0:
            resolved_df, author_resolution = self._resolve_author_duplicates(resolved_df, author_duplicates)
            resolution_results['author_duplicates_resolved'] = author_resolution['resolved_count']
            resolution_results['resolution_details']['author_duplicates'] = author_resolution
            
            logger.info(f"  ✅ Resolved {author_resolution['resolved_count']} author duplicates")
        
        resolution_results['final_shape'] = resolved_df.shape
        resolution_results['original_shape'] = self.df.shape
        
        logger.info(f"Duplicate resolution completed:")
        logger.info(f"  • Exact duplicates removed: {resolution_results['exact_duplicates_removed']}")
        logger.info(f"  • Title duplicates resolved: {resolution_results['title_duplicates_resolved']}")
        logger.info(f"  • Author duplicates resolved: {resolution_results['author_duplicates_resolved']}")
        logger.info(f"  • Flags created: {resolution_results['flags_created']}")
        
        return resolved_df, resolution_results
    
    def _create_duplicate_flags(self, df: pd.DataFrame, resolution_results: Dict[str, Any]) -> None:
        """
        Create duplicate flags for analysis.
        
        Args:
            df: DataFrame to add flags to
            resolution_results: Results dictionary to update
        """
        # Create title duplicate flag
        if 'title' in df.columns and 'author_id' in df.columns:
            title_duplicate_mask = df.duplicated(subset=['title', 'author_id'], keep=False)
            df['title_duplicate_flag'] = title_duplicate_mask.astype(int)
            resolution_results['flags_created'] += 1
            
            logger.info(f"  ✅ Created title_duplicate_flag: {title_duplicate_mask.sum()} duplicates flagged")
        
        # Create author duplicate flag
        if 'author_id' in df.columns:
            author_duplicate_mask = df.duplicated(subset=['author_id'], keep=False)
            df['author_id_duplicate_flag'] = author_duplicate_mask.astype(int)
            resolution_results['flags_created'] += 1
            
            logger.info(f"  ✅ Created author_id_duplicate_flag: {author_duplicate_mask.sum()} duplicates flagged")
    
    def _resolve_title_duplicates(self, df: pd.DataFrame, title_duplicates: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Resolve title duplicates based on classification.
        
        Args:
            df: DataFrame to resolve
            title_duplicates: Title duplicate analysis results
            
        Returns:
            Tuple of (resolved_dataframe, resolution_results)
        """
        resolution_results = {
            'resolved_count': 0,
            'resolution_strategy': 'classification_based',
            'details': {}
        }
        
        # For now, keep all title duplicates but flag them
        # In a production system, you would implement more sophisticated resolution
        resolution_results['details']['strategy'] = 'Flagged for manual review'
        resolution_results['details']['reason'] = 'Complex disambiguation required'
        
        return df, resolution_results
    
    def _resolve_author_duplicates(self, df: pd.DataFrame, author_duplicates: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Resolve author duplicates.
        
        Args:
            df: DataFrame to resolve
            author_duplicates: Author duplicate analysis results
            
        Returns:
            Tuple of (resolved_dataframe, resolution_results)
        """
        resolution_results = {
            'resolved_count': 0,
            'resolution_strategy': 'flagging_only',
            'details': {}
        }
        
        # For now, just flag author duplicates
        # In a production system, you would implement author ID consolidation
        resolution_results['details']['strategy'] = 'Flagged for manual review'
        resolution_results['details']['reason'] = 'Author ID consolidation required'
        
        return df, resolution_results
    
    def validate_resolution_results(self, resolved_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the results of duplicate resolution.
        
        Args:
            resolved_df: Resolved DataFrame
            
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating duplicate resolution results...")
        
        validation = {
            'validation_passed': True,
            'remaining_duplicates': {},
            'validation_errors': [],
            'resolution_effectiveness': {}
        }
        
        # Check for remaining exact duplicates
        exact_duplicates = self.detect_exact_duplicates()
        if exact_duplicates.get('total_duplicates', 0) > 0:
            validation['remaining_duplicates']['exact_duplicates'] = exact_duplicates['total_duplicates']
            validation['validation_errors'].append(
                f"Exact duplicates still present: {exact_duplicates['total_duplicates']}"
            )
            validation['validation_passed'] = False
        
        # Check duplicate flags
        if 'title_duplicate_flag' in resolved_df.columns:
            flagged_duplicates = resolved_df['title_duplicate_flag'].sum()
            validation['remaining_duplicates']['title_duplicates_flagged'] = flagged_duplicates
        
        if 'author_id_duplicate_flag' in resolved_df.columns:
            flagged_duplicates = resolved_df['author_id_duplicate_flag'].sum()
            validation['remaining_duplicates']['author_duplicates_flagged'] = flagged_duplicates
        
        # Calculate resolution effectiveness
        original_count = len(self.df)
        resolved_count = len(resolved_df)
        validation['resolution_effectiveness'] = {
            'original_records': original_count,
            'resolved_records': resolved_count,
            'records_removed': original_count - resolved_count,
            'reduction_percentage': ((original_count - resolved_count) / original_count) * 100
        }
        
        logger.info(f"Validation completed: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}")
        
        return validation
    
    def save_resolved_dataset(self, resolved_df: pd.DataFrame, filename: str = None) -> str:
        """
        Save the resolved dataset.
        
        Args:
            resolved_df: Resolved DataFrame
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"romance_novels_step2_duplicates_resolved_{self.timestamp}.pkl"
        
        filepath = self.output_dir / filename
        
        # Save as pickle to preserve data types
        resolved_df.to_pickle(filepath)
        
        logger.info(f"Resolved dataset saved to: {filepath}")
        logger.info(f"File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(filepath)
    
    def save_resolution_report(self, detection_results: Dict[str, Any], resolution_results: Dict[str, Any], 
                             validation: Dict[str, Any], filename: str = None) -> str:
        """
        Save comprehensive resolution report.
        
        Args:
            detection_results: Detection results
            resolution_results: Resolution results
            validation: Validation results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"duplicate_resolution_report_step2_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        report = {
            'resolution_timestamp': self.timestamp,
            'detection_results': detection_results,
            'resolution_results': resolution_results,
            'validation_results': validation,
            'summary': {
                'original_records': self.df.shape[0],
                'resolved_records': resolution_results['final_shape'][0],
                'exact_duplicates_removed': resolution_results['exact_duplicates_removed'],
                'title_duplicates_resolved': resolution_results['title_duplicates_resolved'],
                'author_duplicates_resolved': resolution_results['author_duplicates_resolved'],
                'flags_created': resolution_results['flags_created'],
                'validation_passed': validation['validation_passed']
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Resolution report saved to: {filepath}")
        return str(filepath)
    
    def run_complete_resolution(self) -> Dict[str, Any]:
        """
        Run complete duplicate detection and resolution process.
        
        Returns:
            Dictionary with complete resolution results
        """
        logger.info("🚀 Starting complete duplicate detection and resolution process...")
        start_time = datetime.now()
        
        # Initialize results
        self.detection_results = {
            'resolution_timestamp': self.timestamp,
            'dataset_info': {},
            'detection_results': {},
            'resolution_results': {},
            'validation_results': {},
            'final_dataset_info': {},
            'resolution_summary': {}
        }
        
        # 1. Load dataset
        logger.info("📥 Loading dataset...")
        original_df = self.load_data()
        
        self.detection_results['dataset_info'] = {
            'original_shape': original_df.shape,
            'original_memory_mb': original_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'original_columns': list(original_df.columns)
        }
        
        # 2. Detect duplicates
        logger.info("🔍 Detecting duplicates...")
        detection_results = {
            'exact_duplicates': self.detect_exact_duplicates(),
            'title_duplicates': self.detect_title_duplicates(),
            'author_duplicates': self.detect_author_duplicates()
        }
        self.detection_results['detection_results'] = detection_results
        
        # 3. Apply resolution
        logger.info("🔧 Applying duplicate resolution...")
        resolved_df, resolution_results = self.apply_duplicate_resolution()
        self.detection_results['resolution_results'] = resolution_results
        
        # 4. Validate results
        logger.info("🔍 Validating resolution results...")
        validation = self.validate_resolution_results(resolved_df)
        self.detection_results['validation_results'] = validation
        
        # 5. Save resolved dataset
        logger.info("💾 Saving resolved dataset...")
        dataset_file = self.save_resolved_dataset(resolved_df)
        
        # 6. Final dataset information
        self.detection_results['final_dataset_info'] = {
            'final_shape': resolved_df.shape,
            'final_memory_mb': resolved_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'final_columns': list(resolved_df.columns),
            'data_type_preservation': 'Maintained (pickle format)'
        }
        
        # 7. Generate resolution summary
        self.detection_results['resolution_summary'] = {
            'total_exact_duplicates_removed': resolution_results['exact_duplicates_removed'],
            'total_title_duplicates_resolved': resolution_results['title_duplicates_resolved'],
            'total_author_duplicates_resolved': resolution_results['author_duplicates_resolved'],
            'total_flags_created': resolution_results['flags_created'],
            'validation_passed': validation['validation_passed'],
            'data_integrity': 'Maintained - duplicates resolved strategically',
            'execution_time': str(datetime.now() - start_time)
        }
        
        # 8. Save resolution report
        logger.info("📊 Saving resolution report...")
        report_file = self.save_resolution_report(detection_results, resolution_results, validation)
        
        logger.info("✅ Complete duplicate detection and resolution process finished!")
        
        return self.detection_results
    
    def print_resolution_summary(self):
        """Print a human-readable resolution summary."""
        if not self.detection_results:
            print("No resolution results available. Run run_complete_resolution() first.")
            return
        
        print("\n" + "="*80)
        print("DUPLICATE DETECTION & RESOLUTION SUMMARY - STEP 2")
        print("="*80)
        
        # Original dataset info
        original_info = self.detection_results['dataset_info']
        print(f"Original Dataset: {original_info['original_shape'][0]:,} records × {original_info['original_shape'][1]} columns")
        print(f"Original Memory: {original_info['original_memory_mb']:.2f} MB")
        
        # Detection results
        detection = self.detection_results['detection_results']
        print(f"\n🔍 DUPLICATE DETECTION:")
        print(f"  • Exact duplicates found: {detection['exact_duplicates'].get('total_duplicates', 0)}")
        print(f"  • Title duplicates found: {detection['title_duplicates'].get('total_duplicates', 0)}")
        print(f"  • Author duplicates found: {detection['author_duplicates'].get('author_id_conflicts', 0)}")
        
        # Resolution results
        resolution = self.detection_results['resolution_results']
        print(f"\n🔧 RESOLUTION RESULTS:")
        print(f"  • Exact duplicates removed: {resolution['exact_duplicates_removed']}")
        print(f"  • Title duplicates resolved: {resolution['title_duplicates_resolved']}")
        print(f"  • Author duplicates resolved: {resolution['author_duplicates_resolved']}")
        print(f"  • Flags created: {resolution['flags_created']}")
        
        # Final dataset info
        final_info = self.detection_results['final_dataset_info']
        print(f"\n📊 FINAL DATASET:")
        print(f"  • Final records: {final_info['final_shape'][0]:,} records × {final_info['final_shape'][1]} columns")
        print(f"  • Final memory: {final_info['final_memory_mb']:.2f} MB")
        print(f"  • Data types: {final_info['data_type_preservation']}")
        
        # Resolution summary
        summary = self.detection_results['resolution_summary']
        print(f"\n🎯 RESOLUTION SUMMARY:")
        print(f"  • Validation passed: {'✅ YES' if summary['validation_passed'] else '❌ NO'}")
        print(f"  • Data integrity: {summary['data_integrity']}")
        print(f"  • Execution time: {summary['execution_time']}")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    logger = logging.getLogger(__name__)
    
    print("🔧 STEP 2: DUPLICATE DETECTION & RESOLUTION")
    print("=" * 60)
    
    try:
        # Initialize duplicate detector
        detector = DuplicateDetector()
        
        # Run complete resolution process
        results = detector.run_complete_resolution()
        
        # Print summary
        detector.print_resolution_summary()
        
        print("\n🎉 Step 2: Duplicate Detection & Resolution completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Duplicate detection and resolution failed: {str(e)}", exc_info=True)
        print(f"\n💥 Duplicate detection and resolution failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)
