#!/usr/bin/env python3
"""
Dataset Validation Script for Integrated Romance Novels Dataset

This script validates the quality of the integrated dataset to:
1. Verify data integrity after integration
2. Check sample records for correctness
3. Validate cleaning metadata was properly applied
4. Assess overall dataset quality for NLP readiness
5. Generate validation report with recommendations

Author: AI Assistant
Date: 2025-09-02
"""

import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/dataset_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validates the integrated dataset quality and integrity."""
    
    def __init__(self, 
                 integrated_dataset_path: str = "data/processed/integrated_romance_novels_nlp_ready_*.csv",
                 output_dir: str = "outputs/validation"):
        """Initialize the dataset validator."""
        self.dataset_path_pattern = integrated_dataset_path
        self.output_dir = Path(output_dir)
        self.integrated_dataset = None
        self.validation_results = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_integrated_dataset(self):
        """Load the most recent integrated dataset."""
        try:
            # Find the most recent integrated dataset
            dataset_files = list(Path("data/processed").glob("integrated_romance_novels_nlp_ready_*.csv"))
            if not dataset_files:
                raise FileNotFoundError("No integrated dataset files found")
            
            latest_file = max(dataset_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading integrated dataset from: {latest_file}")
            
            # Load dataset with progress tracking
            logger.info("Loading integrated dataset...")
            self.integrated_dataset = pd.read_csv(latest_file)
            logger.info(f"Loaded integrated dataset: {len(self.integrated_dataset):,} records")
            
            # Basic info
            logger.info(f"Dataset columns: {len(self.integrated_dataset.columns)}")
            logger.info(f"Dataset size: {self.integrated_dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error loading integrated dataset: {str(e)}")
            raise
    
    def validate_data_structure(self):
        """Validate the dataset structure and column integrity."""
        logger.info("Validating dataset structure...")
        
        structure_validation = {
            'expected_columns': 23,
            'actual_columns': len(self.integrated_dataset.columns),
            'required_columns': [
                'work_id', 'title', 'publication_year', 'author_id', 'author_name',
                'description', 'duplication_status', 'cleaning_strategy', 'disambiguation_notes'
            ],
            'cleaning_metadata_columns': [
                'duplication_status', 'cleaning_strategy', 'disambiguation_notes', 'cleaning_timestamp'
            ],
            'missing_columns': [],
            'extra_columns': []
        }
        
        # Check required columns
        for col in structure_validation['required_columns']:
            if col not in self.integrated_dataset.columns:
                structure_validation['missing_columns'].append(col)
        
        # Check for extra columns
        actual_columns = set(self.integrated_dataset.columns)
        expected_columns = set(structure_validation['required_columns'])
        structure_validation['extra_columns'] = list(actual_columns - expected_columns)
        
        # Validate column count
        structure_validation['structure_valid'] = (
            structure_validation['actual_columns'] == structure_validation['expected_columns'] and
            len(structure_validation['missing_columns']) == 0
        )
        
        self.validation_results['structure_validation'] = structure_validation
        logger.info(f"Structure validation: {'PASSED' if structure_validation['structure_valid'] else 'FAILED'}")
        
        return structure_validation
    
    def validate_data_integrity(self):
        """Validate data integrity and consistency."""
        logger.info("Validating data integrity...")
        
        integrity_validation = {
            'no_duplicate_work_ids': True,
            'no_null_critical_fields': True,
            'publication_year_range_valid': True,
            'data_type_consistency': True,
            'issues_found': []
        }
        
        # Check for duplicate work_ids
        duplicate_work_ids = self.integrated_dataset['work_id'].duplicated().sum()
        if duplicate_work_ids > 0:
            integrity_validation['no_duplicate_work_ids'] = False
            integrity_validation['issues_found'].append(f"Found {duplicate_work_ids} duplicate work_ids")
        
        # Check for null values in critical fields
        critical_fields = ['work_id', 'title', 'publication_year', 'author_id', 'author_name']
        for field in critical_fields:
            null_count = self.integrated_dataset[field].isnull().sum()
            if null_count > 0:
                integrity_validation['no_null_critical_fields'] = False
                integrity_validation['issues_found'].append(f"Found {null_count} null values in {field}")
        
        # Check publication year range
        min_year = self.integrated_dataset['publication_year'].min()
        max_year = self.integrated_dataset['publication_year'].max()
        if min_year < 1900 or max_year > 2025:
            integrity_validation['publication_year_range_valid'] = False
            integrity_validation['issues_found'].append(f"Publication year range invalid: {min_year}-{max_year}")
        
        # Check data types
        expected_types = {
            'work_id': 'int64',
            'publication_year': 'int64',
            'author_id': 'int64'
        }
        for field, expected_type in expected_types.items():
            if field in self.integrated_dataset.columns:
                actual_type = str(self.integrated_dataset[field].dtype)
                if actual_type != expected_type:
                    integrity_validation['data_type_consistency'] = False
                    integrity_validation['issues_found'].append(f"Data type mismatch for {field}: expected {expected_type}, got {actual_type}")
        
        integrity_validation['overall_integrity'] = all([
            integrity_validation['no_duplicate_work_ids'],
            integrity_validation['no_null_critical_fields'],
            integrity_validation['publication_year_range_valid'],
            integrity_validation['data_type_consistency']
        ])
        
        self.validation_results['integrity_validation'] = integrity_validation
        logger.info(f"Integrity validation: {'PASSED' if integrity_validation['overall_integrity'] else 'FAILED'}")
        
        return integrity_validation
    
    def validate_cleaning_metadata(self):
        """Validate that cleaning metadata was properly applied."""
        logger.info("Validating cleaning metadata...")
        
        metadata_validation = {
            'duplication_status_coverage': 0.0,
            'cleaning_strategy_coverage': 0.0,
            'disambiguation_notes_coverage': 0.0,
            'cleaning_timestamp_coverage': 0.0,
            'metadata_quality': 'UNKNOWN',
            'issues_found': []
        }
        
        # Check coverage of cleaning metadata
        total_records = len(self.integrated_dataset)
        
        # Duplication status coverage
        dup_status_coverage = self.integrated_dataset['duplication_status'].notna().sum() / total_records
        metadata_validation['duplication_status_coverage'] = dup_status_coverage
        
        # Cleaning strategy coverage
        cleaning_strategy_coverage = self.integrated_dataset['cleaning_strategy'].notna().sum() / total_records
        metadata_validation['cleaning_strategy_coverage'] = cleaning_strategy_coverage
        
        # Disambiguation notes coverage
        disambig_coverage = self.integrated_dataset['disambiguation_notes'].notna().sum() / total_records
        metadata_validation['disambiguation_notes_coverage'] = disambig_coverage
        
        # Cleaning timestamp coverage
        timestamp_coverage = self.integrated_dataset['cleaning_timestamp'].notna().sum() / total_records
        metadata_validation['cleaning_timestamp_coverage'] = timestamp_coverage
        
        # Assess overall metadata quality
        avg_coverage = np.mean([
            dup_status_coverage, cleaning_strategy_coverage, 
            disambig_coverage, timestamp_coverage
        ])
        
        if avg_coverage >= 0.95:
            metadata_validation['metadata_quality'] = 'EXCELLENT'
        elif avg_coverage >= 0.90:
            metadata_validation['metadata_quality'] = 'GOOD'
        elif avg_coverage >= 0.80:
            metadata_validation['metadata_quality'] = 'FAIR'
        else:
            metadata_validation['metadata_quality'] = 'POOR'
            metadata_validation['issues_found'].append(f"Low metadata coverage: {avg_coverage:.1%}")
        
        # Check for consistency in cleaning strategies
        cleaning_strategies = self.integrated_dataset['cleaning_strategy'].value_counts()
        if 'none' not in cleaning_strategies:
            metadata_validation['issues_found'].append("Missing 'none' cleaning strategy for unique titles")
        
        self.validation_results['metadata_validation'] = metadata_validation
        logger.info(f"Metadata validation: {metadata_validation['metadata_quality']}")
        
        return metadata_validation
    
    def examine_sample_records(self, sample_size: int = 10):
        """Examine sample records to verify data quality."""
        logger.info(f"Examining {sample_size} sample records...")
        
        # Get random sample
        sample_indices = random.sample(range(len(self.integrated_dataset)), min(sample_size, len(self.integrated_dataset)))
        sample_records = self.integrated_dataset.iloc[sample_indices]
        
        sample_analysis = {
            'sample_size': len(sample_records),
            'records': [],
            'quality_observations': [],
            'issues_found': []
        }
        
        for idx, record in sample_records.iterrows():
            record_info = {
                'work_id': record['work_id'],
                'title': record['title'][:100] + '...' if len(str(record['title'])) > 100 else record['title'],
                'author_name': record['author_name'],
                'publication_year': record['publication_year'],
                'duplication_status': record['duplication_status'],
                'cleaning_strategy': record['cleaning_strategy'],
                'has_disambiguation': pd.notna(record['disambiguation_notes'])
            }
            
            # Check for potential issues
            if pd.isna(record['title']) or str(record['title']).strip() == '':
                sample_analysis['issues_found'].append(f"Empty title for work_id {record['work_id']}")
            
            if pd.isna(record['author_name']) or str(record['author_name']).strip() == '':
                sample_analysis['issues_found'].append(f"Empty author for work_id {record['work_id']}")
            
            if record['publication_year'] < 1900 or record['publication_year'] > 2025:
                sample_analysis['issues_found'].append(f"Invalid publication year {record['publication_year']} for work_id {record['work_id']}")
            
            sample_analysis['records'].append(record_info)
        
        # Overall quality observations
        unique_titles = sample_records['duplication_status'].value_counts()
        if 'unique' in unique_titles:
            sample_analysis['quality_observations'].append(f"Sample contains {unique_titles['unique']} unique titles")
        
        if 'legitimate_duplicate' in unique_titles:
            sample_analysis['quality_observations'].append(f"Sample contains {unique_titles['legitimate_duplicate']} legitimate duplicates")
        
        cleaning_strategies = sample_records['cleaning_strategy'].value_counts()
        if 'none' in cleaning_strategies:
            sample_analysis['quality_observations'].append(f"Sample contains {cleaning_strategies['none']} records with no cleaning needed")
        
        self.validation_results['sample_analysis'] = sample_analysis
        logger.info(f"Sample analysis completed: {len(sample_analysis['records'])} records examined")
        
        return sample_analysis
    
    def calculate_quality_metrics(self):
        """Calculate comprehensive quality metrics."""
        logger.info("Calculating quality metrics...")
        
        quality_metrics = {
            'completeness_scores': {},
            'consistency_scores': {},
            'overall_quality_score': 0.0,
            'quality_grade': 'UNKNOWN'
        }
        
        # Calculate completeness scores for key fields
        key_fields = ['work_id', 'title', 'publication_year', 'author_id', 'author_name', 'description']
        for field in key_fields:
            if field in self.integrated_dataset.columns:
                completeness = self.integrated_dataset[field].notna().mean()
                quality_metrics['completeness_scores'][field] = completeness
        
        # Calculate consistency scores
        quality_metrics['consistency_scores'] = {
            'no_duplicate_work_ids': 1.0 if self.integrated_dataset['work_id'].duplicated().sum() == 0 else 0.0,
            'valid_publication_years': 1.0 if all(
                (self.integrated_dataset['publication_year'] >= 1900) & 
                (self.integrated_dataset['publication_year'] <= 2025)
            ) else 0.0,
            'cleaning_metadata_present': 1.0 if all([
                self.integrated_dataset['duplication_status'].notna().all(),
                self.integrated_dataset['cleaning_strategy'].notna().all()
            ]) else 0.0
        }
        
        # Calculate overall quality score
        completeness_avg = np.mean(list(quality_metrics['completeness_scores'].values()))
        consistency_avg = np.mean(list(quality_metrics['consistency_scores'].values()))
        quality_metrics['overall_quality_score'] = (completeness_avg + consistency_avg) / 2 * 100
        
        # Assign quality grade
        score = quality_metrics['overall_quality_score']
        if score >= 95:
            quality_metrics['quality_grade'] = 'A+'
        elif score >= 90:
            quality_metrics['quality_grade'] = 'A'
        elif score >= 85:
            quality_metrics['quality_grade'] = 'B+'
        elif score >= 80:
            quality_metrics['quality_grade'] = 'B'
        elif score >= 75:
            quality_metrics['quality_grade'] = 'C+'
        else:
            quality_metrics['quality_grade'] = 'C'
        
        self.validation_results['quality_metrics'] = quality_metrics
        logger.info(f"Quality metrics calculated: {quality_metrics['quality_grade']} ({score:.1f}%)")
        
        return quality_metrics
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_records': len(self.integrated_dataset),
                'total_columns': len(self.integrated_dataset.columns),
                'memory_usage_mb': self.integrated_dataset.memory_usage(deep=True).sum() / 1024**2,
                'file_path': str(self.dataset_path_pattern)
            },
            'validation_results': self.validation_results,
            'overall_assessment': self._generate_overall_assessment(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = self.output_dir / f"dataset_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {report_path}")
        return report
    
    def _generate_overall_assessment(self):
        """Generate overall assessment of dataset quality."""
        assessment = {
            'status': 'UNKNOWN',
            'ready_for_nlp': False,
            'strengths': [],
            'weaknesses': [],
            'critical_issues': []
        }
        
        # Check structure validation
        structure = self.validation_results.get('structure_validation', {})
        if structure.get('structure_valid', False):
            assessment['strengths'].append("Dataset structure is correct")
        else:
            assessment['critical_issues'].append("Dataset structure validation failed")
        
        # Check integrity validation
        integrity = self.validation_results.get('integrity_validation', {})
        if integrity.get('overall_integrity', False):
            assessment['strengths'].append("Data integrity is good")
        else:
            assessment['weaknesses'].append("Data integrity issues found")
        
        # Check metadata validation
        metadata = self.validation_results.get('metadata_validation', {})
        if metadata.get('metadata_quality') in ['EXCELLENT', 'GOOD']:
            assessment['strengths'].append("Cleaning metadata quality is good")
        else:
            assessment['weaknesses'].append("Cleaning metadata quality needs improvement")
        
        # Check quality metrics
        quality = self.validation_results.get('quality_metrics', {})
        score = quality.get('overall_quality_score', 0)
        if score >= 90:
            assessment['status'] = 'EXCELLENT'
            assessment['ready_for_nlp'] = True
        elif score >= 80:
            assessment['status'] = 'GOOD'
            assessment['ready_for_nlp'] = True
        elif score >= 70:
            assessment['status'] = 'FAIR'
            assessment['ready_for_nlp'] = False
        else:
            assessment['status'] = 'POOR'
            assessment['ready_for_nlp'] = False
        
        return assessment
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results."""
        recommendations = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_goals': []
        }
        
        assessment = self.validation_results.get('overall_assessment', {})
        
        if assessment.get('ready_for_nlp', False):
            recommendations['immediate_actions'].append("Dataset is ready for NLP preprocessing")
            recommendations['immediate_actions'].append("Proceed with topic modeling setup")
        else:
            recommendations['immediate_actions'].append("Address critical data quality issues")
            recommendations['immediate_actions'].append("Re-run validation after fixes")
        
        # Check for specific issues
        integrity = self.validation_results.get('integrity_validation', {})
        if integrity.get('issues_found'):
            recommendations['short_term_improvements'].append("Fix identified integrity issues")
        
        metadata = self.validation_results.get('metadata_validation', {})
        if metadata.get('metadata_quality') in ['FAIR', 'POOR']:
            recommendations['short_term_improvements'].append("Improve cleaning metadata quality")
        
        # Long-term goals
        recommendations['long_term_goals'].append("Implement automated quality monitoring")
        recommendations['long_term_goals'].append("Set up data quality dashboards")
        recommendations['long_term_goals'].append("Establish data quality SLAs")
        
        return recommendations
    
    def run_full_validation(self):
        """Run the complete validation pipeline."""
        logger.info("Starting full dataset validation pipeline...")
        
        try:
            # Load dataset
            self.load_integrated_dataset()
            
            # Run validation components
            self.validate_data_structure()
            self.validate_data_integrity()
            self.validate_cleaning_metadata()
            self.examine_sample_records()
            self.calculate_quality_metrics()
            
            # Generate final report
            report = self.generate_validation_report()
            
            logger.info("Full validation pipeline completed successfully")
            
            return report
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {str(e)}")
            raise

def main():
    """Main execution function."""
    print("=" * 80)
    print("Dataset Validation: Integrated Romance Novels Dataset")
    print("=" * 80)
    
    try:
        # Initialize validator
        validator = DatasetValidator()
        
        # Run full validation
        report = validator.run_full_validation()
        
        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        dataset_info = report['dataset_info']
        print(f"Total Records: {dataset_info['total_records']:,}")
        print(f"Total Columns: {dataset_info['total_columns']}")
        print(f"Memory Usage: {dataset_info['memory_usage_mb']:.2f} MB")
        
        quality_metrics = report['validation_results']['quality_metrics']
        print(f"\nOverall Quality: {quality_metrics['quality_grade']} ({quality_metrics['overall_quality_score']:.1f}%)")
        
        overall_assessment = report['overall_assessment']
        print(f"Status: {overall_assessment['status']}")
        print(f"NLP Ready: {'✅ YES' if overall_assessment['ready_for_nlp'] else '❌ NO'}")
        
        if overall_assessment['strengths']:
            print(f"\nStrengths: {', '.join(overall_assessment['strengths'])}")
        
        if overall_assessment['weaknesses']:
            print(f"Weaknesses: {', '.join(overall_assessment['weaknesses'])}")
        
        print(f"\nValidation Report: {report['validation_timestamp']}")
        print("Check the generated report for detailed findings.")
        
        print("\n" + "=" * 80)
        print("Validation complete! Check the generated files for detailed results.")
        print("=" * 80)
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        logger.error(f"Main execution failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
