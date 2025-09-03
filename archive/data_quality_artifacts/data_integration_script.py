#!/usr/bin/env python3
"""
Data Integration Script for Title Cleaning Outputs

This script integrates the cleaned titles dataset with the main processed dataset to:
1. Merge the two datasets based on work_id
2. Preserve all cleaning metadata and disambiguation information
3. Create a unified, NLP-ready dataset
4. Validate integration results and data integrity
5. Generate integration report and quality metrics

Author: AI Assistant
Date: 2025-09-02
"""

import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/data_integration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataIntegrator:
    """Integrates cleaned titles with main processed dataset."""
    
    def __init__(self, 
                 cleaned_titles_dir: str = "outputs/title_cleaning",
                 processed_data_dir: str = "data/processed",
                 output_dir: str = "data/processed"):
        """Initialize the data integrator."""
        self.cleaned_titles_dir = Path(cleaned_titles_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.output_dir = Path(output_dir)
        
        self.cleaned_titles = None
        self.main_dataset = None
        self.integrated_dataset = None
        self.integration_report = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_datasets(self):
        """Load both datasets for integration."""
        try:
            # Load cleaned titles dataset
            cleaned_files = list(self.cleaned_titles_dir.glob("cleaned_titles_*.csv"))
            if not cleaned_files:
                raise FileNotFoundError("No cleaned titles CSV files found")
            
            latest_cleaned = max(cleaned_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading cleaned titles from: {latest_cleaned}")
            self.cleaned_titles = pd.read_csv(latest_cleaned)
            logger.info(f"Loaded cleaned titles: {len(self.cleaned_titles):,} records")
            
            # Load main processed dataset
            processed_files = list(self.processed_data_dir.glob("final_books_*.csv"))
            if not processed_files:
                raise FileNotFoundError("No main processed CSV files found")
            
            latest_processed = max(processed_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading main dataset from: {latest_processed}")
            self.main_dataset = pd.read_csv(latest_processed)
            logger.info(f"Loaded main dataset: {len(self.main_dataset):,} records")
            
            # Basic validation
            self._validate_datasets()
            
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise
    
    def _validate_datasets(self):
        """Validate that both datasets have required fields."""
        required_fields = ['work_id', 'title', 'publication_year', 'author_id', 'author_name']
        
        for field in required_fields:
            if field not in self.cleaned_titles.columns:
                raise ValueError(f"Missing required field '{field}' in cleaned titles dataset")
            if field not in self.main_dataset.columns:
                raise ValueError(f"Missing required field '{field}' in main dataset")
        
        logger.info("Dataset validation passed - all required fields present")
        
        # Check for overlapping work_ids
        cleaned_ids = set(self.cleaned_titles['work_id'])
        main_ids = set(self.main_dataset['work_id'])
        overlap = cleaned_ids.intersection(main_ids)
        
        logger.info(f"Found {len(overlap):,} overlapping work_ids between datasets")
        logger.info(f"Cleaned titles unique IDs: {len(cleaned_ids):,}")
        logger.info(f"Main dataset unique IDs: {len(main_ids):,}")
        
        if len(overlap) == 0:
            logger.warning("No overlapping work_ids found - datasets may be incompatible")
    
    def prepare_datasets_for_integration(self):
        """Prepare datasets for integration by standardizing formats."""
        logger.info("Preparing datasets for integration...")
        
        # Ensure work_id is the same type in both datasets
        self.cleaned_titles['work_id'] = self.cleaned_titles['work_id'].astype(int)
        self.main_dataset['work_id'] = self.main_dataset['work_id'].astype(int)
        
        # Create a copy of main dataset for integration
        self.integrated_dataset = self.main_dataset.copy()
        
        # Add new columns from cleaned titles if they don't exist
        new_columns = ['duplication_status', 'cleaning_strategy', 'disambiguation_notes', 'cleaning_timestamp']
        for col in new_columns:
            if col not in self.integrated_dataset.columns:
                self.integrated_dataset[col] = None
        
        logger.info("Dataset preparation completed")
    
    def integrate_datasets(self):
        """Integrate the two datasets based on work_id."""
        logger.info("Starting dataset integration...")
        
        # Create a mapping from cleaned titles
        cleaned_mapping = self.cleaned_titles.set_index('work_id')
        
        # Track integration statistics
        integration_stats = {
            'total_main_records': len(self.main_dataset),
            'total_cleaned_records': len(self.cleaned_titles),
            'records_updated': 0,
            'records_with_disambiguation': 0,
            'records_with_cleaning_strategy': 0
        }
        
        # Update main dataset with cleaning information
        for idx, row in self.integrated_dataset.iterrows():
            work_id = row['work_id']
            
            if work_id in cleaned_mapping.index:
                cleaned_row = cleaned_mapping.loc[work_id]
                
                # Update cleaning metadata
                self.integrated_dataset.at[idx, 'duplication_status'] = cleaned_row['duplication_status']
                self.integrated_dataset.at[idx, 'cleaning_strategy'] = cleaned_row['cleaning_strategy']
                self.integrated_dataset.at[idx, 'disambiguation_notes'] = cleaned_row['disambiguation_notes']
                self.integrated_dataset.at[idx, 'cleaning_timestamp'] = cleaned_row['cleaning_timestamp']
                
                integration_stats['records_updated'] += 1
                
                if pd.notna(cleaned_row['disambiguation_notes']):
                    integration_stats['records_with_disambiguation'] += 1
                
                if cleaned_row['cleaning_strategy'] != 'none':
                    integration_stats['records_with_cleaning_strategy'] += 1
        
        self.integration_report['integration_stats'] = integration_stats
        logger.info(f"Integration completed: {integration_stats['records_updated']:,} records updated")
        
        return integration_stats
    
    def validate_integration(self):
        """Validate the integration results."""
        logger.info("Validating integration results...")
        
        validation_results = {
            'total_records': len(self.integrated_dataset),
            'records_with_cleaning_data': 0,
            'data_integrity_checks': {},
            'quality_metrics': {}
        }
        
        # Check how many records have cleaning data
        cleaning_data_mask = (
            self.integrated_dataset['duplication_status'].notna() |
            self.integrated_dataset['cleaning_strategy'].notna() |
            self.integrated_dataset['disambiguation_notes'].notna()
        )
        validation_results['records_with_cleaning_data'] = int(cleaning_data_mask.sum())
        
        # Data integrity checks
        validation_results['data_integrity_checks'] = {
            'no_null_work_ids': int(self.integrated_dataset['work_id'].notna().all()),
            'no_null_titles': int(self.integrated_dataset['title'].notna().all()),
            'no_null_authors': int(self.integrated_dataset['author_id'].notna().all()),
            'no_null_publication_years': int(self.integrated_dataset['publication_year'].notna().all())
        }
        
        # Quality metrics
        validation_results['quality_metrics'] = {
            'completeness_score': self._calculate_completeness_score(),
            'integration_coverage': validation_results['records_with_cleaning_data'] / validation_results['total_records']
        }
        
        self.integration_report['validation_results'] = validation_results
        logger.info("Integration validation completed")
        
        return validation_results
    
    def _calculate_completeness_score(self):
        """Calculate overall data completeness score."""
        critical_columns = ['work_id', 'title', 'publication_year', 'author_id', 'author_name', 'description']
        completeness_scores = []
        
        for col in critical_columns:
            if col in self.integrated_dataset.columns:
                score = self.integrated_dataset[col].notna().mean()
                completeness_scores.append(score)
        
        return np.mean(completeness_scores) * 100 if completeness_scores else 0
    
    def generate_integration_report(self):
        """Generate comprehensive integration report."""
        logger.info("Generating integration report...")
        
        report = {
            'integration_timestamp': datetime.now().isoformat(),
            'input_datasets': {
                'cleaned_titles': {
                    'source': str(self.cleaned_titles_dir),
                    'record_count': len(self.cleaned_titles) if self.cleaned_titles is not None else 0,
                    'columns': list(self.cleaned_titles.columns) if self.cleaned_titles is not None else []
                },
                'main_dataset': {
                    'source': str(self.processed_data_dir),
                    'record_count': len(self.main_dataset) if self.main_dataset is not None else 0,
                    'columns': list(self.main_dataset.columns) if self.main_dataset is not None else []
                }
            },
            'integration_results': self.integration_report,
            'output_dataset': {
                'record_count': len(self.integrated_dataset) if self.integrated_dataset is not None else 0,
                'columns': list(self.integrated_dataset.columns) if self.integrated_dataset is not None else [],
                'memory_usage_mb': self.integrated_dataset.memory_usage(deep=True).sum() / 1024**2 if self.integrated_dataset is not None else 0
            },
            'next_steps_recommendations': self._generate_next_steps()
        }
        
        # Save report
        report_path = self.output_dir / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Integration report saved to: {report_path}")
        return report
    
    def _generate_next_steps(self):
        """Generate next steps recommendations based on integration results."""
        recommendations = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_goals': []
        }
        
        if self.integrated_dataset is not None:
            # Check integration success
            validation = self.integration_report.get('validation_results', {})
            integration_coverage = validation.get('quality_metrics', {}).get('integration_coverage', 0)
            
            if integration_coverage > 0.8:
                recommendations['immediate_actions'].append("Integration successful - proceed with NLP preprocessing")
                recommendations['immediate_actions'].append("Validate sample records for data quality")
            else:
                recommendations['immediate_actions'].append("Investigate low integration coverage")
                recommendations['immediate_actions'].append("Check for data compatibility issues")
            
            # Based on data quality
            completeness = validation.get('quality_metrics', {}).get('completeness_score', 0)
            if completeness >= 95:
                recommendations['short_term_goals'].append("Dataset ready for topic modeling")
                recommendations['short_term_goals'].append("Begin NLP preprocessing pipeline")
            else:
                recommendations['short_term_goals'].append("Address data quality issues before NLP analysis")
            
            # Long-term goals
            recommendations['long_term_goals'].append("Implement series-level analysis")
            recommendations['long_term_goals'].append("Begin review sentiment analysis")
            recommendations['long_term_goals'].append("Develop author theme analysis")
        
        return recommendations
    
    def save_integrated_dataset(self):
        """Save the integrated dataset to file."""
        if self.integrated_dataset is None:
            raise ValueError("No integrated dataset available. Run integration first.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"integrated_romance_novels_nlp_ready_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        logger.info(f"Saving integrated dataset to: {filepath}")
        self.integrated_dataset.to_csv(filepath, index=False)
        logger.info(f"Integrated dataset saved: {len(self.integrated_dataset):,} records")
        
        return filepath
    
    def run_full_integration(self):
        """Run the complete integration pipeline."""
        logger.info("Starting full data integration pipeline...")
        
        try:
            # Load datasets
            self.load_datasets()
            
            # Prepare for integration
            self.prepare_datasets_for_integration()
            
            # Perform integration
            integration_stats = self.integrate_datasets()
            
            # Validate results
            validation_results = self.validate_integration()
            
            # Save integrated dataset
            output_file = self.save_integrated_dataset()
            
            # Generate report
            report = self.generate_integration_report()
            
            logger.info("Full integration pipeline completed successfully")
            
            return {
                'output_file': output_file,
                'report': report,
                'stats': integration_stats,
                'validation': validation_results
            }
            
        except Exception as e:
            logger.error(f"Integration pipeline failed: {str(e)}")
            raise

def main():
    """Main execution function."""
    print("=" * 80)
    print("Data Integration: Cleaned Titles + Main Processed Dataset")
    print("=" * 80)
    
    try:
        # Initialize integrator
        integrator = DataIntegrator()
        
        # Run full integration
        results = integrator.run_full_integration()
        
        # Print summary
        print("\n" + "=" * 80)
        print("INTEGRATION SUMMARY")
        print("=" * 80)
        
        stats = results['stats']
        print(f"Main Dataset Records: {stats['total_main_records']:,}")
        print(f"Cleaned Titles Records: {stats['total_cleaned_records']:,}")
        print(f"Records Updated: {stats['records_updated']:,}")
        print(f"Records with Disambiguation: {stats['records_with_disambiguation']:,}")
        
        validation = results['validation']
        print(f"\nIntegration Coverage: {validation['quality_metrics']['integration_coverage']:.1%}")
        print(f"Data Completeness: {validation['quality_metrics']['completeness_score']:.1f}%")
        
        print(f"\nOutput File: {results['output_file']}")
        print(f"Integration Report: {results['output_file'].parent}/integration_report_*.json")
        
        print("\n" + "=" * 80)
        print("Integration complete! Check the generated files for detailed results.")
        print("=" * 80)
        
    except Exception as e:
        print(f"Integration failed: {str(e)}")
        logger.error(f"Main execution failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
