#!/usr/bin/env python3
"""
Unified Data Quality Pipeline Runner

This script executes the complete 6-step data quality pipeline for romance novel datasets:
1. Missing Values Treatment
2. Duplicate Detection & Resolution
3. Data Type Validation & Conversion
4. Outlier Detection & Treatment
5. Data Type Optimization & Persistence
6. Final Quality Validation & Certification

Author: Research Assistant
Date: 2025-09-02
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import all pipeline components
from .step1_missing_values_cleaning import MissingValuesCleaner
from .step2_duplicate_detection import DuplicateDetector
from .step3_data_type_validation import DataTypeValidator
from .step4_outlier_detection import OutlierDetectionReporter
from .step4_outlier_treatment import OutlierTreatmentApplier
from .step5_data_type_optimization import DataTypeOptimizer
from .step6_final_quality_validation import FinalQualityValidator

def setup_logging():
    """Set up enhanced logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/data_quality_pipeline_{timestamp}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class DataQualityPipelineRunner:
    """
    Unified runner for the complete 6-step data quality pipeline.
    """
    
    def __init__(self, input_data_path: str = None):
        """
        Initialize the pipeline runner.
        
        Args:
            input_data_path: Path to the input dataset file
        """
        self.input_data_path = input_data_path or "data/processed/romance_novels_integrated.csv"
        self.pipeline_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("outputs/pipeline_execution")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline step configurations
        self.pipeline_steps = {
            'step1': {
                'name': 'Missing Values Treatment',
                'class': MissingValuesCleaner,
                'input_path': self.input_data_path,
                'output_path': None,
                'enabled': True
            },
            'step2': {
                'name': 'Duplicate Detection & Resolution',
                'class': DuplicateDetector,
                'input_path': None,  # Will be set from previous step
                'output_path': None,
                'enabled': True
            },
            'step3': {
                'name': 'Data Type Validation & Conversion',
                'class': DataTypeValidator,
                'input_path': None,  # Will be set from previous step
                'output_path': None,
                'enabled': True
            },
            'step4_detection': {
                'name': 'Outlier Detection',
                'class': OutlierDetectionReporter,
                'input_path': None,  # Will be set from previous step
                'output_path': None,
                'enabled': True
            },
            'step4_treatment': {
                'name': 'Outlier Treatment',
                'class': OutlierTreatmentApplier,
                'input_path': None,  # Will be set from previous step
                'output_path': None,
                'enabled': True
            },
            'step5': {
                'name': 'Data Type Optimization & Persistence',
                'class': DataTypeOptimizer,
                'input_path': None,  # Will be set from previous step
                'output_path': None,
                'enabled': True
            },
            'step6': {
                'name': 'Final Quality Validation & Certification',
                'class': FinalQualityValidator,
                'input_path': None,  # Will be set from previous step
                'output_path': None,
                'enabled': True
            }
        }
    
    def validate_input_data(self) -> bool:
        """
        Validate that the input data file exists and is accessible.
        
        Returns:
            bool: True if input data is valid
        """
        input_path = Path(self.input_data_path)
        
        if not input_path.exists():
            print(f"❌ Input data file not found: {self.input_data_path}")
            return False
        
        if not input_path.is_file():
            print(f"❌ Input path is not a file: {self.input_data_path}")
            return False
        
        # Check file size
        file_size_mb = input_path.stat().st_size / 1024 / 1024
        print(f"✅ Input data file found: {self.input_data_path}")
        print(f"   File size: {file_size_mb:.2f} MB")
        
        return True
    
    def run_step1_missing_values_treatment(self) -> Dict[str, Any]:
        """
        Run Step 1: Missing Values Treatment.
        
        Returns:
            Dictionary with step 1 results
        """
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Step 1: Missing Values Treatment...")
        
        try:
            # Initialize cleaner
            cleaner = MissingValuesCleaner(self.input_data_path)
            
            # Run complete treatment
            results = cleaner.run_complete_treatment()
            
            # Get the actual output path from the results
            # The output path is stored in the results, we need to construct it
            actual_output_path = f"outputs/missing_values_cleaning/romance_novels_step1_missing_values_treated_{cleaner.timestamp}.pkl"
            self.pipeline_steps['step1']['output_path'] = actual_output_path
            self.pipeline_steps['step2']['input_path'] = actual_output_path
            
            logger.info("✅ Step 1: Missing Values Treatment completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Step 1 failed: {str(e)}", exc_info=True)
            raise
    
    def run_step2_duplicate_detection(self) -> Dict[str, Any]:
        """
        Run Step 2: Duplicate Detection & Resolution.
        
        Returns:
            Dictionary with step 2 results
        """
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Step 2: Duplicate Detection & Resolution...")
        
        try:
            # Initialize detector
            detector = DuplicateDetector(self.pipeline_steps['step2']['input_path'])
            
            # Run complete resolution
            results = detector.run_complete_resolution()
            
            # Get the actual output path from the detector
            actual_output_path = f"outputs/duplicate_detection/romance_novels_step2_duplicates_resolved_{detector.timestamp}.pkl"
            self.pipeline_steps['step2']['output_path'] = actual_output_path
            self.pipeline_steps['step3']['input_path'] = actual_output_path
            
            logger.info("✅ Step 2: Duplicate Detection & Resolution completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Step 2 failed: {str(e)}", exc_info=True)
            raise
    
    def run_step3_data_type_validation(self) -> Dict[str, Any]:
        """
        Run Step 3: Data Type Validation & Conversion.
        
        Returns:
            Dictionary with step 3 results
        """
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Step 3: Data Type Validation & Conversion...")
        
        try:
            # Initialize validator
            validator = DataTypeValidator(self.pipeline_steps['step3']['input_path'])
            
            # Run complete validation
            results = validator.run_complete_validation()
            
            # Get the actual output path from the validator
            actual_output_path = f"outputs/data_type_validation/romance_novels_step3_data_types_validated_{validator.timestamp}.pkl"
            self.pipeline_steps['step3']['output_path'] = actual_output_path
            self.pipeline_steps['step4_detection']['input_path'] = actual_output_path
            
            logger.info("✅ Step 3: Data Type Validation & Conversion completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Step 3 failed: {str(e)}", exc_info=True)
            raise
    
    def run_step4_outlier_detection(self) -> Dict[str, Any]:
        """
        Run Step 4a: Outlier Detection.
        
        Returns:
            Dictionary with step 4a results
        """
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Step 4a: Outlier Detection...")
        
        try:
            # Initialize reporter
            reporter = OutlierDetectionReporter(self.pipeline_steps['step4_detection']['input_path'])
            
            # Run comprehensive analysis
            results = reporter.run_comprehensive_analysis()
            
            # Save results
            report_file = reporter.save_results()
            
            logger.info("✅ Step 4a: Outlier Detection completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Step 4a failed: {str(e)}", exc_info=True)
            raise
    
    def run_step4_outlier_treatment(self) -> Dict[str, Any]:
        """
        Run Step 4b: Outlier Treatment.
        
        Returns:
            Dictionary with step 4b results
        """
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Step 4b: Outlier Treatment...")
        
        try:
            # Initialize applier
            applier = OutlierTreatmentApplier(self.pipeline_steps['step4_detection']['input_path'])
            
            # Run complete treatment
            results = applier.run_complete_treatment()
            
            # Get the actual output path from the applier
            actual_output_path = f"outputs/outlier_detection/cleaned_romance_novels_step4_treated_{applier.timestamp}.pkl"
            self.pipeline_steps['step4_treatment']['output_path'] = actual_output_path
            self.pipeline_steps['step5']['input_path'] = actual_output_path
            
            logger.info("✅ Step 4b: Outlier Treatment completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Step 4b failed: {str(e)}", exc_info=True)
            raise
    
    def run_step5_data_type_optimization(self) -> Dict[str, Any]:
        """
        Run Step 5: Data Type Optimization & Persistence.
        
        Returns:
            Dictionary with step 5 results
        """
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Step 5: Data Type Optimization & Persistence...")
        
        try:
            # Initialize optimizer
            optimizer = DataTypeOptimizer(self.pipeline_steps['step5']['input_path'])
            
            # Run complete optimization
            results = optimizer.run_complete_optimization()
            
            # Get the actual output path from the optimizer
            actual_output_path = f"outputs/data_type_optimization/cleaned_romance_novels_step5_optimized_{optimizer.timestamp}.parquet"
            self.pipeline_steps['step5']['output_path'] = actual_output_path
            self.pipeline_steps['step6']['input_path'] = actual_output_path
            
            logger.info("✅ Step 5: Data Type Optimization & Persistence completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Step 5 failed: {str(e)}", exc_info=True)
            raise
    
    def run_step6_final_quality_validation(self) -> Dict[str, Any]:
        """
        Run Step 6: Final Quality Validation & Certification.
        
        Returns:
            Dictionary with step 6 results
        """
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Step 6: Final Quality Validation & Certification...")
        
        try:
            # Initialize validator
            validator = FinalQualityValidator(self.pipeline_steps['step6']['input_path'])
            
            # Run complete validation
            results = validator.run_complete_validation()
            
            # Save validation report
            report_file = validator.save_validation_report(results)
            
            logger.info("✅ Step 6: Final Quality Validation & Certification completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Step 6 failed: {str(e)}", exc_info=True)
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete 6-step data quality pipeline.
        
        Returns:
            Dictionary with complete pipeline results
        """
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting Complete Data Quality Pipeline...")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Initialize pipeline results
        self.pipeline_results = {
            'pipeline_timestamp': self.timestamp,
            'input_data_path': self.input_data_path,
            'pipeline_steps': {},
            'execution_summary': {},
            'final_outputs': {}
        }
        
        try:
            # Step 1: Missing Values Treatment
            logger.info("📋 Executing Step 1: Missing Values Treatment...")
            step1_results = self.run_step1_missing_values_treatment()
            self.pipeline_results['pipeline_steps']['step1'] = {
                'name': 'Missing Values Treatment',
                'status': 'completed',
                'results': step1_results
            }
            
            # Step 2: Duplicate Detection & Resolution
            logger.info("📋 Executing Step 2: Duplicate Detection & Resolution...")
            step2_results = self.run_step2_duplicate_detection()
            self.pipeline_results['pipeline_steps']['step2'] = {
                'name': 'Duplicate Detection & Resolution',
                'status': 'completed',
                'results': step2_results
            }
            
            # Step 3: Data Type Validation & Conversion
            logger.info("📋 Executing Step 3: Data Type Validation & Conversion...")
            step3_results = self.run_step3_data_type_validation()
            self.pipeline_results['pipeline_steps']['step3'] = {
                'name': 'Data Type Validation & Conversion',
                'status': 'completed',
                'results': step3_results
            }
            
            # Step 4a: Outlier Detection
            logger.info("📋 Executing Step 4a: Outlier Detection...")
            step4a_results = self.run_step4_outlier_detection()
            self.pipeline_results['pipeline_steps']['step4_detection'] = {
                'name': 'Outlier Detection',
                'status': 'completed',
                'results': step4a_results
            }
            
            # Step 4b: Outlier Treatment
            logger.info("📋 Executing Step 4b: Outlier Treatment...")
            step4b_results = self.run_step4_outlier_treatment()
            self.pipeline_results['pipeline_steps']['step4_treatment'] = {
                'name': 'Outlier Treatment',
                'status': 'completed',
                'results': step4b_results
            }
            
            # Step 5: Data Type Optimization & Persistence
            logger.info("📋 Executing Step 5: Data Type Optimization & Persistence...")
            step5_results = self.run_step5_data_type_optimization()
            self.pipeline_results['pipeline_steps']['step5'] = {
                'name': 'Data Type Optimization & Persistence',
                'status': 'completed',
                'results': step5_results
            }
            
            # Step 6: Final Quality Validation & Certification
            logger.info("📋 Executing Step 6: Final Quality Validation & Certification...")
            step6_results = self.run_step6_final_quality_validation()
            self.pipeline_results['pipeline_steps']['step6'] = {
                'name': 'Final Quality Validation & Certification',
                'status': 'completed',
                'results': step6_results
            }
            
            # Generate execution summary
            total_execution_time = time.time() - start_time
            self.pipeline_results['execution_summary'] = {
                'total_execution_time_seconds': total_execution_time,
                'total_steps_completed': len(self.pipeline_results['pipeline_steps']),
                'pipeline_status': 'completed_successfully',
                'final_outputs': {
                    'step1_output': self.pipeline_steps['step1']['output_path'],
                    'step2_output': self.pipeline_steps['step2']['output_path'],
                    'step3_output': self.pipeline_steps['step3']['output_path'],
                    'step4_output': self.pipeline_steps['step4_treatment']['output_path'],
                    'step5_output': self.pipeline_steps['step5']['output_path'],
                    'step6_output': self.pipeline_steps['step6']['input_path']
                }
            }
            
            logger.info("✅ Complete Data Quality Pipeline executed successfully!")
            logger.info(f"⏱️ Total execution time: {total_execution_time:.2f} seconds")
            
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}", exc_info=True)
            
            # Update execution summary with failure information
            self.pipeline_results['execution_summary'] = {
                'total_execution_time_seconds': time.time() - start_time,
                'pipeline_status': 'failed',
                'failure_error': str(e)
            }
            
            raise
    
    def save_pipeline_report(self, filename: str = None) -> str:
        """
        Save comprehensive pipeline execution report.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            filename = f"data_quality_pipeline_report_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        import json
        with open(filepath, 'w') as f:
            json.dump(self.pipeline_results, f, indent=2, default=str)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Pipeline report saved to: {filepath}")
        return str(filepath)
    
    def print_pipeline_summary(self):
        """Print a human-readable pipeline execution summary."""
        if not self.pipeline_results:
            print("No pipeline results available. Run run_complete_pipeline() first.")
            return
        
        print("\n" + "="*80)
        print("DATA QUALITY PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        # Pipeline overview
        print(f"Pipeline Timestamp: {self.pipeline_results['pipeline_timestamp']}")
        print(f"Input Data: {self.pipeline_results['input_data_path']}")
        
        # Step execution summary
        print(f"\n📋 PIPELINE STEPS EXECUTION:")
        for step_key, step_info in self.pipeline_results['pipeline_steps'].items():
            status_icon = "✅" if step_info['status'] == 'completed' else "❌"
            print(f"  {status_icon} {step_info['name']}: {step_info['status']}")
        
        # Execution summary
        summary = self.pipeline_results['execution_summary']
        print(f"\n⏱️ EXECUTION SUMMARY:")
        print(f"  • Total execution time: {summary['total_execution_time_seconds']:.2f} seconds")
        print(f"  • Steps completed: {summary['total_steps_completed']}")
        print(f"  • Pipeline status: {summary['pipeline_status']}")
        
        # Final outputs
        if 'final_outputs' in summary:
            print(f"\n📁 FINAL OUTPUTS:")
            for step, output_path in summary['final_outputs'].items():
                if output_path:
                    print(f"  • {step}: {Path(output_path).name}")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    # Set up logging
    logger = setup_logging()
    
    print("🚀 UNIFIED DATA QUALITY PIPELINE")
    print("=" * 60)
    print("Complete 6-Step Data Quality Pipeline for Romance Novel Datasets")
    print("=" * 60)
    
    # Configuration
    input_data_path = "data/processed/final_books_2000_2020_en_enhanced_20250907_013708.csv"
    
    try:
        # Initialize pipeline runner
        runner = DataQualityPipelineRunner(input_data_path)
        
        # Validate input data
        if not runner.validate_input_data():
            print("❌ Input data validation failed. Exiting.")
            sys.exit(1)
        
        # Run complete pipeline
        results = runner.run_complete_pipeline()
        
        # Save pipeline report
        logger.info("📊 Saving pipeline execution report...")
        report_file = runner.save_pipeline_report()
        
        # Print summary
        runner.print_pipeline_summary()
        
        print(f"\n✅ Pipeline execution report saved to: {report_file}")
        print("\n🎉 Complete Data Quality Pipeline executed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}", exc_info=True)
        print(f"\n💥 Pipeline execution failed: {str(e)}")
        print("Check the logs for detailed error information.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
