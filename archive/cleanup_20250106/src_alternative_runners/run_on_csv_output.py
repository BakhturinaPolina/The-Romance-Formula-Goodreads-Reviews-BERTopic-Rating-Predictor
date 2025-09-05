#!/usr/bin/env python3
"""
Run Data Quality Pipeline on CSV Builder Output with Batch Processing
This script runs the complete 6-step data quality pipeline on the output from the CSV builder,
processing data in batches to handle large datasets efficiently.
"""

import sys
import logging
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import all pipeline components
from step1_missing_values_cleaning import MissingValuesCleaner
from step2_duplicate_detection import DuplicateDetector
from step3_data_type_validation import DataTypeValidator
from step4_outlier_detection import OutlierDetectionReporter
from step4_outlier_treatment import OutlierTreatmentApplier
from step5_data_type_optimization import DataTypeOptimizer
from step6_final_quality_validation import FinalQualityValidator


class BatchDataQualityPipeline:
    """
    Batch processing version of the data quality pipeline for large datasets.
    """
    
    def __init__(self, input_data_path: str, batch_size: int = 10000, use_sample: bool = True):
        """
        Initialize the batch processing pipeline.
        
        Args:
            input_data_path: Path to the input CSV file
            batch_size: Number of records to process in each batch
            use_sample: Whether to use sample data for testing
        """
        self.input_data_path = input_data_path
        self.batch_size = batch_size
        self.use_sample = use_sample
        
        # Setup logging
        self.setup_logging()
        
        # Initialize pipeline steps
        self.pipeline_steps = {}
        
        # Track progress
        self.total_records = 0
        self.processed_records = 0
        self.current_batch = 0
        self.start_time = None
        
    def setup_logging(self):
        """Setup comprehensive logging for batch processing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("../../logs/data_quality_batch")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"batch_pipeline_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Batch Data Quality Pipeline initialized")
        self.logger.info(f"📁 Input file: {self.input_data_path}")
        self.logger.info(f"📊 Batch size: {self.batch_size}")
        self.logger.info(f"📝 Log file: {log_file}")
        
    def get_total_records(self) -> int:
        """Get total number of records in the dataset."""
        try:
            # Read just the first few rows to get column info
            sample_df = pd.read_csv(self.input_data_path, nrows=5)
            self.logger.info(f"📋 Dataset columns: {list(sample_df.columns)}")
            
            # Count total records efficiently
            with open(self.input_data_path, 'r') as f:
                total_records = sum(1 for line in f) - 1  # Subtract header
                
            self.logger.info(f"📊 Total records in dataset: {total_records:,}")
            return total_records
            
        except Exception as e:
            self.logger.error(f"❌ Error getting total records: {e}")
            return 0
    
    def process_batch(self, batch_df: pd.DataFrame, batch_num: int) -> Dict[str, Any]:
        """
        Process a single batch through the data quality pipeline.
        
        Args:
            batch_df: DataFrame containing the batch data
            batch_num: Batch number for logging
            
        Returns:
            Dictionary containing batch processing results
        """
        self.logger.info(f"🔄 Processing batch {batch_num} ({len(batch_df):,} records)")
        
        batch_results = {
            'batch_num': batch_num,
            'records_in': len(batch_df),
            'records_out': 0,
            'processing_time': 0,
            'steps_completed': [],
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Missing Values Treatment
            self.logger.info(f"  📋 Step 1: Missing Values Treatment (batch {batch_num})")
            cleaner = MissingValuesCleaner()
            cleaned_df, _ = cleaner.apply_missing_value_treatment()
            batch_results['steps_completed'].append('step1_missing_values')
            
            # Step 2: Duplicate Detection & Resolution
            self.logger.info(f"  🔍 Step 2: Duplicate Detection (batch {batch_num})")
            detector = DuplicateDetector()
            deduplicated_df, _ = detector.run_complete_resolution()
            batch_results['steps_completed'].append('step2_duplicates')
            
            # Step 3: Data Type Validation & Conversion
            self.logger.info(f"  🔧 Step 3: Data Type Validation (batch {batch_num})")
            validator = DataTypeValidator()
            validated_df = validator.validate_and_convert_types(deduplicated_df)
            batch_results['steps_completed'].append('step3_data_types')
            
            # Step 4: Outlier Detection (analysis only)
            self.logger.info(f"  📊 Step 4a: Outlier Detection (batch {batch_num})")
            reporter = OutlierDetectionReporter()
            outlier_results = reporter.analyze_outliers(validated_df)
            batch_results['steps_completed'].append('step4a_outlier_detection')
            
            # Step 4b: Outlier Treatment
            self.logger.info(f"  🛠️ Step 4b: Outlier Treatment (batch {batch_num})")
            applier = OutlierTreatmentApplier()
            treated_df = applier.apply_outlier_treatment(validated_df)
            batch_results['steps_completed'].append('step4b_outlier_treatment')
            
            # Step 5: Data Type Optimization
            self.logger.info(f"  ⚡ Step 5: Data Type Optimization (batch {batch_num})")
            optimizer = DataTypeOptimizer()
            optimized_df = optimizer.optimize_data_types(treated_df)
            batch_results['steps_completed'].append('step5_optimization')
            
            # Step 6: Final Quality Validation
            self.logger.info(f"  ✅ Step 6: Final Quality Validation (batch {batch_num})")
            validator_final = FinalQualityValidator()
            validation_results = validator_final.validate_final_quality(optimized_df)
            batch_results['steps_completed'].append('step6_final_validation')
            
            batch_results['records_out'] = len(optimized_df)
            batch_results['processing_time'] = time.time() - start_time
            
            self.logger.info(f"  ✅ Batch {batch_num} completed: {len(batch_df):,} → {len(optimized_df):,} records in {batch_results['processing_time']:.2f}s")
            
            return batch_results, optimized_df
            
        except Exception as e:
            self.logger.error(f"  ❌ Error processing batch {batch_num}: {e}")
            batch_results['errors'].append(str(e))
            return batch_results, None
    
    def run_batch_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete data quality pipeline in batches.
        
        Returns:
            Dictionary containing overall pipeline results
        """
        self.logger.info("🚀 Starting Batch Data Quality Pipeline")
        self.start_time = time.time()
        
        # Get total records
        self.total_records = self.get_total_records()
        if self.total_records == 0:
            self.logger.error("❌ No records found in dataset")
            return {'success': False, 'error': 'No records found'}
        
        # Calculate number of batches
        num_batches = (self.total_records + self.batch_size - 1) // self.batch_size
        self.logger.info(f"📊 Processing {self.total_records:,} records in {num_batches} batches of {self.batch_size:,}")
        
        # Initialize results tracking
        pipeline_results = {
            'total_records': self.total_records,
            'batch_size': self.batch_size,
            'num_batches': num_batches,
            'batches_processed': 0,
            'total_processing_time': 0,
            'batch_results': [],
            'success': True,
            'errors': []
        }
        
        # Process each batch
        for batch_num in range(1, num_batches + 1):
            try:
                self.logger.info(f"🔄 Starting batch {batch_num}/{num_batches}")
                
                # Read batch
                start_row = (batch_num - 1) * self.batch_size
                end_row = min(start_row + self.batch_size, self.total_records)
                
                self.logger.info(f"  📥 Reading rows {start_row:,} to {end_row:,}")
                batch_df = pd.read_csv(
                    self.input_data_path,
                    skiprows=range(1, start_row + 1),
                    nrows=self.batch_size
                )
                
                self.logger.info(f"  📊 Batch {batch_num} loaded: {len(batch_df):,} records")
                
                # Process batch
                batch_results, processed_df = self.process_batch(batch_df, batch_num)
                pipeline_results['batch_results'].append(batch_results)
                pipeline_results['batches_processed'] += 1
                
                # Update progress
                self.processed_records += len(batch_df)
                progress_pct = (self.processed_records / self.total_records) * 100
                self.logger.info(f"  📈 Progress: {self.processed_records:,}/{self.total_records:,} ({progress_pct:.1f}%)")
                
                # Save batch results if needed
                if processed_df is not None:
                    batch_output_path = f"../../outputs/batch_processing/batch_{batch_num:03d}_processed.csv"
                    Path(batch_output_path).parent.mkdir(parents=True, exist_ok=True)
                    processed_df.to_csv(batch_output_path, index=False)
                    self.logger.info(f"  💾 Batch {batch_num} saved to: {batch_output_path}")
                
            except Exception as e:
                self.logger.error(f"❌ Error processing batch {batch_num}: {e}")
                pipeline_results['errors'].append(f"Batch {batch_num}: {str(e)}")
                pipeline_results['success'] = False
        
        # Calculate total processing time
        pipeline_results['total_processing_time'] = time.time() - self.start_time
        
        # Log final results
        self.logger.info("🎉 Batch Pipeline Processing Complete!")
        self.logger.info(f"📊 Total records processed: {self.processed_records:,}")
        self.logger.info(f"⏱️ Total processing time: {pipeline_results['total_processing_time']:.2f} seconds")
        self.logger.info(f"📈 Average time per batch: {pipeline_results['total_processing_time']/num_batches:.2f} seconds")
        self.logger.info(f"✅ Batches completed successfully: {pipeline_results['batches_processed']}/{num_batches}")
        
        return pipeline_results


def main():
    """Main function to run the batch data quality pipeline."""
    print("🚀 Batch Data Quality Pipeline for CSV Builder Output")
    print("=" * 60)
    
    # Ask user for processing mode
    print("\n📊 Choose processing mode:")
    print("1. Test with sample data (100 books)")
    print("2. Process full dataset in batches")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        use_sample = True
        batch_size = 50  # Smaller batches for testing
        print(f"\n🧪 Testing mode: Sample data with batch size {batch_size}")
    else:
        use_sample = False
        batch_size = 5000  # Larger batches for full dataset
        print(f"\n🏭 Production mode: Full dataset with batch size {batch_size}")
    
    # Set input file path
    if use_sample:
        input_data_path = "../../data/processed/final_books_2000_2020_en_enhanced_titles_sampled_100_20250904_210625.csv"
        print(f"\n📊 Processing sample dataset: {input_data_path}")
    else:
        input_data_path = "../../data/processed/final_books_2000_2020_en_enhanced_20250904_215835.csv"
        print(f"\n📊 Processing full dataset: {input_data_path}")
    
    # Initialize and run pipeline
    pipeline = BatchDataQualityPipeline(
        input_data_path=input_data_path,
        batch_size=batch_size,
        use_sample=use_sample
    )
    
    # Run the pipeline
    results = pipeline.run_batch_pipeline()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BATCH PIPELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Success: {results['success']}")
    print(f"📊 Total records: {results['total_records']:,}")
    print(f"📦 Batches processed: {results['batches_processed']}/{results['num_batches']}")
    print(f"⏱️ Total time: {results['total_processing_time']:.2f} seconds")
    
    if results['errors']:
        print(f"❌ Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  • {error}")
    
    print(f"\n📝 Detailed logs saved to: logs/data_quality_batch/")
    print(f"💾 Batch outputs saved to: outputs/batch_processing/")


if __name__ == "__main__":
    main()