#!/usr/bin/env python3
"""
Batch Processing Runner for Data Quality Pipeline
This script processes the CSV builder output in batches to handle large datasets efficiently.
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


class BatchProcessor:
    """
    Simple batch processor that splits large CSV files into smaller chunks
    and processes them through the existing data quality pipeline.
    """
    
    def __init__(self, input_file: str, batch_size: int = 5000):
        """
        Initialize the batch processor.
        
        Args:
            input_file: Path to the input CSV file
            batch_size: Number of records per batch
        """
        self.input_file = input_file
        self.batch_size = batch_size
        
        # Setup logging
        self.setup_logging()
        
        # Track progress
        self.total_records = 0
        self.processed_records = 0
        self.start_time = None
        
    def setup_logging(self):
        """Setup comprehensive logging for batch processing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("../../logs/batch_processing")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"batch_processing_{timestamp}.log"
        
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
        self.logger.info(f"🚀 Batch Processor initialized")
        self.logger.info(f"📁 Input file: {self.input_file}")
        self.logger.info(f"📊 Batch size: {self.batch_size}")
        self.logger.info(f"📝 Log file: {log_file}")
        
    def get_total_records(self) -> int:
        """Get total number of records in the dataset."""
        try:
            # Count total records efficiently
            with open(self.input_file, 'r') as f:
                total_records = sum(1 for line in f) - 1  # Subtract header
                
            self.logger.info(f"📊 Total records in dataset: {total_records:,}")
            return total_records
            
        except Exception as e:
            self.logger.error(f"❌ Error getting total records: {e}")
            return 0
    
    def create_batch_files(self) -> list:
        """
        Split the large CSV file into smaller batch files.
        
        Returns:
            List of batch file paths
        """
        self.logger.info("🔄 Creating batch files...")
        
        # Create output directory
        batch_dir = Path("../../outputs/batch_files")
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        batch_files = []
        batch_num = 1
        
        try:
            # Read the input file in chunks
            chunk_iter = pd.read_csv(self.input_file, chunksize=self.batch_size)
            
            for chunk in chunk_iter:
                batch_file = batch_dir / f"batch_{batch_num:03d}.csv"
                chunk.to_csv(batch_file, index=False)
                batch_files.append(str(batch_file))
                
                self.logger.info(f"  📦 Created batch {batch_num}: {len(chunk):,} records → {batch_file}")
                batch_num += 1
                
        except Exception as e:
            self.logger.error(f"❌ Error creating batch files: {e}")
            return []
        
        self.logger.info(f"✅ Created {len(batch_files)} batch files")
        return batch_files
    
    def process_batch_file(self, batch_file: str, batch_num: int) -> Dict[str, Any]:
        """
        Process a single batch file through the data quality pipeline.
        
        Args:
            batch_file: Path to the batch file
            batch_num: Batch number for logging
            
        Returns:
            Dictionary containing batch processing results
        """
        self.logger.info(f"🔄 Processing batch {batch_num}: {batch_file}")
        
        batch_results = {
            'batch_num': batch_num,
            'batch_file': batch_file,
            'records_in': 0,
            'records_out': 0,
            'processing_time': 0,
            'steps_completed': [],
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Load batch data
            df = pd.read_csv(batch_file)
            batch_results['records_in'] = len(df)
            
            self.logger.info(f"  📊 Loaded {len(df):,} records")
            
            # Step 1: Basic Data Quality Checks
            self.logger.info(f"  📋 Step 1: Basic Data Quality Checks (batch {batch_num})")
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            total_missing = missing_counts.sum()
            self.logger.info(f"    • Missing values: {total_missing:,}")
            
            # Check data types
            self.logger.info(f"    • Data types: {df.dtypes.value_counts().to_dict()}")
            
            # Check for duplicates
            duplicates = df.duplicated().sum()
            self.logger.info(f"    • Duplicates: {duplicates:,}")
            
            batch_results['steps_completed'].append('step1_basic_checks')
            
            # Step 2: Data Type Optimization
            self.logger.info(f"  🔧 Step 2: Data Type Optimization (batch {batch_num})")
            
            # Optimize numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if df[col].dtype == 'int64':
                    # Check if we can downcast to int32
                    if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype('int32')
                        self.logger.info(f"    • Downcasted {col} to int32")
            
            batch_results['steps_completed'].append('step2_data_optimization')
            
            # Step 3: Save processed batch
            self.logger.info(f"  💾 Step 3: Saving processed batch (batch {batch_num})")
            
            processed_file = batch_file.replace('.csv', '_processed.csv')
            df.to_csv(processed_file, index=False)
            
            batch_results['records_out'] = len(df)
            batch_results['processing_time'] = time.time() - start_time
            batch_results['steps_completed'].append('step3_save')
            
            self.logger.info(f"  ✅ Batch {batch_num} completed: {len(df):,} records in {batch_results['processing_time']:.2f}s")
            self.logger.info(f"  💾 Saved to: {processed_file}")
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"  ❌ Error processing batch {batch_num}: {e}")
            batch_results['errors'].append(str(e))
            return batch_results
    
    def run_batch_processing(self) -> Dict[str, Any]:
        """
        Run the complete batch processing pipeline.
        
        Returns:
            Dictionary containing overall processing results
        """
        self.logger.info("🚀 Starting Batch Processing Pipeline")
        self.start_time = time.time()
        
        # Get total records
        self.total_records = self.get_total_records()
        if self.total_records == 0:
            self.logger.error("❌ No records found in dataset")
            return {'success': False, 'error': 'No records found'}
        
        # Create batch files
        batch_files = self.create_batch_files()
        if not batch_files:
            self.logger.error("❌ Failed to create batch files")
            return {'success': False, 'error': 'Failed to create batch files'}
        
        # Initialize results tracking
        processing_results = {
            'total_records': self.total_records,
            'batch_size': self.batch_size,
            'num_batches': len(batch_files),
            'batches_processed': 0,
            'total_processing_time': 0,
            'batch_results': [],
            'success': True,
            'errors': []
        }
        
        # Process each batch
        for batch_num, batch_file in enumerate(batch_files, 1):
            try:
                batch_results = self.process_batch_file(batch_file, batch_num)
                processing_results['batch_results'].append(batch_results)
                processing_results['batches_processed'] += 1
                
                # Update progress
                self.processed_records += batch_results['records_in']
                progress_pct = (self.processed_records / self.total_records) * 100
                self.logger.info(f"  📈 Progress: {self.processed_records:,}/{self.total_records:,} ({progress_pct:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"❌ Error processing batch {batch_num}: {e}")
                processing_results['errors'].append(f"Batch {batch_num}: {str(e)}")
                processing_results['success'] = False
        
        # Calculate total processing time
        processing_results['total_processing_time'] = time.time() - self.start_time
        
        # Log final results
        self.logger.info("🎉 Batch Processing Complete!")
        self.logger.info(f"📊 Total records processed: {self.processed_records:,}")
        self.logger.info(f"⏱️ Total processing time: {processing_results['total_processing_time']:.2f} seconds")
        self.logger.info(f"📈 Average time per batch: {processing_results['total_processing_time']/len(batch_files):.2f} seconds")
        self.logger.info(f"✅ Batches completed successfully: {processing_results['batches_processed']}/{len(batch_files)}")
        
        return processing_results


def main():
    """Main function to run the batch processing pipeline."""
    print("🚀 Batch Processing Pipeline for CSV Builder Output")
    print("=" * 60)
    
    # Ask user for processing mode
    print("\n📊 Choose processing mode:")
    print("1. Test with sample data (100 books)")
    print("2. Process full dataset in batches")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        batch_size = 50  # Smaller batches for testing
        input_file = "../../data/processed/final_books_2000_2020_en_enhanced_titles_sampled_100_20250904_210625.csv"
        print(f"\n🧪 Testing mode: Sample data with batch size {batch_size}")
    else:
        batch_size = 5000  # Larger batches for full dataset
        input_file = "../../data/processed/final_books_2000_2020_en_enhanced_20250904_215835.csv"
        print(f"\n🏭 Production mode: Full dataset with batch size {batch_size}")
    
    print(f"\n📊 Processing file: {input_file}")
    
    # Initialize and run processor
    processor = BatchProcessor(
        input_file=input_file,
        batch_size=batch_size
    )
    
    # Run the processing
    results = processor.run_batch_processing()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BATCH PROCESSING RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Success: {results['success']}")
    print(f"📊 Total records: {results['total_records']:,}")
    print(f"📦 Batches processed: {results['batches_processed']}/{results['num_batches']}")
    print(f"⏱️ Total time: {results['total_processing_time']:.2f} seconds")
    
    if results['errors']:
        print(f"❌ Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  • {error}")
    
    print(f"\n📝 Detailed logs saved to: logs/batch_processing/")
    print(f"💾 Batch files saved to: outputs/batch_files/")


if __name__ == "__main__":
    main()
