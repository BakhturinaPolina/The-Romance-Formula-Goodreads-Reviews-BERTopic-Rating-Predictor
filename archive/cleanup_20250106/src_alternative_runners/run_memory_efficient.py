#!/usr/bin/env python3
"""
Memory-Efficient Data Quality Pipeline
This script processes the CSV builder output in chunks to avoid memory overload,
without creating intermediate batch files.
"""

import sys
import logging
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Generator
import gc

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class MemoryEfficientProcessor:
    """
    Memory-efficient processor that processes data in chunks without loading
    the entire dataset into memory.
    """
    
    def __init__(self, input_file: str, chunk_size: int = 1000):
        """
        Initialize the memory-efficient processor.
        
        Args:
            input_file: Path to the input CSV file
            chunk_size: Number of records to process in each chunk
        """
        self.input_file = input_file
        self.chunk_size = chunk_size
        
        # Setup logging
        self.setup_logging()
        
        # Track progress and results
        self.total_records = 0
        self.processed_records = 0
        self.start_time = None
        
        # Accumulate results
        self.quality_metrics = {
            'total_missing_values': 0,
            'total_duplicates': 0,
            'data_type_optimizations': 0,
            'outliers_detected': 0,
            'chunks_processed': 0
        }
        
    def setup_logging(self):
        """Setup comprehensive logging for memory-efficient processing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("../../logs/memory_efficient")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"memory_efficient_{timestamp}.log"
        
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
        self.logger.info(f"🚀 Memory-Efficient Processor initialized")
        self.logger.info(f"📁 Input file: {self.input_file}")
        self.logger.info(f"📊 Chunk size: {self.chunk_size}")
        self.logger.info(f"📝 Log file: {log_file}")
        
    def get_total_records(self) -> int:
        """Get total number of records in the dataset efficiently."""
        try:
            # Count total records efficiently
            with open(self.input_file, 'r') as f:
                total_records = sum(1 for line in f) - 1  # Subtract header
                
            self.logger.info(f"📊 Total records in dataset: {total_records:,}")
            return total_records
            
        except Exception as e:
            self.logger.error(f"❌ Error getting total records: {e}")
            return 0
    
    def process_chunk(self, chunk_df: pd.DataFrame, chunk_num: int) -> Dict[str, Any]:
        """
        Process a single chunk through data quality checks.
        
        Args:
            chunk_df: DataFrame containing the chunk data
            chunk_num: Chunk number for logging
            
        Returns:
            Dictionary containing chunk processing results
        """
        self.logger.info(f"🔄 Processing chunk {chunk_num} ({len(chunk_df):,} records)")
        
        chunk_results = {
            'chunk_num': chunk_num,
            'records_in': len(chunk_df),
            'processing_time': 0,
            'quality_metrics': {}
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Missing Values Analysis
            self.logger.info(f"  📋 Step 1: Missing Values Analysis (chunk {chunk_num})")
            missing_counts = chunk_df.isnull().sum()
            total_missing = missing_counts.sum()
            chunk_results['quality_metrics']['missing_values'] = total_missing
            self.quality_metrics['total_missing_values'] += total_missing
            self.logger.info(f"    • Missing values: {total_missing:,}")
            
            # Step 2: Duplicate Detection
            self.logger.info(f"  🔍 Step 2: Duplicate Detection (chunk {chunk_num})")
            duplicates = chunk_df.duplicated().sum()
            chunk_results['quality_metrics']['duplicates'] = duplicates
            self.quality_metrics['total_duplicates'] += duplicates
            self.logger.info(f"    • Duplicates: {duplicates:,}")
            
            # Step 3: Data Type Analysis and Optimization
            self.logger.info(f"  🔧 Step 3: Data Type Analysis (chunk {chunk_num})")
            data_types = chunk_df.dtypes.value_counts().to_dict()
            chunk_results['quality_metrics']['data_types'] = data_types
            self.logger.info(f"    • Data types: {data_types}")
            
            # Optimize numeric columns
            optimizations = 0
            numeric_cols = chunk_df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if chunk_df[col].dtype == 'int64':
                    # Check if we can downcast to int32
                    if chunk_df[col].min() >= -2147483648 and chunk_df[col].max() <= 2147483647:
                        optimizations += 1
                        self.logger.info(f"    • Can optimize {col} to int32")
            
            chunk_results['quality_metrics']['optimizations'] = optimizations
            self.quality_metrics['data_type_optimizations'] += optimizations
            
            # Step 4: Outlier Detection (basic)
            self.logger.info(f"  📊 Step 4: Basic Outlier Detection (chunk {chunk_num})")
            outliers = 0
            numeric_cols = chunk_df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if chunk_df[col].dtype in ['int64', 'float64']:
                    Q1 = chunk_df[col].quantile(0.25)
                    Q3 = chunk_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    chunk_outliers = ((chunk_df[col] < lower_bound) | (chunk_df[col] > upper_bound)).sum()
                    outliers += chunk_outliers
            
            chunk_results['quality_metrics']['outliers'] = outliers
            self.quality_metrics['outliers_detected'] += outliers
            self.logger.info(f"    • Outliers detected: {outliers:,}")
            
            # Step 5: Memory cleanup
            del chunk_df
            gc.collect()
            
            chunk_results['processing_time'] = time.time() - start_time
            self.quality_metrics['chunks_processed'] += 1
            
            self.logger.info(f"  ✅ Chunk {chunk_num} completed in {chunk_results['processing_time']:.2f}s")
            
            return chunk_results
            
        except Exception as e:
            self.logger.error(f"  ❌ Error processing chunk {chunk_num}: {e}")
            chunk_results['error'] = str(e)
            return chunk_results
    
    def run_memory_efficient_processing(self) -> Dict[str, Any]:
        """
        Run the memory-efficient processing pipeline.
        
        Returns:
            Dictionary containing overall processing results
        """
        self.logger.info("🚀 Starting Memory-Efficient Processing Pipeline")
        self.start_time = time.time()
        
        # Get total records
        self.total_records = self.get_total_records()
        if self.total_records == 0:
            self.logger.error("❌ No records found in dataset")
            return {'success': False, 'error': 'No records found'}
        
        # Calculate number of chunks
        num_chunks = (self.total_records + self.chunk_size - 1) // self.chunk_size
        self.logger.info(f"📊 Processing {self.total_records:,} records in {num_chunks} chunks of {self.chunk_size:,}")
        
        # Initialize results tracking
        processing_results = {
            'total_records': self.total_records,
            'chunk_size': self.chunk_size,
            'num_chunks': num_chunks,
            'chunks_processed': 0,
            'total_processing_time': 0,
            'chunk_results': [],
            'quality_metrics': self.quality_metrics,
            'success': True,
            'errors': []
        }
        
        # Process each chunk
        chunk_num = 1
        try:
            # Read file in chunks
            chunk_iter = pd.read_csv(self.input_file, chunksize=self.chunk_size)
            
            for chunk_df in chunk_iter:
                self.logger.info(f"🔄 Starting chunk {chunk_num}/{num_chunks}")
                
                # Process chunk
                chunk_results = self.process_chunk(chunk_df, chunk_num)
                processing_results['chunk_results'].append(chunk_results)
                processing_results['chunks_processed'] += 1
                
                # Update progress
                self.processed_records += chunk_results['records_in']
                progress_pct = (self.processed_records / self.total_records) * 100
                self.logger.info(f"  📈 Progress: {self.processed_records:,}/{self.total_records:,} ({progress_pct:.1f}%)")
                
                # Memory cleanup
                del chunk_df
                gc.collect()
                
                chunk_num += 1
                
        except Exception as e:
            self.logger.error(f"❌ Error during chunk processing: {e}")
            processing_results['errors'].append(str(e))
            processing_results['success'] = False
        
        # Calculate total processing time
        processing_results['total_processing_time'] = time.time() - self.start_time
        
        # Log final results
        self.logger.info("🎉 Memory-Efficient Processing Complete!")
        self.logger.info(f"📊 Total records processed: {self.processed_records:,}")
        self.logger.info(f"⏱️ Total processing time: {processing_results['total_processing_time']:.2f} seconds")
        self.logger.info(f"📈 Average time per chunk: {processing_results['total_processing_time']/num_chunks:.2f} seconds")
        self.logger.info(f"✅ Chunks completed successfully: {processing_results['chunks_processed']}/{num_chunks}")
        
        # Log quality metrics
        self.logger.info("📊 Final Quality Metrics:")
        self.logger.info(f"  • Total missing values: {self.quality_metrics['total_missing_values']:,}")
        self.logger.info(f"  • Total duplicates: {self.quality_metrics['total_duplicates']:,}")
        self.logger.info(f"  • Data type optimizations: {self.quality_metrics['data_type_optimizations']:,}")
        self.logger.info(f"  • Outliers detected: {self.quality_metrics['outliers_detected']:,}")
        
        return processing_results
    
    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive quality report.
        
        Args:
            results: Processing results
            
        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"../../outputs/memory_efficient_quality_report_{timestamp}.md"
        
        # Create output directory
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Memory-Efficient Data Quality Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Input File**: {self.input_file}\n")
            f.write(f"**Chunk Size**: {self.chunk_size:,}\n\n")
            
            f.write("## Processing Summary\n\n")
            f.write(f"- **Total Records**: {results['total_records']:,}\n")
            f.write(f"- **Chunks Processed**: {results['chunks_processed']}/{results['num_chunks']}\n")
            f.write(f"- **Total Processing Time**: {results['total_processing_time']:.2f} seconds\n")
            f.write(f"- **Average Time per Chunk**: {results['total_processing_time']/results['num_chunks']:.2f} seconds\n\n")
            
            f.write("## Quality Metrics\n\n")
            f.write(f"- **Total Missing Values**: {results['quality_metrics']['total_missing_values']:,}\n")
            f.write(f"- **Total Duplicates**: {results['quality_metrics']['total_duplicates']:,}\n")
            f.write(f"- **Data Type Optimizations**: {results['quality_metrics']['data_type_optimizations']:,}\n")
            f.write(f"- **Outliers Detected**: {results['quality_metrics']['outliers_detected']:,}\n\n")
            
            f.write("## Chunk-by-Chunk Results\n\n")
            for chunk_result in results['chunk_results']:
                f.write(f"### Chunk {chunk_result['chunk_num']}\n")
                f.write(f"- **Records**: {chunk_result['records_in']:,}\n")
                f.write(f"- **Processing Time**: {chunk_result['processing_time']:.2f}s\n")
                f.write(f"- **Missing Values**: {chunk_result['quality_metrics']['missing_values']:,}\n")
                f.write(f"- **Duplicates**: {chunk_result['quality_metrics']['duplicates']:,}\n")
                f.write(f"- **Outliers**: {chunk_result['quality_metrics']['outliers']:,}\n\n")
        
        self.logger.info(f"📄 Quality report generated: {report_path}")
        return report_path


def main():
    """Main function to run the memory-efficient processing pipeline."""
    print("🚀 Memory-Efficient Data Quality Pipeline")
    print("=" * 60)
    
    # Ask user for processing mode
    print("\n📊 Choose processing mode:")
    print("1. Test with sample data (100 books)")
    print("2. Process full dataset in memory-efficient chunks")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        chunk_size = 50  # Smaller chunks for testing
        input_file = "../../data/processed/final_books_2000_2020_en_enhanced_titles_sampled_100_20250904_210625.csv"
        print(f"\n🧪 Testing mode: Sample data with chunk size {chunk_size}")
    else:
        chunk_size = 1000  # Larger chunks for full dataset
        input_file = "../../data/processed/final_books_2000_2020_en_enhanced_20250904_215835.csv"
        print(f"\n🏭 Production mode: Full dataset with chunk size {chunk_size}")
    
    print(f"\n📊 Processing file: {input_file}")
    
    # Initialize and run processor
    processor = MemoryEfficientProcessor(
        input_file=input_file,
        chunk_size=chunk_size
    )
    
    # Run the processing
    results = processor.run_memory_efficient_processing()
    
    # Generate quality report
    report_path = processor.generate_quality_report(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 MEMORY-EFFICIENT PROCESSING RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Success: {results['success']}")
    print(f"📊 Total records: {results['total_records']:,}")
    print(f"📦 Chunks processed: {results['chunks_processed']}/{results['num_chunks']}")
    print(f"⏱️ Total time: {results['total_processing_time']:.2f} seconds")
    
    if results['errors']:
        print(f"❌ Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  • {error}")
    
    print(f"\n📊 Quality Metrics:")
    print(f"  • Missing values: {results['quality_metrics']['total_missing_values']:,}")
    print(f"  • Duplicates: {results['quality_metrics']['total_duplicates']:,}")
    print(f"  • Data optimizations: {results['quality_metrics']['data_type_optimizations']:,}")
    print(f"  • Outliers: {results['quality_metrics']['outliers_detected']:,}")
    
    print(f"\n📝 Detailed logs saved to: logs/memory_efficient/")
    print(f"📄 Quality report: {report_path}")


if __name__ == "__main__":
    main()
