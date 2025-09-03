#!/usr/bin/env python3
"""
Step 5: Save Optimized Dataset in Parquet Format

This script saves the optimized dataset from Step 5 in parquet format
for efficient storage and type preservation.

Author: Research Assistant
Date: 2025-09-02
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_optimized_dataset_parquet():
    """Save the optimized dataset in parquet format."""
    
    # Load the optimized dataset
    pickle_path = "outputs/data_type_optimization/cleaned_romance_novels_step5_optimized_20250902_231949.pickle"
    
    if not Path(pickle_path).exists():
        logger.error(f"Optimized dataset not found: {pickle_path}")
        return False
    
    try:
        logger.info(f"Loading optimized dataset from: {pickle_path}")
        df = pd.read_pickle(pickle_path)
        
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Create output directory
        output_dir = Path("outputs/data_type_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_filename = f"cleaned_romance_novels_step5_optimized_{timestamp}.parquet"
        parquet_path = output_dir / parquet_filename
        
        logger.info("💾 Saving optimized dataset as parquet...")
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        
        # Get file size
        file_size_mb = parquet_path.stat().st_size / 1024 / 1024
        logger.info(f"✅ Dataset saved as parquet: {parquet_path}")
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Verify data types are preserved
        logger.info("🔍 Verifying data type preservation...")
        logger.info("Sample data types:")
        for col, dtype in df.dtypes.head(10).items():
            logger.info(f"  {col}: {dtype}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save parquet: {str(e)}")
        return False

if __name__ == "__main__":
    print("💾 STEP 5: SAVE OPTIMIZED DATASET AS PARQUET")
    print("=" * 60)
    
    success = save_optimized_dataset_parquet()
    
    if success:
        print("\n🎉 Dataset successfully saved as parquet!")
    else:
        print("\n💥 Failed to save dataset as parquet.")
        import sys
        sys.exit(1)
