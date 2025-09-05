#!/usr/bin/env python3
"""
Script to inspect all JSON.gz files and report column names and null counts.
"""

import json
import gzip
import pandas as pd
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_gz(file_path, sample_size=None):
    """
    Load JSON.gz file and return as DataFrame.
    
    Args:
        file_path: Path to the JSON.gz file
        sample_size: If provided, sample this many records for large files
        
    Returns:
        pandas DataFrame
    """
    logger.info(f"Loading {file_path}")
    
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on line {i+1}: {e}")
                continue
    
    if not data:
        logger.warning(f"No valid data found in {file_path}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} records from {file_path}")
    return df

def inspect_file(file_path):
    """
    Inspect a single JSON.gz file and return column info.
    
    Args:
        file_path: Path to the JSON.gz file
        
    Returns:
        dict with file info
    """
    file_name = os.path.basename(file_path)
    logger.info(f"Processing {file_name}")
    
    # First, try to load a small sample to check structure
    sample_df = load_json_gz(file_path, sample_size=1000)
    
    if sample_df.empty:
        return {
            'file': file_name,
            'total_records': 0,
            'columns': [],
            'null_counts': {},
            'error': 'No valid data found'
        }
    
    # Get column info from sample
    columns = list(sample_df.columns)
    null_counts_sample = sample_df.isnull().sum().to_dict()
    
    # If file seems manageable, load full dataset
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    
    if file_size_mb < 500:  # Less than 500MB, load full file
        logger.info(f"File size {file_size_mb:.1f}MB - loading full dataset")
        full_df = load_json_gz(file_path)
        total_records = len(full_df)
        null_counts = full_df.isnull().sum().to_dict()
    else:
        logger.info(f"File size {file_size_mb:.1f}MB - using sample for null counts")
        # Estimate null counts based on sample
        total_records = "Large file (>500MB) - using sample"
        null_counts = null_counts_sample
    
    return {
        'file': file_name,
        'total_records': total_records,
        'columns': columns,
        'null_counts': null_counts,
        'file_size_mb': file_size_mb
    }

def main():
    """Main function to inspect all JSON.gz files."""
    
    # Define the data directory
    data_dir = Path("/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/raw")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Find all JSON.gz files
    json_files = list(data_dir.glob("*.json.gz"))
    
    if not json_files:
        logger.error("No JSON.gz files found in data directory")
        return
    
    logger.info(f"Found {len(json_files)} JSON.gz files to inspect")
    
    # Process each file
    results = []
    for file_path in sorted(json_files):
        try:
            result = inspect_file(file_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            results.append({
                'file': os.path.basename(file_path),
                'error': str(e)
            })
    
    # Print results
    print("\n" + "="*80)
    print("JSON.GZ FILES INSPECTION REPORT")
    print("="*80)
    
    for result in results:
        print(f"\n📁 FILE: {result['file']}")
        print("-" * 60)
        
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            continue
        
        print(f"📊 Total Records: {result['total_records']}")
        print(f"💾 File Size: {result.get('file_size_mb', 'Unknown'):.1f} MB")
        print(f"📋 Columns ({len(result['columns'])}):")
        
        if result['columns']:
            for i, col in enumerate(result['columns'], 1):
                null_count = result['null_counts'].get(col, 0)
                null_pct = (null_count / result['total_records'] * 100) if isinstance(result['total_records'], int) else "N/A"
                print(f"  {i:2d}. {col:<30} | Nulls: {null_count:>8} ({null_pct:>5.1f}%)" if isinstance(null_pct, (int, float)) else f"  {i:2d}. {col:<30} | Nulls: {null_count:>8} ({null_pct})")
        else:
            print("  No columns found")
    
    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
