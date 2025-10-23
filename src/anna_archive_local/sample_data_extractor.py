#!/usr/bin/env python3
"""
Sample Data Extractor for Anna's Archive

Extracts a sample of records from Anna's Archive JSON.gz files for testing
and development purposes. This allows working with smaller datasets before
processing the full 500GB+ dataset.

Usage:
    python sample_data_extractor.py --input-dir ../../data/anna_archive/elasticsearch/ \
                                   --output-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
                                   --sample-size 10000
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Iterator, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_json_records(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Read JSON records from a .json.gz file.
    
    Args:
        file_path: Path to the .json.gz file
        
    Yields:
        Dict containing parsed JSON records
    """
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    record = json.loads(line)
                    yield record
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        raise


def extract_sample(
    input_dir: str,
    output_file: str,
    sample_size: int,
    max_files: int = None
) -> None:
    """
    Extract a sample of records from JSON.gz files in input directory.
    
    Args:
        input_dir: Directory containing .json.gz files
        output_file: Output file path for sample
        sample_size: Number of records to extract
        max_files: Maximum number of files to process (None for all)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all .json.gz files
    json_files = list(input_path.glob("*.json.gz"))
    if not json_files:
        raise FileNotFoundError(f"No .json.gz files found in {input_dir}")
    
    # Sort files for consistent sampling
    json_files.sort()
    
    # Limit number of files if specified
    if max_files:
        json_files = json_files[:max_files]
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    logger.info(f"Target sample size: {sample_size:,} records")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    records_extracted = 0
    files_processed = 0
    
    try:
        with gzip.open(output_file, 'wt', encoding='utf-8') as out_f:
            for file_path in json_files:
                if records_extracted >= sample_size:
                    break
                    
                logger.info(f"Processing {file_path.name}...")
                files_processed += 1
                
                file_records = 0
                for record in read_json_records(str(file_path)):
                    if records_extracted >= sample_size:
                        break
                    
                    # Write record to output
                    json.dump(record, out_f, ensure_ascii=False, separators=(',', ':'))
                    out_f.write('\n')
                    
                    records_extracted += 1
                    file_records += 1
                    
                    # Progress update every 1000 records
                    if records_extracted % 1000 == 0:
                        logger.info(f"Extracted {records_extracted:,}/{sample_size:,} records")
                
                logger.info(f"Extracted {file_records:,} records from {file_path.name}")
        
        logger.info(f"Sample extraction complete!")
        logger.info(f"Records extracted: {records_extracted:,}")
        logger.info(f"Files processed: {files_processed}")
        logger.info(f"Output file: {output_file}")
        
        # Verify output file
        output_size = Path(output_file).stat().st_size
        logger.info(f"Output file size: {output_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        # Clean up partial output file
        if Path(output_file).exists():
            Path(output_file).unlink()
        raise


def analyze_sample(output_file: str) -> None:
    """
    Analyze the extracted sample to show basic statistics.
    
    Args:
        output_file: Path to the sample file
    """
    logger.info("Analyzing sample...")
    
    total_records = 0
    title_count = 0
    author_count = 0
    md5_count = 0
    
    try:
        for record in read_json_records(output_file):
            total_records += 1
            
            # Check for common fields
            source = record.get('_source', {})
            file_data = source.get('file_unified_data', {})
            
            if file_data.get('title', {}).get('best'):
                title_count += 1
            if file_data.get('author', {}).get('best'):
                author_count += 1
            if file_data.get('identifiers_unified', {}).get('md5'):
                md5_count += 1
        
        logger.info(f"Sample analysis:")
        logger.info(f"  Total records: {total_records:,}")
        logger.info(f"  Records with titles: {title_count:,} ({title_count/total_records*100:.1f}%)")
        logger.info(f"  Records with authors: {author_count:,} ({author_count/total_records*100:.1f}%)")
        logger.info(f"  Records with MD5: {md5_count:,} ({md5_count/total_records*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error analyzing sample: {e}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract sample records from Anna's Archive JSON.gz files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 10,000 records for development
  python sample_data_extractor.py \\
    --input-dir ../../data/anna_archive/elasticsearch/ \\
    --output-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \\
    --sample-size 10000

  # Extract 1,000 records for quick testing
  python sample_data_extractor.py \\
    --input-dir ../../data/anna_archive/elasticsearch/ \\
    --output-file ../../data/anna_archive/elasticsearch/sample_1k.json.gz \\
    --sample-size 1000

  # Extract from first 5 files only
  python sample_data_extractor.py \\
    --input-dir ../../data/anna_archive/elasticsearch/ \\
    --output-file ../../data/anna_archive/elasticsearch/sample_5k.json.gz \\
    --sample-size 5000 \\
    --max-files 5
        """
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing .json.gz files'
    )
    
    parser.add_argument(
        '--output-file',
        required=True,
        help='Output file path for sample (.json.gz)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        required=True,
        help='Number of records to extract'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process (default: all)'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze the extracted sample'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Extract sample
        extract_sample(
            input_dir=args.input_dir,
            output_file=args.output_file,
            sample_size=args.sample_size,
            max_files=args.max_files
        )
        
        # Analyze if requested
        if args.analyze:
            analyze_sample(args.output_file)
            
    except Exception as e:
        logger.error(f"Sample extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
