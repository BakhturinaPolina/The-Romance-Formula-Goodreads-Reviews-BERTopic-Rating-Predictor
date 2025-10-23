#!/usr/bin/env python3
"""
JSON to Parquet Converter for Anna's Archive

Converts Anna's Archive JSON.gz files to Parquet format for efficient querying.
Implements chunked processing and smart schema detection to handle large datasets
with limited memory.

Usage:
    python json_to_parquet.py --input-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
                             --output-dir ../../data/anna_archive/parquet/sample_10k/
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Iterator
import logging
from collections import defaultdict, Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaDetector:
    """Detects schema from JSON records with smart field selection."""
    
    def __init__(self, sample_size: int = 1000, field_threshold: float = 0.5):
        """
        Initialize schema detector.
        
        Args:
            sample_size: Number of records to sample for schema detection
            field_threshold: Minimum fraction of records that must contain a field
        """
        self.sample_size = sample_size
        self.field_threshold = field_threshold
        self.field_counts = Counter()
        self.sample_records = []
        self.processed_records = 0
    
    def add_record(self, record: Dict[str, Any]) -> None:
        """Add a record for schema analysis."""
        if len(self.sample_records) < self.sample_size:
            self.sample_records.append(record)
        
        # Count field presence
        self._count_fields(record)
        self.processed_records += 1
    
    def _count_fields(self, record: Dict[str, Any], prefix: str = "") -> None:
        """Recursively count field presence in nested records."""
        for key, value in record.items():
            field_name = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._count_fields(value, field_name)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Handle list of objects
                for i, item in enumerate(value[:3]):  # Sample first 3 items
                    self._count_fields(item, f"{field_name}[{i}]")
            else:
                # Count non-null values
                if value is not None and value != "":
                    self.field_counts[field_name] += 1
    
    def get_schema(self) -> List[str]:
        """Get list of fields that meet the threshold requirement."""
        if not self.field_counts:
            return []
        
        threshold_count = max(1, int(self.processed_records * self.field_threshold))
        schema_fields = [
            field for field, count in self.field_counts.items()
            if count >= threshold_count
        ]
        
        # Sort fields for consistent output
        schema_fields.sort()
        
        logger.info(f"Schema detection complete:")
        logger.info(f"  Records analyzed: {self.processed_records:,}")
        logger.info(f"  Fields found: {len(self.field_counts):,}")
        logger.info(f"  Fields selected: {len(schema_fields):,} (threshold: {self.field_threshold})")
        
        return schema_fields
    
    def get_field_stats(self) -> Dict[str, int]:
        """Get statistics about field presence."""
        return dict(self.field_counts.most_common(20))


class JSONToParquetConverter:
    """Converts JSON records to Parquet format with chunked processing."""
    
    def __init__(
        self,
        chunk_size: int = 10000,
        sample_size: int = 1000,
        field_threshold: float = 0.5
    ):
        """
        Initialize converter.
        
        Args:
            chunk_size: Number of records per Parquet file
            sample_size: Records to sample for schema detection
            field_threshold: Minimum field presence threshold
        """
        self.chunk_size = chunk_size
        self.schema_detector = SchemaDetector(sample_size, field_threshold)
        self.schema_fields = None
        self.output_dir = None
    
    def detect_schema(self, input_file: str) -> List[str]:
        """
        Detect schema from input file.
        
        Args:
            input_file: Path to input JSON.gz file
            
        Returns:
            List of field names to include in schema
        """
        logger.info("Detecting schema...")
        
        record_count = 0
        for record in self._read_json_records(input_file):
            self.schema_detector.add_record(record)
            record_count += 1
            
            if record_count % 1000 == 0:
                logger.info(f"Analyzed {record_count:,} records for schema")
        
        self.schema_fields = self.schema_detector.get_schema()
        
        # Log top fields
        top_fields = self.schema_detector.get_field_stats()
        logger.info("Top fields by frequency:")
        for field, count in list(top_fields.items())[:10]:
            logger.info(f"  {field}: {count:,} ({count/record_count*100:.1f}%)")
        
        return self.schema_fields
    
    def _read_json_records(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """Read JSON records from file."""
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
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise
    
    def _extract_field_value(self, record: Dict[str, Any], field_path: str) -> Any:
        """Extract value from nested record using dot notation."""
        try:
            parts = field_path.split('.')
            value = record
            
            for part in parts:
                if '[' in part and ']' in part:
                    # Handle array access like "authors[0]"
                    key, index = part.split('[')
                    index = int(index.rstrip(']'))
                    value = value[key][index]
                else:
                    value = value[part]
            
            return value
        except (KeyError, IndexError, TypeError):
            return None
    
    def _record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert record to flat dictionary using detected schema."""
        flat_record = {}
        
        for field in self.schema_fields:
            value = self._extract_field_value(record, field)
            
            # Convert complex types to strings
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif value is None:
                value = None
            
            flat_record[field] = value
        
        return flat_record
    
    def convert(
        self,
        input_file: str,
        output_dir: str,
        schema_fields: List[str] = None
    ) -> None:
        """
        Convert JSON file to Parquet format.
        
        Args:
            input_file: Path to input JSON.gz file
            output_dir: Output directory for Parquet files
            schema_fields: List of fields to include (if None, auto-detect)
        """
        if schema_fields:
            self.schema_fields = schema_fields
        elif not self.schema_fields:
            self.schema_fields = self.detect_schema(input_file)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting {input_file} to Parquet...")
        logger.info(f"Schema fields: {len(self.schema_fields)}")
        logger.info(f"Output directory: {output_dir}")
        
        # Process records in chunks
        chunk_records = []
        chunk_num = 0
        total_records = 0
        
        # Count total records for progress bar
        total_count = sum(1 for _ in self._read_json_records(input_file))
        logger.info(f"Total records to process: {total_count:,}")
        
        with tqdm(total=total_count, desc="Converting") as pbar:
            for record in self._read_json_records(input_file):
                # Convert record to flat dictionary
                flat_record = self._record_to_dict(record)
                chunk_records.append(flat_record)
                
                # Write chunk when full
                if len(chunk_records) >= self.chunk_size:
                    self._write_chunk(chunk_records, chunk_num)
                    total_records += len(chunk_records)
                    pbar.update(len(chunk_records))
                    chunk_records = []
                    chunk_num += 1
            
            # Write remaining records
            if chunk_records:
                self._write_chunk(chunk_records, chunk_num)
                total_records += len(chunk_records)
                pbar.update(len(chunk_records))
        
        logger.info(f"Conversion complete!")
        logger.info(f"Total records processed: {total_records:,}")
        logger.info(f"Parquet files created: {chunk_num + 1}")
        logger.info(f"Output directory: {output_dir}")
    
    def _write_chunk(self, records: List[Dict[str, Any]], chunk_num: int) -> None:
        """Write a chunk of records to Parquet file."""
        try:
            # Create DataFrame
            df = pd.DataFrame(records)
            
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df)
            
            # Write Parquet file
            output_file = self.output_dir / f"part-{chunk_num:05d}.parquet"
            pq.write_table(table, output_file, compression='snappy')
            
            logger.debug(f"Wrote chunk {chunk_num} with {len(records):,} records to {output_file}")
            
        except Exception as e:
            logger.error(f"Error writing chunk {chunk_num}: {e}")
            raise


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert Anna's Archive JSON.gz files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert sample file
  python json_to_parquet.py \\
    --input-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \\
    --output-dir ../../data/anna_archive/parquet/sample_10k/

  # Convert with custom chunk size
  python json_to_parquet.py \\
    --input-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \\
    --output-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --chunk-size 5000

  # Convert with custom schema detection
  python json_to_parquet.py \\
    --input-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \\
    --output-dir ../../data/anna_archive/parquet/sample_10k/ \\
    --sample-size 2000 \\
    --field-threshold 0.3
        """
    )
    
    parser.add_argument(
        '--input-file',
        required=True,
        help='Input JSON.gz file path'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for Parquet files'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Number of records per Parquet file (default: 10000)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Records to sample for schema detection (default: 1000)'
    )
    
    parser.add_argument(
        '--field-threshold',
        type=float,
        default=0.5,
        help='Minimum field presence threshold (default: 0.5)'
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
        # Create converter
        converter = JSONToParquetConverter(
            chunk_size=args.chunk_size,
            sample_size=args.sample_size,
            field_threshold=args.field_threshold
        )
        
        # Convert file
        converter.convert(
            input_file=args.input_file,
            output_dir=args.output_dir
        )
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
