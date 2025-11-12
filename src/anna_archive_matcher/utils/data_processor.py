"""
Data processing utilities for Anna's Archive datasets
Based on the DuckDB analysis environment from the Data Science Starter Kit
"""

import gzip
import json
import pandas as pd
import duckdb
from pathlib import Path
import logging
from typing import Dict, List, Optional
import zstandard as zstd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AnnaArchiveDataProcessor:
    """
    Process Anna's Archive datasets from raw files to Parquet format
    """
    
    def __init__(self, data_dir: str = "data", chunk_size: int = 10 * 1024 * 1024):
        """
        Initialize the data processor
        
        Args:
            data_dir: Path to data directory
            chunk_size: Chunk size for processing (default 10MB)
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.conn = duckdb.connect()
        
        # Set up DuckDB configuration
        self.conn.execute("SET memory_limit='28GB'")
        self.conn.execute("SET threads=4")
        
        logger.info(f"DataProcessor initialized with chunk size: {chunk_size}")
    
    def process_elasticsearch_data(self) -> None:
        """
        Process Elasticsearch .gz files to Parquet format
        """
        logger.info("Processing Elasticsearch data...")
        
        input_dir = self.data_dir / "elasticsearch"
        output_dir = self.data_dir / "elasticsearchF"
        output_dir.mkdir(exist_ok=True)
        
        gz_files = list(input_dir.glob("*.gz"))
        logger.info(f"Found {len(gz_files)} .gz files to process")
        
        for gz_file in tqdm(gz_files, desc="Processing Elasticsearch files"):
            self._process_gz_file(gz_file, output_dir, "elasticsearch")
    
    def process_aac_data(self) -> None:
        """
        Process AAC .zst files to Parquet format
        """
        logger.info("Processing AAC data...")
        
        input_dir = self.data_dir / "aac"
        output_dir = self.data_dir / "aacF"
        output_dir.mkdir(exist_ok=True)
        
        zst_files = list(input_dir.glob("*.zst"))
        logger.info(f"Found {len(zst_files)} .zst files to process")
        
        for zst_file in tqdm(zst_files, desc="Processing AAC files"):
            self._process_zst_file(zst_file, output_dir, "aac")
    
    def process_mariadb_data(self) -> None:
        """
        Process MariaDB .gz files to Parquet format
        """
        logger.info("Processing MariaDB data...")
        
        input_dir = self.data_dir / "mariadb"
        output_dir = self.data_dir / "mariadbF"
        output_dir.mkdir(exist_ok=True)
        
        gz_files = list(input_dir.glob("*.gz"))
        logger.info(f"Found {len(gz_files)} .gz files to process")
        
        for gz_file in tqdm(gz_files, desc="Processing MariaDB files"):
            self._process_gz_file(gz_file, output_dir, "mariadb")
    
    def _process_gz_file(self, gz_file: Path, output_dir: Path, dataset_type: str) -> None:
        """
        Process a single .gz file
        """
        logger.info(f"Processing {gz_file.name}")
        
        records = []
        schema_analyzer = SchemaAnalyzer()
        
        try:
            with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 10000 == 0 and line_num > 0:
                        logger.info(f"Processed {line_num} lines from {gz_file.name}")
                    
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                        schema_analyzer.analyze_record(record)
                        
                        # Process in chunks
                        if len(records) >= 10000:
                            self._process_chunk(records, gz_file, output_dir, dataset_type)
                            records = []
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in {gz_file.name} line {line_num}: {e}")
                        continue
                
                # Process remaining records
                if records:
                    self._process_chunk(records, gz_file, output_dir, dataset_type)
        
        except Exception as e:
            logger.error(f"Error processing {gz_file.name}: {e}")
    
    def _process_zst_file(self, zst_file: Path, output_dir: Path, dataset_type: str) -> None:
        """
        Process a single .zst file
        """
        logger.info(f"Processing {zst_file.name}")
        
        records = []
        schema_analyzer = SchemaAnalyzer()
        
        try:
            with open(zst_file, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text_reader = reader.read().decode('utf-8')
                    
                    for line_num, line in enumerate(text_reader.splitlines()):
                        if line_num % 10000 == 0 and line_num > 0:
                            logger.info(f"Processed {line_num} lines from {zst_file.name}")
                        
                        try:
                            record = json.loads(line.strip())
                            records.append(record)
                            schema_analyzer.analyze_record(record)
                            
                            # Process in chunks
                            if len(records) >= 10000:
                                self._process_chunk(records, zst_file, output_dir, dataset_type)
                                records = []
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON decode error in {zst_file.name} line {line_num}: {e}")
                            continue
                    
                    # Process remaining records
                    if records:
                        self._process_chunk(records, zst_file, output_dir, dataset_type)
        
        except Exception as e:
            logger.error(f"Error processing {zst_file.name}: {e}")
    
    def _process_chunk(self, records: List[Dict], source_file: Path, 
                      output_dir: Path, dataset_type: str) -> None:
        """
        Process a chunk of records
        """
        if not records:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Create output filename
        output_file = output_dir / f"{source_file.stem}.parquet"
        
        # Save as Parquet
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(records)} records to {output_file}")


class SchemaAnalyzer:
    """
    Analyze JSON schema for efficient column creation
    """
    
    def __init__(self, sample_size: int = 1000, threshold: float = 0.5):
        """
        Initialize schema analyzer
        
        Args:
            sample_size: Number of records to sample for schema analysis
            threshold: Minimum frequency threshold for column inclusion
        """
        self.sample_size = sample_size
        self.threshold = threshold
        self.key_frequencies = {}
        self.sample_count = 0
    
    def analyze_record(self, record: Dict) -> None:
        """
        Analyze a single record for schema information
        """
        if self.sample_count >= self.sample_size:
            return
        
        self._analyze_keys(record)
        self.sample_count += 1
    
    def _analyze_keys(self, record: Dict, prefix: str = "") -> None:
        """
        Recursively analyze keys in the record
        """
        for key, value in record.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if full_key not in self.key_frequencies:
                self.key_frequencies[full_key] = 0
            
            self.key_frequencies[full_key] += 1
            
            # Recursively analyze nested objects
            if isinstance(value, dict):
                self._analyze_keys(value, full_key)
    
    def get_schema(self) -> Dict[str, float]:
        """
        Get the analyzed schema with frequencies
        """
        return {
            key: freq / self.sample_count 
            for key, freq in self.key_frequencies.items()
            if freq / self.sample_count >= self.threshold
        }
