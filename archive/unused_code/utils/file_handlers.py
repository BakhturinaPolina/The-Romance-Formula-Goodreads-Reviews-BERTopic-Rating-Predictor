"""
File handling utilities for the Romance Novel NLP Research Project.

This module provides utility functions for reading JSON files, writing CSV files,
and performing various file operations with proper error handling and logging.
"""

import json
import gzip
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional, Union
import logging
from tqdm import tqdm

from config.logging_config import get_logger

logger = get_logger("utils")


class JSONFileHandler:
    """Handler for reading and processing JSON files."""
    
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize the JSON file handler.
        
        Args:
            chunk_size: Number of records to process in each chunk
        """
        self.chunk_size = chunk_size
    
    def read_json_gz(self, file_path: Path, max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Read a gzipped JSON file line by line.
        
        Args:
            file_path: Path to the gzipped JSON file
            max_records: Maximum number of records to read (None for all)
            
        Yields:
            Dictionary representing each JSON record
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
            
        logger.info(f"Reading JSON file: {file_path}")
        
        records_read = 0
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc=f"Reading {file_path.name}")):
                    if max_records and records_read >= max_records:
                        break
                        
                    try:
                        record = json.loads(line.strip())
                        yield record
                        records_read += 1
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
            
        logger.info(f"Read {records_read} records from {file_path}")
    
    def read_json_gz_chunked(self, file_path: Path, max_records: Optional[int] = None) -> Iterator[List[Dict[str, Any]]]:
        """
        Read a gzipped JSON file in chunks.
        
        Args:
            file_path: Path to the gzipped JSON file
            max_records: Maximum number of records to read (None for all)
            
        Yields:
            List of dictionaries representing JSON records in each chunk
        """
        chunk = []
        records_read = 0
        
        for record in self.read_json_gz(file_path, max_records):
            chunk.append(record)
            records_read += 1
            
            if len(chunk) >= self.chunk_size:
                yield chunk
                chunk = []
                
        # Yield remaining records
        if chunk:
            yield chunk
    
    def count_records(self, file_path: Path) -> int:
        """
        Count the number of records in a gzipped JSON file.
        
        Args:
            file_path: Path to the gzipped JSON file
            
        Returns:
            Number of records in the file
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 0
            
        count = 0
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Counting records in {file_path.name}"):
                    try:
                        json.loads(line.strip())
                        count += 1
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Error counting records in {file_path}: {e}")
            return 0
            
        logger.info(f"Found {count} records in {file_path}")
        return count


class CSVFileHandler:
    """Handler for writing CSV files with proper formatting."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the CSV file handler.
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_dataframe(self, df: pd.DataFrame, filename: str, index: bool = False) -> Path:
        """
        Write a pandas DataFrame to CSV file.
        
        Args:
            df: DataFrame to write
            filename: Name of the output file
            index: Whether to include index in output
            
        Returns:
            Path to the written file
        """
        output_path = self.output_dir / filename
        
        try:
            df.to_csv(output_path, index=index, encoding='utf-8')
            logger.info(f"DataFrame written to {output_path} ({len(df)} rows, {len(df.columns)} columns)")
            return output_path
            
        except Exception as e:
            logger.error(f"Error writing DataFrame to {output_path}: {e}")
            raise
    
    def write_records(self, records: List[Dict[str, Any]], filename: str) -> Path:
        """
        Write a list of dictionaries to CSV file.
        
        Args:
            records: List of dictionaries to write
            filename: Name of the output file
            
        Returns:
            Path to the written file
        """
        if not records:
            logger.warning("No records to write")
            # Create empty file for consistency
            output_path = self.output_dir / filename
            output_path.touch()
            return output_path
            
        df = pd.DataFrame(records)
        return self.write_dataframe(df, filename)
    
    def append_records(self, records: List[Dict[str, Any]], filename: str, header: bool = False) -> Path:
        """
        Append records to an existing CSV file.
        
        Args:
            records: List of dictionaries to append
            filename: Name of the output file
            header: Whether to write header (only for new files)
            
        Returns:
            Path to the written file
        """
        output_path = self.output_dir / filename
        
        if not records:
            logger.warning("No records to append")
            return output_path
            
        df = pd.DataFrame(records)
        
        try:
            df.to_csv(output_path, mode='a', header=header, index=False, encoding='utf-8')
            logger.info(f"Appended {len(records)} records to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error appending records to {output_path}: {e}")
            raise


class DataValidator:
    """Validator for data quality and consistency."""
    
    def __init__(self, required_fields: List[str], optional_fields: List[str] = None):
        """
        Initialize the data validator.
        
        Args:
            required_fields: List of required field names
            optional_fields: List of optional field names
        """
        self.required_fields = required_fields
        self.optional_fields = optional_fields or []
    
    def validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single record.
        
        Args:
            record: Dictionary representing a record
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "missing_required": [],
            "missing_optional": [],
            "invalid_types": [],
            "warnings": []
        }
        
        # Check required fields
        for field in self.required_fields:
            if field not in record or record[field] is None or record[field] == "":
                validation_result["missing_required"].append(field)
                validation_result["is_valid"] = False
        
        # Check optional fields
        for field in self.optional_fields:
            if field not in record or record[field] is None or record[field] == "":
                validation_result["missing_optional"].append(field)
        
        return validation_result
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "total_records": len(df),
            "valid_records": 0,
            "invalid_records": 0,
            "missing_required_counts": {},
            "missing_optional_counts": {},
            "data_types": {},
            "unique_counts": {}
        }
        
        # Check for required columns
        for field in self.required_fields:
            if field not in df.columns:
                validation_result["missing_required_counts"][field] = len(df)
                validation_result["invalid_records"] = len(df)
            else:
                missing_count = df[field].isna().sum()
                validation_result["missing_required_counts"][field] = missing_count
                if missing_count > 0:
                    validation_result["invalid_records"] += missing_count
        
        # Check optional columns
        for field in self.optional_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                validation_result["missing_optional_counts"][field] = missing_count
        
        # Data types
        for col in df.columns:
            validation_result["data_types"][col] = str(df[col].dtype)
            validation_result["unique_counts"][col] = df[col].nunique()
        
        validation_result["valid_records"] = validation_result["total_records"] - validation_result["invalid_records"]
        
        return validation_result


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON text with error handling.
    
    Args:
        text: JSON text to parse
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in megabytes
    """
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024 * 1024)


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_extension(file_path: Path) -> str:
    """
    Get file extension (handles multiple extensions like .json.gz).
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension(s)
    """
    return ''.join(file_path.suffixes)


def is_gzipped(file_path: Path) -> bool:
    """
    Check if a file is gzipped.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is gzipped
    """
    return file_path.suffix == '.gz'
