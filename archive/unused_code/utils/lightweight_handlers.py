"""
Lightweight file handling utilities for the Romance Novel NLP Research Project.

This module provides utility functions for reading JSON files and performing
various file operations with proper error handling and logging.
Uses only core Python libraries to avoid dependency issues.
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional, Union
import logging
from datetime import datetime
from collections import defaultdict, Counter

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightweightJSONHandler:
    """Lightweight handler for reading and processing JSON files."""
    
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
                for line_num, line in enumerate(f):
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
                for line in f:
                    try:
                        json.loads(line.strip())
                        count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error counting records in {file_path}: {e}")
            return 0
            
        return count
    
    def sample_records(self, file_path: Path, sample_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Sample records from a gzipped JSON file.
        
        Args:
            file_path: Path to the gzipped JSON file
            sample_size: Number of records to sample
            
        Returns:
            List of sampled records
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
            
        records = []
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if len(records) >= sample_size:
                        break
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error sampling records from {file_path}: {e}")
            return []
            
        return records


class LightweightSchemaInspector:
    """Lightweight schema inspector for JSON files."""
    
    def __init__(self):
        """Initialize the schema inspector."""
        self.handler = LightweightJSONHandler()
    
    def inspect_file_schema(self, file_path: Path, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Inspect the schema of a JSON file.
        
        Args:
            file_path: Path to the JSON file
            sample_size: Number of records to sample for analysis
            
        Returns:
            Dictionary containing schema information
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {}
            
        logger.info(f"Inspecting schema for: {file_path}")
        
        # Sample records
        records = self.handler.sample_records(file_path, sample_size)
        if not records:
            logger.warning(f"No valid records found in {file_path}")
            return {}
            
        # Analyze schema
        field_types = defaultdict(set)
        field_values = defaultdict(set)
        missing_values = defaultdict(int)
        field_lengths = defaultdict(list)
        
        for record in records:
            for field_name, field_value in record.items():
                # Track field types
                field_types[field_name].add(type(field_value).__name__)
                
                # Track field lengths
                if isinstance(field_value, str):
                    field_lengths[field_name].append(len(field_value))
                elif isinstance(field_value, list):
                    field_lengths[field_name].append(len(field_value))
                elif isinstance(field_value, dict):
                    field_lengths[field_name].append(len(field_value))
                
                # Track unique values (limit to avoid memory issues)
                if isinstance(field_value, (str, int, float)) and len(field_values[field_name]) < 100:
                    field_values[field_name].add(str(field_value))
                
                # Track missing values
                if field_value is None or field_value == "":
                    missing_values[field_name] += 1
        
        # Compile schema information
        schema_info = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "records_analyzed": len(records),
            "fields": {},
            "sample_records": records[:3] if records else []
        }
        
        for field_name in field_types:
            schema_info["fields"][field_name] = {
                "types": list(field_types[field_name]),
                "missing_count": missing_values[field_name],
                "missing_percentage": (missing_values[field_name] / len(records)) * 100,
                "unique_values_count": len(field_values[field_name]),
                "sample_values": list(field_values[field_name])[:10]
            }
            
            # Add length statistics
            if field_lengths[field_name]:
                lengths = field_lengths[field_name]
                schema_info["fields"][field_name]["length_stats"] = {
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                    "avg_length": sum(lengths) / len(lengths)
                }
        
        return schema_info
    
    def inspect_multiple_files(self, file_paths: List[Path], sample_size: int = 1000) -> Dict[str, Any]:
        """
        Inspect schema for multiple files.
        
        Args:
            file_paths: List of file paths to inspect
            sample_size: Number of records to sample per file
            
        Returns:
            Dictionary containing schema information for all files
        """
        all_schemas = {
            "summary": {
                "total_files": len(file_paths),
                "files_processed": 0,
                "total_size_mb": 0
            },
            "files": {}
        }
        
        for file_path in file_paths:
            if file_path.exists():
                schema_info = self.inspect_file_schema(file_path, sample_size)
                if schema_info:
                    all_schemas["files"][file_path.name] = schema_info
                    all_schemas["summary"]["files_processed"] += 1
                    all_schemas["summary"]["total_size_mb"] += schema_info.get("file_size_mb", 0)
        
        return all_schemas


def create_log_directories(project_root: Path) -> Dict[str, Path]:
    """
    Create log directories for exploration.
    
    Args:
        project_root: Project root directory
        
    Returns:
        Dictionary of created directory paths
    """
    directories = {
        "log_full": project_root / "logs" / "exploration" / "full",
        "log_summary": project_root / "logs" / "exploration" / "summary",
        "manifests": project_root / "data" / "intermediate" / "manifests",
        "schemas": project_root / "data" / "intermediate" / "schemas",
        "samples": project_root / "data" / "intermediate" / "samples"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def save_json_artifact(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save data as JSON artifact.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved artifact: {file_path}")
    except Exception as e:
        logger.error(f"Error saving artifact {file_path}: {e}")


def save_log_entry(log_file: Path, content: str) -> None:
    """
    Save a log entry to file.
    
    Args:
        log_file: Path to log file
        content: Content to write
    """
    try:
        with open(log_file, 'w') as f:
            f.write(content)
        logger.info(f"Saved log: {log_file}")
    except Exception as e:
        logger.error(f"Error saving log {log_file}: {e}")
