#!/usr/bin/env python3
"""
JSON Structure Inspector for Goodreads Romance Dataset.

This script analyzes the structure of all JSON files in the Goodreads dataset,
providing detailed information about fields, data types, missing values,
and sample records for each file.
"""

import json
import gzip
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import DATA_FILES, OUTPUT_FILES
from config.logging_config import setup_component_logging

logger = setup_component_logging("data_exploration")


class JSONStructureInspector:
    """Class to inspect and analyze JSON file structures."""
    
    def __init__(self, data_files: Dict[str, Path]):
        """
        Initialize the inspector with data file paths.
        
        Args:
            data_files: Dictionary mapping file names to file paths
        """
        self.data_files = data_files
        self.structure_info = {}
        self.sample_records = {}
        
    def inspect_file(self, file_name: str, file_path: Path, max_records: int = 1000) -> Dict[str, Any]:
        """
        Inspect a single JSON file and extract structure information.
        
        Args:
            file_name: Name of the file
            file_path: Path to the file
            max_records: Maximum number of records to process
            
        Returns:
            Dictionary containing structure information
        """
        logger.info(f"Inspecting file: {file_name}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {}
            
        structure_info = {
            "file_name": file_name,
            "file_path": str(file_path),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "total_records": 0,
            "fields": {},
            "sample_records": [],
            "field_statistics": {},
            "data_quality": {}
        }
        
        try:
            # Open and read the file
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                records_processed = 0
                field_types = defaultdict(set)
                field_lengths = defaultdict(list)
                field_values = defaultdict(set)
                missing_values = defaultdict(int)
                
                for line_num, line in enumerate(f):
                    if records_processed >= max_records:
                        break
                        
                    try:
                        record = json.loads(line.strip())
                        records_processed += 1
                        
                        # Store sample records
                        if records_processed <= 5:
                            structure_info["sample_records"].append(record)
                        
                        # Analyze fields
                        for field_name, field_value in record.items():
                            # Track field types
                            field_types[field_name].add(type(field_value).__name__)
                            
                            # Track field lengths for strings
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
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
                        
            # Calculate statistics
            structure_info["total_records"] = records_processed
            
            for field_name in field_types:
                structure_info["fields"][field_name] = {
                    "types": list(field_types[field_name]),
                    "missing_count": missing_values[field_name],
                    "missing_percentage": (missing_values[field_name] / records_processed) * 100,
                    "unique_values_count": len(field_values[field_name]),
                    "sample_values": list(field_values[field_name])[:10]
                }
                
                # Add length statistics for string/list/dict fields
                if field_lengths[field_name]:
                    lengths = field_lengths[field_name]
                    structure_info["field_statistics"][field_name] = {
                        "min_length": min(lengths),
                        "max_length": max(lengths),
                        "avg_length": sum(lengths) / len(lengths),
                        "median_length": sorted(lengths)[len(lengths)//2]
                    }
            
            # Overall data quality metrics
            structure_info["data_quality"] = {
                "completeness": (records_processed - sum(missing_values.values())) / (records_processed * len(field_types)),
                "field_count": len(field_types),
                "records_processed": records_processed
            }
            
            logger.info(f"Processed {records_processed} records from {file_name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            structure_info["error"] = str(e)
            
        return structure_info
    
    def inspect_all_files(self, max_records_per_file: int = 1000) -> Dict[str, Any]:
        """
        Inspect all JSON files in the dataset.
        
        Args:
            max_records_per_file: Maximum records to process per file
            
        Returns:
            Dictionary containing structure information for all files
        """
        logger.info("Starting inspection of all JSON files")
        
        all_structure_info = {
            "summary": {
                "total_files": len(self.data_files),
                "files_processed": 0,
                "total_size_mb": 0,
                "total_records": 0
            },
            "files": {}
        }
        
        for file_name, file_path in self.data_files.items():
            logger.info(f"Processing file: {file_name}")
            
            structure_info = self.inspect_file(file_name, file_path, max_records_per_file)
            
            if structure_info:
                all_structure_info["files"][file_name] = structure_info
                all_structure_info["summary"]["files_processed"] += 1
                all_structure_info["summary"]["total_size_mb"] += structure_info.get("file_size_mb", 0)
                all_structure_info["summary"]["total_records"] += structure_info.get("total_records", 0)
        
        logger.info(f"Inspection completed. Processed {all_structure_info['summary']['files_processed']} files")
        
        return all_structure_info
    
    def generate_report(self, structure_info: Dict[str, Any], output_path: Path) -> None:
        """
        Generate a comprehensive report of the JSON structure analysis.
        
        Args:
            structure_info: Structure information from inspection
            output_path: Path to save the report
        """
        logger.info(f"Generating report to {output_path}")
        
        # Create HTML report
        html_content = self._create_html_report(structure_info)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Report saved to {output_path}")
    
    def _create_html_report(self, structure_info: Dict[str, Any]) -> str:
        """Create HTML report content."""
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Goodreads JSON Structure Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .file-section { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .field-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                .field-table th, .field-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .field-table th { background-color: #f2f2f2; }
                .sample-record { background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 3px; }
                .error { color: red; }
                .warning { color: orange; }
                .success { color: green; }
            </style>
        </head>
        <body>
            <h1>Goodreads JSON Structure Analysis Report</h1>
        """
        
        # Summary section
        summary = structure_info["summary"]
        html += f"""
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Files:</strong> {summary['total_files']}</p>
                <p><strong>Files Processed:</strong> {summary['files_processed']}</p>
                <p><strong>Total Size:</strong> {summary['total_size_mb']:.2f} MB</p>
                <p><strong>Total Records:</strong> {summary['total_records']:,}</p>
            </div>
        """
        
        # File sections
        for file_name, file_info in structure_info["files"].items():
            html += f"""
                <div class="file-section">
                    <h2>{file_name}</h2>
                    <p><strong>File Size:</strong> {file_info['file_size_mb']:.2f} MB</p>
                    <p><strong>Records Processed:</strong> {file_info['total_records']:,}</p>
                    <p><strong>Fields:</strong> {len(file_info['fields'])}</p>
            """
            
            if "error" in file_info:
                html += f'<p class="error"><strong>Error:</strong> {file_info["error"]}</p>'
            else:
                # Fields table
                html += """
                    <h3>Fields Analysis</h3>
                    <table class="field-table">
                        <tr>
                            <th>Field Name</th>
                            <th>Types</th>
                            <th>Missing %</th>
                            <th>Unique Values</th>
                            <th>Sample Values</th>
                        </tr>
                """
                
                for field_name, field_data in file_info["fields"].items():
                    missing_class = "error" if field_data["missing_percentage"] > 50 else "success"
                    html += f"""
                        <tr>
                            <td>{field_name}</td>
                            <td>{', '.join(field_data['types'])}</td>
                            <td class="{missing_class}">{field_data['missing_percentage']:.1f}%</td>
                            <td>{field_data['unique_values_count']}</td>
                            <td>{', '.join(field_data['sample_values'][:3])}</td>
                        </tr>
                    """
                
                html += "</table>"
                
                # Sample records
                if file_info["sample_records"]:
                    html += "<h3>Sample Records</h3>"
                    for i, record in enumerate(file_info["sample_records"][:2]):  # Limit to 2 samples
                        html += f"""
                            <div class="sample-record">
                                <h4>Record {i+1}</h4>
                                <pre>{json.dumps(record, indent=2)[:500]}{'...' if len(json.dumps(record)) > 500 else ''}</pre>
                            </div>
                        """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    """Main function to run the JSON structure inspection."""
    logger.info("Starting JSON structure inspection")
    
    # Initialize inspector
    inspector = JSONStructureInspector(DATA_FILES)
    
    # Inspect all files
    structure_info = inspector.inspect_all_files(max_records_per_file=1000)
    
    # Generate report
    report_path = OUTPUT_FILES["data_quality"].parent / "json_structure_analysis.html"
    inspector.generate_report(structure_info, report_path)
    
    # Save structure info as JSON
    json_report_path = OUTPUT_FILES["data_quality"].parent / "json_structure_analysis.json"
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(structure_info, f, indent=2, default=str)
    
    logger.info("JSON structure inspection completed")
    logger.info(f"HTML report: {report_path}")
    logger.info(f"JSON report: {json_report_path}")


if __name__ == "__main__":
    main()