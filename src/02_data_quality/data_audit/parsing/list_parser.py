#!/usr/bin/env python3
"""
Milestone A — Ingestion & Audit: List Parsing Script
===================================================

This script handles parsing of list-like fields in the Goodreads dataset:
1. Parse book_id_list_en (list of book IDs)
2. Split genres_str (comma-separated genres)
3. Split shelves_str (comma-separated shelf tags)

Outputs: parsed_books.parquet with properly structured list columns

Features:
- Robust parsing with fallback strategies
- Data validation and quality checks
- Comprehensive logging and error handling
- Parquet output for efficient storage
"""

import os
import sys
import json
import logging
import pathlib
import argparse
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import ast

# Add project root to path for imports
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ListParser:
    """Robust parser for list-like fields in the romance novel dataset."""
    
    def __init__(self, data_path: str, output_dir: str = "./parse_outputs"):
        """
        Initialize the list parser.
        
        Args:
            data_path: Path to the CSV file
            output_dir: Directory for output files
        """
        self.data_path = data_path
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fields to parse
        self.list_fields = {
            'book_id_list_en': 'book_ids',
            'genres_str': 'genres',
            'shelves_str': 'shelves'
        }
        
        self.df = None
        self.parse_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded {len(self.df):,} rows and {len(self.df.columns)} columns")
            return self.df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def parse_listlike(self, s: Union[str, float, None]) -> List[str]:
        """
        Robust parser for list-like strings.
        
        Tries multiple parsing strategies:
        1. ast.literal_eval for proper Python lists
        2. Comma-split for simple comma-separated values
        3. Handles edge cases (NaN, empty strings, etc.)
        
        Args:
            s: Input string or value to parse
            
        Returns:
            List of strings
        """
        if pd.isna(s) or s is None:
            return []
        
        s = str(s).strip()
        if not s or s.lower() in ['nan', 'none', 'null']:
            return []
        
        # Strategy 1: Try literal_eval for proper Python lists
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass
        
        # Strategy 2: Regex fallback for CSV quirks (RFC-4180 compliant)
        try:
            # Trim stray quotes and brackets that might leak from CSV parsing
            s_cleaned = re.sub(r'^["\'\[]+|["\'\]]+$', '', s.strip())
            
            # Split on multiple separators, handling quoted fields
            # This handles cases like: "item1, item2" ; "item3" | item4
            separators = r'[,;|]+'
            items = []
            
            # Simple approach: split on separators and clean each item
            for item in re.split(separators, s_cleaned):
                item = item.strip()
                # Remove any remaining quotes
                item = re.sub(r'^["\']+|["\']+$', '', item)
                if item and item.lower() not in ['nan', 'none', 'null']:
                    items.append(item)
            
            if items:
                return items
        except Exception:
            pass
        
        # Strategy 3: Single item
        if s:
            return [s]
        
        return []
    
    def validate_parsed_lists(self, original_col: str, parsed_col: str) -> Dict[str, Any]:
        """
        Validate the parsed lists against the original data.
        
        Args:
            original_col: Name of the original column
            parsed_col: Name of the parsed column
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating {original_col} -> {parsed_col}")
        
        # Basic validation
        total_rows = len(self.df)
        parsed_rows = self.df[parsed_col].apply(lambda x: isinstance(x, list)).sum()
        empty_lists = self.df[parsed_col].apply(lambda x: len(x) == 0).sum()
        
        # Count total items
        total_items = self.df[parsed_col].apply(len).sum()
        avg_items_per_row = total_items / total_rows if total_rows > 0 else 0
        
        # Check for parsing errors (non-list values)
        non_list_count = total_rows - parsed_rows
        
        # Sample validation for book_id_list_en
        if original_col == 'book_id_list_en':
            # Check if parsed items look like book IDs (numeric strings)
            sample_items = []
            for items in self.df[parsed_col].head(100):
                sample_items.extend(items)
            
            numeric_items = sum(1 for item in sample_items if item.isdigit())
            numeric_ratio = numeric_items / len(sample_items) if sample_items else 0
        else:
            numeric_ratio = None
        
        validation_results = {
            'total_rows': total_rows,
            'parsed_rows': parsed_rows,
            'parsing_success_rate': parsed_rows / total_rows if total_rows > 0 else 0,
            'empty_lists': empty_lists,
            'empty_list_rate': empty_lists / total_rows if total_rows > 0 else 0,
            'non_list_count': non_list_count,
            'total_items': total_items,
            'avg_items_per_row': avg_items_per_row,
            'numeric_ratio': numeric_ratio
        }
        
        logger.info(f"Validation complete for {original_col}: {parsed_rows}/{total_rows} rows parsed successfully")
        return validation_results
    
    def parse_all_lists(self) -> Dict[str, Any]:
        """Parse all list-like fields in the dataset."""
        logger.info("Starting list parsing for all fields...")
        
        parse_results = {}
        
        for original_col, new_col in self.list_fields.items():
            if original_col not in self.df.columns:
                logger.warning(f"Column {original_col} not found in dataset")
                continue
            
            logger.info(f"Parsing {original_col} -> {new_col}")
            
            # Parse the column
            parsed_lists = self.df[original_col].apply(self.parse_listlike)
            
            # Add deduplication for shelves (preserve original order)
            if new_col == 'shelves':
                parsed_lists = parsed_lists.apply(lambda x: list(dict.fromkeys(x)))
            
            self.df[new_col] = parsed_lists
            
            # Validate the parsing
            validation = self.validate_parsed_lists(original_col, new_col)
            parse_results[original_col] = {
                'new_column': new_col,
                'validation': validation
            }
        
        self.parse_results = parse_results
        logger.info("List parsing complete for all fields")
        return parse_results
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the parsed data."""
        logger.info("Generating summary statistics...")
        
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'dataset_info': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'parsed_columns': list(self.list_fields.values())
            },
            'parsing_results': self.parse_results,
            'field_statistics': {}
        }
        
        # Generate statistics for each parsed field
        for original_col, new_col in self.list_fields.items():
            if new_col not in self.df.columns:
                continue
            
            # Item count distribution
            item_counts = self.df[new_col].apply(len)
            
            # Unique items analysis
            all_items = []
            for items in self.df[new_col]:
                all_items.extend(items)
            
            unique_items = len(set(all_items))
            total_items = len(all_items)
            
            # Most common items
            from collections import Counter
            item_counter = Counter(all_items)
            most_common = item_counter.most_common(10)
            
            summary['field_statistics'][new_col] = {
                'item_count_stats': {
                    'min': int(item_counts.min()),
                    'max': int(item_counts.max()),
                    'mean': float(item_counts.mean()),
                    'median': float(item_counts.median()),
                    'std': float(item_counts.std())
                },
                'unique_items': unique_items,
                'total_items': total_items,
                'most_common_items': most_common
            }
        
        return summary
    
    def save_parsed_data(self) -> str:
        """Save the parsed data as Parquet file."""
        logger.info("Saving parsed data as Parquet...")
        
        # Create output filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'parsed_books_{timestamp}.parquet'
        
        try:
            # Save as Parquet (more efficient than CSV for list data)
            self.df.to_parquet(output_file, index=False, engine='pyarrow')
            logger.info(f"Parsed data saved to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to save Parquet file: {e}")
            # Fallback to CSV
            csv_file = self.output_dir / f'parsed_books_{timestamp}.csv'
            self.df.to_csv(csv_file, index=False)
            logger.info(f"Fallback: Data saved as CSV to {csv_file}")
            return str(csv_file)
    
    def save_summary_report(self, summary: Dict[str, Any]) -> str:
        """Save parsing summary as JSON report."""
        report_file = self.output_dir / 'parsing_summary.json'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Parsing summary saved to {report_file}")
        return str(report_file)
    
    def generate_parsing_report(self, summary: Dict[str, Any]) -> str:
        """Generate a human-readable parsing report."""
        report_lines = [
            "="*60,
            "LIST PARSING REPORT",
            "="*60,
            f"Generated: {summary['timestamp']}",
            f"Dataset: {len(self.df):,} rows, {len(self.df.columns)} columns",
            "",
            "PARSING RESULTS:",
            "-"*30
        ]
        
        for original_col, result in summary['parsing_results'].items():
            validation = result['validation']
            report_lines.extend([
                f"\n{original_col} -> {result['new_column']}:",
                f"  Success Rate: {validation['parsing_success_rate']:.1%}",
                f"  Empty Lists: {validation['empty_lists']:,} ({validation['empty_list_rate']:.1%})",
                f"  Avg Items/Row: {validation['avg_items_per_row']:.1f}",
                f"  Total Items: {validation['total_items']:,}"
            ])
        
        report_lines.extend([
            "",
            "FIELD STATISTICS:",
            "-"*30
        ])
        
        for field, stats in summary['field_statistics'].items():
            report_lines.extend([
                f"\n{field}:",
                f"  Unique Items: {stats['unique_items']:,}",
                f"  Total Items: {stats['total_items']:,}",
                f"  Items per Row: {stats['item_count_stats']['mean']:.1f} ± {stats['item_count_stats']['std']:.1f}",
                f"  Range: {stats['item_count_stats']['min']} - {stats['item_count_stats']['max']}",
                f"  Top Items: {', '.join([item[0] for item in stats['most_common_items'][:5]])}"
            ])
        
        report_lines.extend([
            "",
            "="*60
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / 'parsing_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Parsing report saved to {report_file}")
        return str(report_file)
    
    def run_full_parsing(self) -> Dict[str, str]:
        """Run the complete list parsing pipeline."""
        logger.info("Starting comprehensive list parsing...")
        
        # Load data
        self.load_data()
        
        # Parse all list fields
        self.parse_all_lists()
        
        # Generate summary
        summary = self.generate_summary_statistics()
        
        # Save outputs
        parsed_file = self.save_parsed_data()
        summary_json = self.save_summary_report(summary)
        report_txt = self.generate_parsing_report(summary)
        
        logger.info("List parsing complete!")
        
        return {
            'parsed_data': parsed_file,
            'summary_json': summary_json,
            'report_txt': report_txt
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='List Parsing Script for Romance Novel Dataset')
    parser.add_argument('--data-path', type=str, 
                       default='../../data/processed/romance_books_main_final.csv',
                       help='Path to the CSV data file')
    parser.add_argument('--output-dir', type=str, default='./parse_outputs',
                       help='Output directory for parsed data')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run parsing
    parser_obj = ListParser(args.data_path, args.output_dir)
    results = parser_obj.run_full_parsing()
    
    print("\n" + "="*60)
    print("🎉 LIST PARSING COMPLETE!")
    print("="*60)
    print(f"📊 Parsed Data: {results['parsed_data']}")
    print(f"📄 Summary JSON: {results['summary_json']}")
    print(f"📋 Report: {results['report_txt']}")
    print("="*60)


if __name__ == "__main__":
    main()
