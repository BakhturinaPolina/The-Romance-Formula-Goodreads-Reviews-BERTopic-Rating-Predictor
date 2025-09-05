#!/usr/bin/env python3
"""
Diagnostic script to identify sources of null values in CSV building process.
This script will help understand why nulls appear in the CSV output despite
the raw JSON.gz files showing 0% null values.
"""

import pandas as pd
import json
import gzip
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NullDiagnostic:
    """Diagnostic tool to identify null value sources in CSV building process."""
    
    def __init__(self):
        self.english_regex = re.compile(r'^(eng|en(?:-[A-Za-z]+)?)$', re.IGNORECASE)
        
    def is_english_language(self, language_code: str) -> bool:
        """Check if language code represents English."""
        if not language_code or not isinstance(language_code, str):
            return False
        return bool(self.english_regex.match(language_code.strip()))
    
    def load_sample_data(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Load a sample of data for analysis."""
        print(f"🔍 Loading sample data ({sample_size} records)...")
        
        # Load books data
        books_records = []
        with gzip.open("data/raw/goodreads_books_romance.json.gz", 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        books_records.append(record)
                    except json.JSONDecodeError:
                        continue
        
        books_df = pd.DataFrame(books_records)
        print(f"✅ Loaded {len(books_df)} book records")
        
        # Convert to numeric with coerce (this is where nulls might be introduced)
        print("🔍 Analyzing data type conversions...")
        
        # Check original data types and values
        numeric_columns = ['work_id', 'book_id', 'publication_year', 'num_pages', 
                          'ratings_count', 'text_reviews_count', 'average_rating']
        
        conversion_results = {}
        
        for col in numeric_columns:
            if col in books_df.columns:
                original_values = books_df[col].head(10).tolist()
                original_dtypes = books_df[col].dtype
                original_nulls = books_df[col].isnull().sum()
                
                # Convert to numeric
                converted = pd.to_numeric(books_df[col], errors='coerce')
                converted_nulls = converted.isnull().sum()
                new_nulls = converted_nulls - original_nulls
                
                conversion_results[col] = {
                    'original_dtype': str(original_dtypes),
                    'original_nulls': original_nulls,
                    'converted_nulls': converted_nulls,
                    'new_nulls_introduced': new_nulls,
                    'sample_values': original_values[:5]
                }
                
                if new_nulls > 0:
                    print(f"⚠️  {col}: {new_nulls} nulls introduced during conversion")
                    # Show examples of problematic values
                    problematic = books_df[converted.isnull() & books_df[col].notnull()][col].head(5)
                    print(f"   Problematic values: {problematic.tolist()}")
        
        return {
            'books_df': books_df,
            'conversion_results': conversion_results
        }
    
    def analyze_english_filtering(self, books_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the English filtering process."""
        print("🌍 Analyzing English language filtering...")
        
        # Check language_code column
        if 'language_code' not in books_df.columns:
            print("❌ No language_code column found")
            return {}
        
        language_analysis = {
            'total_records': len(books_df),
            'language_code_nulls': books_df['language_code'].isnull().sum(),
            'unique_language_codes': books_df['language_code'].nunique(),
            'language_code_distribution': books_df['language_code'].value_counts().head(10).to_dict()
        }
        
        # Apply English filtering
        books_df['is_english'] = books_df['language_code'].apply(self.is_english_language)
        english_books = books_df[books_df['is_english']]
        
        language_analysis.update({
            'english_records': len(english_books),
            'non_english_records': len(books_df) - len(english_books),
            'english_percentage': (len(english_books) / len(books_df)) * 100
        })
        
        print(f"📊 Language Analysis:")
        print(f"  - Total records: {language_analysis['total_records']:,}")
        print(f"  - English records: {language_analysis['english_records']:,} ({language_analysis['english_percentage']:.1f}%)")
        print(f"  - Language code nulls: {language_analysis['language_code_nulls']}")
        
        return language_analysis
    
    def analyze_work_aggregation(self, books_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze work-level aggregation process."""
        print("🔗 Analyzing work-level aggregation...")
        
        # Filter to English editions
        books_df['is_english'] = books_df['language_code'].apply(self.is_english_language)
        english_books_df = books_df[books_df['is_english']].copy()
        
        # Convert work_id to numeric
        english_books_df['work_id'] = pd.to_numeric(english_books_df['work_id'], errors='coerce')
        
        # Check for work_id nulls after conversion
        work_id_nulls = english_books_df['work_id'].isnull().sum()
        print(f"📊 Work ID Analysis:")
        print(f"  - English records: {len(english_books_df):,}")
        print(f"  - Work ID nulls after conversion: {work_id_nulls}")
        
        if work_id_nulls > 0:
            print("⚠️  Records with null work_id will be excluded from aggregation")
            # Show examples
            null_work_ids = english_books_df[english_books_df['work_id'].isnull()].head(5)
            print(f"   Examples of records with null work_id:")
            for idx, row in null_work_ids.iterrows():
                print(f"     Book ID {row.get('book_id', 'N/A')}: work_id = {row.get('work_id', 'N/A')}")
        
        # Group by work_id
        work_groups = english_books_df.groupby('work_id')
        print(f"  - Unique works: {len(work_groups):,}")
        
        # Analyze aggregation potential issues
        aggregation_issues = []
        
        for work_id, group in list(work_groups)[:5]:  # Sample first 5 works
            group_issues = []
            
            # Check for missing data in key fields
            if group['publication_year'].isnull().all():
                group_issues.append("All publication_years are null")
            
            if group['num_pages'].isnull().all():
                group_issues.append("All num_pages are null")
            
            if group['average_rating'].isnull().all():
                group_issues.append("All average_ratings are null")
            
            if group_issues:
                aggregation_issues.append({
                    'work_id': work_id,
                    'issues': group_issues,
                    'editions_count': len(group)
                })
        
        if aggregation_issues:
            print("⚠️  Aggregation issues found:")
            for issue in aggregation_issues:
                print(f"   Work {issue['work_id']}: {', '.join(issue['issues'])}")
        
        return {
            'work_id_nulls': work_id_nulls,
            'unique_works': len(work_groups),
            'aggregation_issues': aggregation_issues
        }
    
    def analyze_author_series_lookup(self, books_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze author and series data lookup issues."""
        print("👤 Analyzing author and series data lookup...")
        
        # Load author and series data
        authors_data = {}
        series_data = {}
        
        # Load authors data
        with gzip.open("data/raw/goodreads_book_authors.json.gz", 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        authors_data[str(record['author_id'])] = record
                    except json.JSONDecodeError:
                        continue
        
        # Load series data
        with gzip.open("data/raw/goodreads_book_series.json.gz", 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        series_data[str(record['series_id'])] = record
                    except json.JSONDecodeError:
                        continue
        
        print(f"📚 Loaded {len(authors_data):,} authors and {len(series_data):,} series")
        
        # Analyze author lookup issues
        author_lookup_issues = 0
        series_lookup_issues = 0
        
        # Sample analysis
        sample_books = books_df.head(100)
        
        for _, book in sample_books.iterrows():
            # Check author lookup
            authors = book.get('authors', [])
            if authors and isinstance(authors, list):
                for author in authors:
                    author_id = str(author.get('author_id', ''))
                    if author_id and author_id not in authors_data:
                        author_lookup_issues += 1
            
            # Check series lookup
            series = book.get('series', [])
            if series and isinstance(series, list):
                for series_id in series:
                    series_id_str = str(series_id)
                    if series_id_str and series_id_str not in series_data:
                        series_lookup_issues += 1
        
        print(f"📊 Lookup Analysis (sample of 100 books):")
        print(f"  - Author lookup issues: {author_lookup_issues}")
        print(f"  - Series lookup issues: {series_lookup_issues}")
        
        return {
            'authors_loaded': len(authors_data),
            'series_loaded': len(series_data),
            'author_lookup_issues': author_lookup_issues,
            'series_lookup_issues': series_lookup_issues
        }
    
    def run_full_diagnosis(self, sample_size: int = 1000):
        """Run complete diagnosis of null value sources."""
        print("🔍 NULL VALUE DIAGNOSIS")
        print("=" * 50)
        
        # Load sample data
        data = self.load_sample_data(sample_size)
        books_df = data['books_df']
        conversion_results = data['conversion_results']
        
        print("\n📊 DATA TYPE CONVERSION ANALYSIS:")
        print("-" * 40)
        for col, results in conversion_results.items():
            print(f"{col}:")
            print(f"  - Original dtype: {results['original_dtype']}")
            print(f"  - Original nulls: {results['original_nulls']}")
            print(f"  - New nulls introduced: {results['new_nulls_introduced']}")
            if results['new_nulls_introduced'] > 0:
                print(f"  - Sample problematic values: {results['sample_values']}")
        
        # Analyze English filtering
        print("\n🌍 ENGLISH FILTERING ANALYSIS:")
        print("-" * 40)
        language_analysis = self.analyze_english_filtering(books_df)
        
        # Analyze work aggregation
        print("\n🔗 WORK AGGREGATION ANALYSIS:")
        print("-" * 40)
        aggregation_analysis = self.analyze_work_aggregation(books_df)
        
        # Analyze author/series lookup
        print("\n👤 AUTHOR/SERIES LOOKUP ANALYSIS:")
        print("-" * 40)
        lookup_analysis = self.analyze_author_series_lookup(books_df)
        
        # Summary
        print("\n📋 DIAGNOSIS SUMMARY:")
        print("=" * 50)
        
        total_nulls_introduced = sum(r['new_nulls_introduced'] for r in conversion_results.values())
        print(f"1. Data Type Conversion: {total_nulls_introduced} nulls introduced")
        
        if language_analysis.get('language_code_nulls', 0) > 0:
            print(f"2. Language Filtering: {language_analysis['language_code_nulls']} records with null language_code")
        
        if aggregation_analysis.get('work_id_nulls', 0) > 0:
            print(f"3. Work Aggregation: {aggregation_analysis['work_id_nulls']} records with null work_id")
        
        if lookup_analysis.get('author_lookup_issues', 0) > 0:
            print(f"4. Author Lookup: {lookup_analysis['author_lookup_issues']} author references not found")
        
        if lookup_analysis.get('series_lookup_issues', 0) > 0:
            print(f"5. Series Lookup: {lookup_analysis['series_lookup_issues']} series references not found")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 20)
        if total_nulls_introduced > 0:
            print("• Review data type conversion logic - some values may be non-numeric strings")
        if language_analysis.get('language_code_nulls', 0) > 0:
            print("• Handle null language_code values in filtering logic")
        if aggregation_analysis.get('work_id_nulls', 0) > 0:
            print("• Handle null work_id values before aggregation")
        if lookup_analysis.get('author_lookup_issues', 0) > 0:
            print("• Some author references may be missing from authors dataset")
        if lookup_analysis.get('series_lookup_issues', 0) > 0:
            print("• Some series references may be missing from series dataset")

def main():
    """Run the null value diagnosis."""
    diagnostic = NullDiagnostic()
    diagnostic.run_full_diagnosis(sample_size=1000)

if __name__ == "__main__":
    main()
