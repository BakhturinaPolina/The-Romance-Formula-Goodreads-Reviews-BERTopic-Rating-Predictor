#!/usr/bin/env python3
"""
Text Preprocessor for Romance Novel NLP Research
Handles comprehensive text preprocessing for NLP analysis.

Key Features:
- HTML cleaning and text normalization for descriptions
- Popular shelves format standardization
- Genre normalization and consistent categorization
- Comprehensive logging and validation
- Memory-efficient processing for large datasets

Based on analysis of cleaned dataset:
- 76,244 records with text fields
- 170 descriptions contain HTML indicators
- Popular shelves already comma-separated and lowercase
- Genres have inconsistent spacing and formatting

Author: Research Assistant
Date: 2025-01-06
"""

import pandas as pd
import numpy as np
import re
import html
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing for romance novel NLP analysis.
    
    Implements the Coding Agent Pattern:
    - Code Analyzer: Analyzes text fields for preprocessing needs
    - Change Planner: Plans text transformations and validations
    - Code Modifier: Applies text cleaning and normalization
    - Test Runner: Validates preprocessing results
    """
    
    def __init__(self, data_path: str = None, output_dir: str = "data/processed"):
        """
        Initialize the text preprocessor.
        
        Args:
            data_path: Path to input dataset (CSV or pickle)
            output_dir: Directory for output files
        """
        self.data_path = Path(data_path) if data_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.preprocessing_results = {}
        
        # HTML patterns for cleaning
        self.html_patterns = {
            'tags': re.compile(r'<[^>]+>'),
            'entities': re.compile(r'&[a-zA-Z0-9#]+;'),
            'whitespace': re.compile(r'\s+'),
            'line_breaks': re.compile(r'[\r\n]+'),
            'quotes': re.compile(r'["""]'),
            'apostrophes': re.compile(r'[''']')
        }
        
        # Genre normalization mappings
        self.genre_mappings = {
            # Romance subgenres
            'romance': ['romance', 'romantic', 'love story', 'love-story'],
            'contemporary romance': ['contemporary romance', 'contemporary-romance', 'modern romance'],
            'historical romance': ['historical romance', 'historical-romance', 'period romance'],
            'paranormal romance': ['paranormal romance', 'paranormal-romance', 'supernatural romance'],
            'erotic romance': ['erotic romance', 'erotic-romance', 'steamy romance'],
            
            # Fiction categories
            'fiction': ['fiction', 'novel', 'literature'],
            'contemporary fiction': ['contemporary fiction', 'contemporary-fiction', 'modern fiction'],
            'historical fiction': ['historical fiction', 'historical-fiction', 'period fiction'],
            'women\'s fiction': ['women\'s fiction', 'womens fiction', 'women fiction', 'chick lit', 'chick-lit'],
            
            # Other genres
            'mystery': ['mystery', 'thriller', 'suspense', 'crime'],
            'fantasy': ['fantasy', 'magic', 'supernatural'],
            'young adult': ['young adult', 'young-adult', 'ya', 'teen'],
            'biography': ['biography', 'memoir', 'autobiography'],
            'history': ['history', 'historical', 'non-fiction', 'nonfiction']
        }
        
        # Popular shelves validation patterns
        self.shelf_patterns = {
            'valid_chars': re.compile(r'^[a-z0-9\-,]+$'),
            'multiple_commas': re.compile(r',,+'),
            'trailing_commas': re.compile(r',\s*$'),
            'leading_commas': re.compile(r'^\s*,'),
            'spaces_around_commas': re.compile(r'\s*,\s*')
        }
        
        logger.info("TextPreprocessor initialized")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load dataset with preference for pickle format.
        
        Returns:
            Loaded DataFrame
        """
        if self.data_path is None:
            # Find most recent processed data
            processed_dir = Path("data/processed")
            csv_files = list(processed_dir.glob("cleaned_romance_novels_*.csv"))
            if csv_files:
                self.data_path = max(csv_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Using most recent CSV file: {self.data_path}")
            else:
                raise FileNotFoundError("No processed CSV files found in data/processed/")
        
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Try pickle first for better performance
            pickle_path = self.data_path.with_suffix('.pkl')
            if pickle_path.exists():
                logger.info("Loading from pickle file for better performance")
                with open(pickle_path, 'rb') as f:
                    self.df = pickle.load(f)
            else:
                logger.info("Loading from CSV file")
                self.df = pd.read_csv(self.data_path)
            
            logger.info(f"Loaded dataset with shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_text_fields(self) -> Dict[str, Any]:
        """
        Analyze text fields for preprocessing needs.
        
        Returns:
            Analysis results for each text field
        """
        logger.info("Analyzing text fields for preprocessing needs...")
        
        analysis = {
            'dataset_info': {
                'total_records': len(self.df),
                'text_fields': ['description', 'popular_shelves', 'genres']
            },
            'description_analysis': {},
            'popular_shelves_analysis': {},
            'genres_analysis': {}
        }
        
        # Analyze descriptions
        descriptions = self.df['description'].dropna()
        html_count = 0
        html_samples = []
        
        for i, desc in enumerate(descriptions):
            if any(pattern.search(str(desc)) for pattern in [self.html_patterns['tags'], self.html_patterns['entities']]):
                html_count += 1
                if len(html_samples) < 5:
                    html_samples.append(desc[:200])
        
        analysis['description_analysis'] = {
            'total_descriptions': len(descriptions),
            'html_indicators_found': html_count,
            'html_percentage': (html_count / len(descriptions)) * 100 if len(descriptions) > 0 else 0,
            'html_samples': html_samples,
            'avg_length': descriptions.str.len().mean(),
            'max_length': descriptions.str.len().max(),
            'min_length': descriptions.str.len().min()
        }
        
        # Analyze popular_shelves
        shelves = self.df['popular_shelves'].dropna()
        format_issues = []
        
        for shelf in shelves.head(100):  # Sample analysis
            if not self.shelf_patterns['valid_chars'].match(shelf):
                format_issues.append('invalid_chars')
            if self.shelf_patterns['multiple_commas'].search(shelf):
                format_issues.append('multiple_commas')
            if self.shelf_patterns['trailing_commas'].search(shelf):
                format_issues.append('trailing_commas')
            if self.shelf_patterns['leading_commas'].search(shelf):
                format_issues.append('leading_commas')
        
        analysis['popular_shelves_analysis'] = {
            'total_shelves': len(shelves),
            'format_issues': Counter(format_issues),
            'avg_length': shelves.str.len().mean(),
            'max_length': shelves.str.len().max(),
            'min_length': shelves.str.len().min(),
            'unique_shelf_count': len(set(shelves))
        }
        
        # Analyze genres
        genres = self.df['genres'].dropna()
        genre_issues = []
        all_genres = set()
        
        for genre in genres.head(100):  # Sample analysis
            if '  ' in genre:  # Multiple spaces
                genre_issues.append('multiple_spaces')
            if genre != genre.lower():
                genre_issues.append('mixed_case')
            if genre.startswith(',') or genre.endswith(','):
                genre_issues.append('trailing_commas')
            
            # Extract individual genres
            individual_genres = [g.strip() for g in genre.split(',')]
            all_genres.update(individual_genres)
        
        analysis['genres_analysis'] = {
            'total_genres': len(genres),
            'format_issues': Counter(genre_issues),
            'unique_genre_terms': len(all_genres),
            'sample_genres': list(all_genres)[:20],
            'avg_length': genres.str.len().mean(),
            'max_length': genres.str.len().max(),
            'min_length': genres.str.len().min()
        }
        
        logger.info(f"Text field analysis completed")
        logger.info(f"  - Descriptions with HTML: {analysis['description_analysis']['html_indicators_found']}")
        logger.info(f"  - Popular shelves format issues: {len(analysis['popular_shelves_analysis']['format_issues'])}")
        logger.info(f"  - Genre format issues: {len(analysis['genres_analysis']['format_issues'])}")
        
        return analysis
    
    def clean_html_and_normalize_text(self, text: str) -> str:
        """
        Clean HTML and normalize text for descriptions.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned and normalized text
        """
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = self.html_patterns['tags'].sub('', text)
        
        # Normalize quotes and apostrophes
        text = self.html_patterns['quotes'].sub('"', text)
        text = self.html_patterns['apostrophes'].sub("'", text)
        
        # Normalize line breaks
        text = self.html_patterns['line_breaks'].sub(' ', text)
        
        # Normalize whitespace
        text = self.html_patterns['whitespace'].sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def standardize_popular_shelves(self, shelves: str) -> str:
        """
        Standardize popular_shelves format.
        
        Args:
            shelves: Input shelves string
            
        Returns:
            Standardized shelves string
        """
        if pd.isna(shelves) or not isinstance(shelves, str):
            return shelves
        
        # Remove leading/trailing commas
        shelves = shelves.strip(',').strip()
        
        # Fix multiple consecutive commas
        shelves = self.shelf_patterns['multiple_commas'].sub(',', shelves)
        
        # Normalize spaces around commas
        shelves = self.shelf_patterns['spaces_around_commas'].sub(',', shelves)
        
        # Ensure lowercase
        shelves = shelves.lower()
        
        # Remove any invalid characters (keep only alphanumeric, hyphens, commas)
        shelves = re.sub(r'[^a-z0-9\-,]', '', shelves)
        
        # Final cleanup of multiple commas
        shelves = self.shelf_patterns['multiple_commas'].sub(',', shelves)
        shelves = shelves.strip(',').strip()
        
        return shelves
    
    def normalize_genres(self, genres: str) -> str:
        """
        Normalize genres field for consistent categorization.
        
        Args:
            genres: Input genres string
            
        Returns:
            Normalized genres string
        """
        if pd.isna(genres) or not isinstance(genres, str):
            return genres
        
        # Split by comma and clean each genre
        genre_list = [g.strip() for g in genres.split(',')]
        
        # Remove empty genres
        genre_list = [g for g in genre_list if g]
        
        # Normalize each genre
        normalized_genres = []
        for genre in genre_list:
            # Convert to lowercase
            genre = genre.lower()
            
            # Remove extra spaces
            genre = re.sub(r'\s+', ' ', genre).strip()
            
            # Map to standard genre names
            mapped_genre = self._map_genre_to_standard(genre)
            if mapped_genre and mapped_genre not in normalized_genres:
                normalized_genres.append(mapped_genre)
        
        # Sort for consistency
        normalized_genres.sort()
        
        return ','.join(normalized_genres)
    
    def _map_genre_to_standard(self, genre: str) -> Optional[str]:
        """
        Map a genre to its standard form.
        
        Args:
            genre: Input genre string
            
        Returns:
            Standard genre name or None if no mapping found
        """
        genre = genre.strip().lower()
        
        # Direct mapping
        for standard_genre, variants in self.genre_mappings.items():
            if genre in variants:
                return standard_genre
        
        # Fuzzy matching for common variations
        if 'romance' in genre:
            if 'contemporary' in genre or 'modern' in genre:
                return 'contemporary romance'
            elif 'historical' in genre or 'period' in genre:
                return 'historical romance'
            elif 'paranormal' in genre or 'supernatural' in genre:
                return 'paranormal romance'
            elif 'erotic' in genre or 'steamy' in genre:
                return 'erotic romance'
            else:
                return 'romance'
        
        if 'fiction' in genre:
            if 'contemporary' in genre or 'modern' in genre:
                return 'contemporary fiction'
            elif 'historical' in genre or 'period' in genre:
                return 'historical fiction'
            elif 'women' in genre or 'chick' in genre:
                return 'women\'s fiction'
            else:
                return 'fiction'
        
        if 'mystery' in genre or 'thriller' in genre or 'suspense' in genre or 'crime' in genre:
            return 'mystery'
        
        if 'fantasy' in genre or 'magic' in genre or 'supernatural' in genre:
            return 'fantasy'
        
        if 'young' in genre or 'ya' in genre or 'teen' in genre:
            return 'young adult'
        
        if 'biography' in genre or 'memoir' in genre or 'autobiography' in genre:
            return 'biography'
        
        if 'history' in genre or 'historical' in genre:
            return 'history'
        
        # Return original if no mapping found
        return genre
    
    def apply_text_preprocessing(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply comprehensive text preprocessing to all text fields.
        
        Returns:
            Tuple of (processed_dataframe, preprocessing_results)
        """
        logger.info("Applying comprehensive text preprocessing...")
        
        # Create a copy to avoid modifying original
        processed_df = self.df.copy()
        
        preprocessing_results = {
            'preprocessing_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'original_shape': self.df.shape,
            'text_field_processing': {}
        }
        
        # Process descriptions
        logger.info("Processing descriptions...")
        original_desc_count = processed_df['description'].notna().sum()
        processed_df['description'] = processed_df['description'].apply(self.clean_html_and_normalize_text)
        processed_desc_count = processed_df['description'].notna().sum()
        
        preprocessing_results['text_field_processing']['descriptions'] = {
            'original_count': original_desc_count,
            'processed_count': processed_desc_count,
            'html_cleaned': True,
            'whitespace_normalized': True
        }
        
        # Process popular_shelves
        logger.info("Processing popular_shelves...")
        original_shelves_count = processed_df['popular_shelves'].notna().sum()
        processed_df['popular_shelves'] = processed_df['popular_shelves'].apply(self.standardize_popular_shelves)
        processed_shelves_count = processed_df['popular_shelves'].notna().sum()
        
        preprocessing_results['text_field_processing']['popular_shelves'] = {
            'original_count': original_shelves_count,
            'processed_count': processed_shelves_count,
            'format_standardized': True,
            'case_normalized': True
        }
        
        # Process genres
        logger.info("Processing genres...")
        original_genres_count = processed_df['genres'].notna().sum()
        processed_df['genres'] = processed_df['genres'].apply(self.normalize_genres)
        processed_genres_count = processed_df['genres'].notna().sum()
        
        preprocessing_results['text_field_processing']['genres'] = {
            'original_count': original_genres_count,
            'processed_count': processed_genres_count,
            'normalized': True,
            'categorized': True
        }
        
        # Add preprocessing metadata
        processed_df['text_preprocessing_applied'] = True
        processed_df['text_preprocessing_timestamp'] = preprocessing_results['preprocessing_timestamp']
        
        preprocessing_results['final_shape'] = processed_df.shape
        preprocessing_results['processing_successful'] = True
        
        logger.info("Text preprocessing completed successfully")
        logger.info(f"  - Descriptions processed: {processed_desc_count}")
        logger.info(f"  - Popular shelves processed: {processed_shelves_count}")
        logger.info(f"  - Genres processed: {processed_genres_count}")
        
        return processed_df, preprocessing_results
    
    def validate_preprocessing_results(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate preprocessing results for quality assurance.
        
        Args:
            processed_df: Processed DataFrame
            
        Returns:
            Validation results
        """
        logger.info("Validating preprocessing results...")
        
        validation_results = {
            'validation_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'overall_status': 'success',
            'field_validations': {}
        }
        
        # Validate descriptions
        descriptions = processed_df['description'].dropna()
        html_remaining = 0
        for desc in descriptions:
            if any(pattern.search(str(desc)) for pattern in [self.html_patterns['tags'], self.html_patterns['entities']]):
                html_remaining += 1
        
        validation_results['field_validations']['descriptions'] = {
            'total_descriptions': len(descriptions),
            'html_remaining': html_remaining,
            'html_cleaning_success': html_remaining == 0,
            'avg_length': descriptions.str.len().mean(),
            'status': 'success' if html_remaining == 0 else 'warning'
        }
        
        # Validate popular_shelves
        shelves = processed_df['popular_shelves'].dropna()
        format_issues = 0
        for shelf in shelves.head(100):  # Sample validation
            if not self.shelf_patterns['valid_chars'].match(shelf):
                format_issues += 1
        
        validation_results['field_validations']['popular_shelves'] = {
            'total_shelves': len(shelves),
            'format_issues': format_issues,
            'format_standardization_success': format_issues == 0,
            'status': 'success' if format_issues == 0 else 'warning'
        }
        
        # Validate genres
        genres = processed_df['genres'].dropna()
        format_issues = 0
        for genre in genres.head(100):  # Sample validation
            if '  ' in genre or genre != genre.lower():
                format_issues += 1
        
        validation_results['field_validations']['genres'] = {
            'total_genres': len(genres),
            'format_issues': format_issues,
            'normalization_success': format_issues == 0,
            'status': 'success' if format_issues == 0 else 'warning'
        }
        
        # Overall validation status
        all_success = all(
            field['status'] == 'success' 
            for field in validation_results['field_validations'].values()
        )
        validation_results['overall_status'] = 'success' if all_success else 'warning'
        
        logger.info(f"Validation completed with status: {validation_results['overall_status']}")
        
        return validation_results
    
    def save_preprocessed_dataset(self, processed_df: pd.DataFrame, filename: str = None, format_type: str = 'csv') -> str:
        """
        Save preprocessed dataset.
        
        Args:
            processed_df: Processed DataFrame
            filename: Output filename
            format_type: Output format ('csv', 'pickle', 'parquet')
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"romance_novels_text_preprocessed_{timestamp}"
        
        output_path = self.output_dir / filename
        
        if format_type.lower() == 'csv':
            file_path = output_path.with_suffix('.csv')
            processed_df.to_csv(file_path, index=False)
        elif format_type.lower() == 'pickle':
            file_path = output_path.with_suffix('.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(processed_df, f)
        elif format_type.lower() == 'parquet':
            file_path = output_path.with_suffix('.parquet')
            processed_df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Preprocessed dataset saved to {file_path}")
        return str(file_path)
    
    def save_preprocessing_report(self, analysis: Dict[str, Any], preprocessing_results: Dict[str, Any], 
                                validation: Dict[str, Any], filename: str = None) -> str:
        """
        Save comprehensive preprocessing report.
        
        Args:
            analysis: Text field analysis results
            preprocessing_results: Preprocessing results
            validation: Validation results
            filename: Output filename
            
        Returns:
            Path to saved report
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"text_preprocessing_report_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'preprocessor_version': '1.0',
                'dataset_path': str(self.data_path) if self.data_path else None
            },
            'text_field_analysis': analysis,
            'preprocessing_results': preprocessing_results,
            'validation_results': validation,
            'summary': {
                'total_records_processed': preprocessing_results['final_shape'][0],
                'text_fields_processed': len(preprocessing_results['text_field_processing']),
                'overall_success': validation['overall_status'] == 'success',
                'html_cleaning_applied': True,
                'format_standardization_applied': True,
                'genre_normalization_applied': True
            }
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Preprocessing report saved to {output_path}")
        return str(output_path)
    
    def run_complete_preprocessing(self) -> Dict[str, Any]:
        """
        Run complete text preprocessing pipeline.
        
        Returns:
            Complete preprocessing results
        """
        logger.info("Starting complete text preprocessing pipeline...")
        
        # Load data
        self.load_data()
        
        # Analyze text fields
        analysis = self.analyze_text_fields()
        
        # Apply preprocessing
        processed_df, preprocessing_results = self.apply_text_preprocessing()
        
        # Validate results
        validation = self.validate_preprocessing_results(processed_df)
        
        # Save results
        dataset_path = self.save_preprocessed_dataset(processed_df, format_type='csv')
        report_path = self.save_preprocessing_report(analysis, preprocessing_results, validation)
        
        # Store results
        self.preprocessing_results = {
            'analysis': analysis,
            'preprocessing_results': preprocessing_results,
            'validation': validation,
            'dataset_path': dataset_path,
            'report_path': report_path
        }
        
        logger.info("Complete text preprocessing pipeline finished")
        return self.preprocessing_results
    
    def print_preprocessing_summary(self):
        """Print a summary of the preprocessing results."""
        if not self.preprocessing_results:
            logger.warning("No preprocessing results available. Run run_complete_preprocessing() first.")
            return
        
        results = self.preprocessing_results
        
        print("\n" + "="*80)
        print("TEXT PREPROCESSING SUMMARY")
        print("="*80)
        
        print(f"\n📊 DATASET PROCESSING:")
        print(f"  Original records: {results['preprocessing_results']['original_shape'][0]:,}")
        print(f"  Final records: {results['preprocessing_results']['final_shape'][0]:,}")
        
        print(f"\n🔧 TEXT FIELD PROCESSING:")
        for field, processing in results['preprocessing_results']['text_field_processing'].items():
            print(f"  {field}:")
            print(f"    - Processed: {processing['processed_count']:,}")
            print(f"    - HTML cleaned: {'✅' if processing.get('html_cleaned') else '❌'}")
            print(f"    - Format standardized: {'✅' if processing.get('format_standardized') else '❌'}")
            print(f"    - Normalized: {'✅' if processing.get('normalized') else '❌'}")
        
        print(f"\n✅ VALIDATION RESULTS:")
        validation = results['validation']
        print(f"  Overall status: {validation['overall_status'].upper()}")
        for field, field_validation in validation['field_validations'].items():
            status_icon = "✅" if field_validation['status'] == 'success' else "⚠️"
            print(f"  {field}: {status_icon} {field_validation['status'].upper()}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  Dataset: {results['dataset_path']}")
        print(f"  Report: {results['report_path']}")
        
        print("\n" + "="*80)


def main():
    """Main function to run text preprocessing."""
    print("🔤 Starting Text Preprocessing for NLP Analysis...")
    
    preprocessor = TextPreprocessor()
    results = preprocessor.run_complete_preprocessing()
    
    # Print summary
    preprocessor.print_preprocessing_summary()
    
    print(f"\n📄 Preprocessing completed successfully!")
    print(f"📄 Processed dataset: {results['dataset_path']}")
    print(f"📄 Detailed report: {results['report_path']}")
    
    return results


if __name__ == "__main__":
    main()
