#!/usr/bin/env python3
"""
Test Script for Text Preprocessor on Small Dataset Sample
Tests the text preprocessing functionality on a small subset of the processed dataset.

Features:
- Loads small sample from processed dataset
- Tests individual preprocessing functions
- Validates preprocessing results
- Generates comprehensive test report
- Follows project coding patterns and safety guidelines

Author: Research Assistant
Date: 2025-01-06
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from nlp_preprocessing.text_preprocessor import TextPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextPreprocessorTester:
    """
    Test class for Text Preprocessor functionality.
    
    Implements the Coding Agent Pattern:
    - Code Analyzer: Analyzes test requirements and data structure
    - Change Planner: Plans test scenarios and validation steps
    - Code Modifier: Applies preprocessing and captures results
    - Test Runner: Validates preprocessing outputs
    """
    
    def __init__(self, sample_size: int = 100, output_dir: str = "data/processed/test_output"):
        """
        Initialize the text preprocessor tester.
        
        Args:
            sample_size: Number of records to use for testing
            output_dir: Directory for test output files
        """
        self.sample_size = sample_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        self.sample_data = None
        self.preprocessor = None
        
        logger.info(f"Initialized TextPreprocessorTester with sample size: {sample_size}")
    
    def load_sample_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load a small sample from the processed dataset.
        
        Args:
            data_path: Path to the processed dataset
            
        Returns:
            DataFrame with sample data
        """
        if data_path is None:
            # Use the most recent processed dataset
            data_path = "data/processed/final_books_2000_2020_en_enhanced_20250905_003102.csv"
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        logger.info(f"Loading sample of {self.sample_size} records from {data_path}")
        
        # Load sample data
        self.sample_data = pd.read_csv(data_path, nrows=self.sample_size)
        
        logger.info(f"Loaded {len(self.sample_data)} records with {len(self.sample_data.columns)} columns")
        logger.info(f"Columns: {list(self.sample_data.columns)}")
        
        # Save sample for reference
        sample_path = self.output_dir / f"test_sample_{self.sample_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.sample_data.to_csv(sample_path, index=False)
        logger.info(f"Sample saved to: {sample_path}")
        
        return self.sample_data
    
    def analyze_sample_data(self) -> Dict[str, Any]:
        """
        Analyze the sample data to understand its structure and content.
        
        Returns:
            Dictionary with analysis results
        """
        if self.sample_data is None:
            raise ValueError("No sample data loaded. Call load_sample_data() first.")
        
        logger.info("Analyzing sample data structure...")
        
        analysis = {
            'total_records': len(self.sample_data),
            'total_columns': len(self.sample_data.columns),
            'text_columns': [],
            'missing_values': {},
            'data_types': {},
            'sample_values': {}
        }
        
        # Identify text columns
        text_columns = ['description', 'popular_shelves', 'genres', 'title', 'author_name', 'series_title']
        for col in text_columns:
            if col in self.sample_data.columns:
                analysis['text_columns'].append(col)
                analysis['missing_values'][col] = self.sample_data[col].isna().sum()
                analysis['data_types'][col] = str(self.sample_data[col].dtype)
                
                # Get sample values
                non_null_values = self.sample_data[col].dropna()
                if len(non_null_values) > 0:
                    analysis['sample_values'][col] = non_null_values.iloc[0] if len(non_null_values) > 0 else None
        
        # Check for HTML content in descriptions
        if 'description' in self.sample_data.columns:
            descriptions = self.sample_data['description'].dropna()
            html_indicators = descriptions.str.contains(r'<[^>]+>', na=False).sum()
            analysis['html_indicators'] = html_indicators
        
        logger.info(f"Analysis complete: {len(analysis['text_columns'])} text columns found")
        return analysis
    
    def test_html_cleaning(self) -> Dict[str, Any]:
        """
        Test HTML cleaning functionality.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing HTML cleaning functionality...")
        
        if 'description' in self.sample_data.columns:
            descriptions = self.sample_data['description'].dropna()
            
            # Find descriptions with HTML
            html_descriptions = descriptions[descriptions.str.contains(r'<[^>]+>', na=False)]
            
            test_results = {
                'total_descriptions': len(descriptions),
                'html_descriptions': len(html_descriptions),
                'html_percentage': len(html_descriptions) / len(descriptions) * 100 if len(descriptions) > 0 else 0,
                'examples': {}
            }
            
            if len(html_descriptions) > 0:
                # Test cleaning on first few examples
                for i, (idx, desc) in enumerate(html_descriptions.head(3).items()):
                    # Initialize preprocessor for cleaning
                    if self.preprocessor is None:
                        self.preprocessor = TextPreprocessor()
                    
                    cleaned = self.preprocessor.clean_html_content(desc)
                    test_results['examples'][f'example_{i+1}'] = {
                        'original': desc[:200] + "..." if len(desc) > 200 else desc,
                        'cleaned': cleaned[:200] + "..." if len(cleaned) > 200 else cleaned,
                        'html_removed': len(desc) - len(cleaned)
                    }
            
            logger.info(f"HTML cleaning test: {test_results['html_descriptions']}/{test_results['total_descriptions']} descriptions contain HTML")
            return test_results
        else:
            logger.warning("No 'description' column found for HTML cleaning test")
            return {'error': 'No description column found'}
    
    def test_genre_normalization(self) -> Dict[str, Any]:
        """
        Test genre normalization functionality.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing genre normalization functionality...")
        
        if 'genres' in self.sample_data.columns:
            genres = self.sample_data['genres'].dropna()
            
            test_results = {
                'total_genres': len(genres),
                'unique_genres': genres.nunique(),
                'genre_examples': {},
                'normalization_examples': {}
            }
            
            # Get sample genre values
            sample_genres = genres.head(5)
            for i, (idx, genre) in enumerate(sample_genres.items()):
                test_results['genre_examples'][f'example_{i+1}'] = genre
            
            # Test normalization
            if self.preprocessor is None:
                self.preprocessor = TextPreprocessor()
            
            for i, (idx, genre) in enumerate(sample_genres.items()):
                normalized = self.preprocessor.normalize_genres(genre)
                test_results['normalization_examples'][f'example_{i+1}'] = {
                    'original': genre,
                    'normalized': normalized
                }
            
            logger.info(f"Genre normalization test: {test_results['unique_genres']} unique genres found")
            return test_results
        else:
            logger.warning("No 'genres' column found for genre normalization test")
            return {'error': 'No genres column found'}
    
    def test_popular_shelves_standardization(self) -> Dict[str, Any]:
        """
        Test popular shelves standardization functionality.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing popular shelves standardization functionality...")
        
        if 'popular_shelves' in self.sample_data.columns:
            shelves = self.sample_data['popular_shelves'].dropna()
            
            test_results = {
                'total_shelves': len(shelves),
                'validation_examples': {}
            }
            
            # Test validation on sample shelves
            sample_shelves = shelves.head(5)
            if self.preprocessor is None:
                self.preprocessor = TextPreprocessor()
            
            for i, (idx, shelf) in enumerate(sample_shelves.items()):
                standardized = self.preprocessor.standardize_popular_shelves(shelf)
                test_results['validation_examples'][f'example_{i+1}'] = {
                    'original': shelf,
                    'standardized': standardized
                }
            
            logger.info(f"Popular shelves standardization test: {len(shelves)} shelves to standardize")
            return test_results
        else:
            logger.warning("No 'popular_shelves' column found for validation test")
            return {'error': 'No popular_shelves column found'}
    
    def run_complete_test(self, data_path: str = None) -> Dict[str, Any]:
        """
        Run complete preprocessing test suite.
        
        Args:
            data_path: Path to the processed dataset
            
        Returns:
            Dictionary with all test results
        """
        logger.info("Starting complete preprocessing test suite...")
        
        # Load sample data
        self.load_sample_data(data_path)
        
        # Analyze sample data
        analysis = self.analyze_sample_data()
        
        # Run individual tests
        tests = {
            'data_analysis': analysis,
            'html_cleaning': self.test_html_cleaning(),
            'genre_normalization': self.test_genre_normalization(),
            'popular_shelves_standardization': self.test_popular_shelves_standardization()
        }
        
        # Generate test report
        report = self.generate_test_report(tests)
        
        # Save results
        self.save_test_results(tests, report)
        
        logger.info("Complete preprocessing test suite finished")
        return {
            'tests': tests,
            'report': report,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_test_report(self, tests: Dict[str, Any]) -> str:
        """
        Generate a comprehensive test report.
        
        Args:
            tests: Dictionary with test results
            
        Returns:
            Formatted test report string
        """
        report_lines = [
            "TEXT PREPROCESSOR TEST REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sample size: {self.sample_size}",
            "",
            "DATA ANALYSIS",
            "-" * 20,
            f"Total records: {tests['data_analysis']['total_records']}",
            f"Total columns: {tests['data_analysis']['total_columns']}",
            f"Text columns: {', '.join(tests['data_analysis']['text_columns'])}",
            ""
        ]
        
        # HTML Cleaning Results
        if 'html_cleaning' in tests and 'error' not in tests['html_cleaning']:
            html_test = tests['html_cleaning']
            report_lines.extend([
                "HTML CLEANING TEST",
                "-" * 20,
                f"Total descriptions: {html_test['total_descriptions']}",
                f"HTML descriptions: {html_test['html_descriptions']}",
                f"HTML percentage: {html_test['html_percentage']:.2f}%",
                ""
            ])
        
        # Genre Normalization Results
        if 'genre_normalization' in tests and 'error' not in tests['genre_normalization']:
            genre_test = tests['genre_normalization']
            report_lines.extend([
                "GENRE NORMALIZATION TEST",
                "-" * 20,
                f"Total genres: {genre_test['total_genres']}",
                f"Unique genres: {genre_test['unique_genres']}",
                ""
            ])
        
        # Popular Shelves Standardization Results
        if 'popular_shelves_standardization' in tests and 'error' not in tests['popular_shelves_standardization']:
            shelves_test = tests['popular_shelves_standardization']
            report_lines.extend([
                "POPULAR SHELVES STANDARDIZATION TEST",
                "-" * 20,
                f"Total shelves: {shelves_test['total_shelves']}",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def save_test_results(self, tests: Dict[str, Any], report: str):
        """
        Save test results to files.
        
        Args:
            tests: Dictionary with test results
            report: Test report string
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save test report
        report_path = self.output_dir / f"preprocessing_test_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save detailed results as JSON
        import json
        results_path = self.output_dir / f"preprocessing_test_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(tests, f, indent=2, default=str)
        
        logger.info(f"Test results saved to: {report_path}")
        logger.info(f"Detailed results saved to: {results_path}")


def main():
    """Main function to run text preprocessing tests."""
    print("🧪 Starting Text Preprocessing Test Suite...")
    print("=" * 60)
    
    # Initialize tester with small sample
    tester = TextPreprocessorTester(sample_size=100)
    
    # Run complete test suite
    results = tester.run_complete_test()
    
    # Print summary
    print("\n📊 TEST SUMMARY")
    print("-" * 30)
    
    if 'data_analysis' in results['tests']:
        analysis = results['tests']['data_analysis']
        print(f"Sample size: {analysis['total_records']} records")
        print(f"Text columns: {', '.join(analysis['text_columns'])}")
    
    if 'html_cleaning' in results['tests'] and 'error' not in results['tests']['html_cleaning']:
        html_test = results['tests']['html_cleaning']
        print(f"HTML cleaning: {html_test['html_descriptions']}/{html_test['total_descriptions']} descriptions contain HTML")
    
    if 'genre_normalization' in results['tests'] and 'error' not in results['tests']['genre_normalization']:
        genre_test = results['tests']['genre_normalization']
        print(f"Genre normalization: {genre_test['unique_genres']} unique genres found")
    
    print(f"\n✅ Text preprocessing test completed successfully!")
    print(f"📄 Test report: {results['report']}")
    
    return results


if __name__ == "__main__":
    main()
