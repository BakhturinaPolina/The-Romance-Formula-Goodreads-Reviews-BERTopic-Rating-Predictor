#!/usr/bin/env python3
"""
Step 2: Duplicate and Inconsistency Detection
Comprehensive detection of duplicates and data inconsistencies in the cleaned dataset.

Implementation Strategy:
1. Start Simple: Begin with obvious duplicates (work IDs, exact title matches)
2. Iterate: Build complexity gradually based on findings
3. Document Everything: Maintain research transparency throughout
4. Test Thoroughly: Validate all detection algorithms before finalizing
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DuplicateInconsistencyDetection:
    """
    Step 2: Duplicate and Inconsistency Detection
    
    Implements systematic detection of:
    - Work ID duplicates
    - Title similarity analysis
    - Author consistency checks
    - Series ordering validation
    - Genre classification patterns
    """
    
    def __init__(self, data_path: str = "data/processed"):
        """Initialize the duplicate and inconsistency detection."""
        # Convert to absolute path relative to project root
        if Path(data_path).is_absolute():
            self.data_path = Path(data_path)
        else:
            # Find project root (look for src directory)
            current_dir = Path.cwd()
            while current_dir.parent != current_dir:
                if (current_dir / "src").exists():
                    break
                current_dir = current_dir.parent
            
            self.data_path = current_dir / data_path
        self.data = None
        self.detection_results = {}
        self.duplicates_found = {}
        self.inconsistencies_found = {}
        
        # Define detection priorities
        self.detection_priorities = {
            'work_id_duplicates': 'CRITICAL',
            'title_duplicates': 'HIGH',
            'series_inconsistencies': 'HIGH',
            'author_consistency': 'MEDIUM',
            'genre_patterns': 'MEDIUM',
            'publication_year_validation': 'MEDIUM'
        }
        
        logger.info("DuplicateInconsistencyDetection initialized")
    
    def load_cleaned_dataset(self) -> bool:
        """Load the cleaned dataset for analysis."""
        try:
            # Look for the cleaned dataset
            cleaned_files = list(self.data_path.glob("final_books_*_cleaned_nlp_ready_*.csv"))
            if not cleaned_files:
                logger.error("No cleaned dataset found")
                return False
            
            # Use the most recent cleaned dataset
            cleaned_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            cleaned_file = cleaned_files[0]
            
            logger.info(f"Loading cleaned dataset: {cleaned_file.name}")
            self.data = pd.read_csv(cleaned_file)
            
            logger.info(f"Dataset loaded: {len(self.data):,} books, {len(self.data.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cleaned dataset: {e}")
            return False
    
    def detect_work_id_duplicates(self) -> Dict[str, Any]:
        """
        Detect duplicate work IDs (CRITICAL priority).
        
        Returns:
            Dict containing duplicate detection results
        """
        logger.info("🔍 Detecting work ID duplicates...")
        
        if self.data is None:
            return {}
        
        # Check for duplicate work IDs
        work_id_duplicates = self.data['work_id'].duplicated().sum()
        work_id_unique = self.data['work_id'].nunique()
        total_books = len(self.data)
        
        result = {
            'duplicates_found': work_id_duplicates,
            'unique_work_ids': work_id_unique,
            'total_books': total_books,
            'status': 'PASS' if work_id_duplicates == 0 else 'FAIL',
            'priority': 'CRITICAL',
            'details': {}
        }
        
        if work_id_duplicates > 0:
            logger.warning(f"Found {work_id_duplicates} duplicate work IDs!")
            result['details']['duplicate_work_ids'] = self.data[
                self.data['work_id'].duplicated(keep=False)
            ]['work_id'].value_counts().to_dict()
        else:
            logger.info("✅ No duplicate work IDs found")
        
        self.detection_results['work_id_duplicates'] = result
        return result
    
    def detect_title_duplicates(self) -> Dict[str, Any]:
        """
        Detect duplicate titles (HIGH priority).
        
        Returns:
            Dict containing title duplicate detection results
        """
        logger.info("🔍 Detecting title duplicates...")
        
        if self.data is None:
            return {}
        
        # Check for exact title duplicates
        title_duplicates = self.data['title'].duplicated().sum()
        title_unique = self.data['title'].nunique()
        total_books = len(self.data)
        duplication_rate = (title_duplicates / total_books) * 100
        
        result = {
            'duplicates_found': title_duplicates,
            'unique_titles': title_unique,
            'total_books': total_books,
            'duplication_rate': duplication_rate,
            'status': 'PASS' if title_duplicates == 0 else 'WARNING',
            'priority': 'HIGH',
            'details': {}
        }
        
        if title_duplicates > 0:
            logger.warning(f"Found {title_duplicates} duplicate titles ({duplication_rate:.1f}%)")
            
            # Get examples of duplicate titles
            duplicate_titles = self.data[self.data['title'].duplicated(keep=False)]['title'].value_counts()
            result['details']['duplicate_titles'] = duplicate_titles.head(10).to_dict()
            
            # Show top duplicates
            print(f"  📖 Top duplicate titles:")
            for title, count in duplicate_titles.head(5).items():
                print(f"    - '{title}': {count} books")
        else:
            logger.info("✅ No duplicate titles found")
        
        self.detection_results['title_duplicates'] = result
        return result
    
    def detect_series_inconsistencies(self) -> Dict[str, Any]:
        """
        Detect series data inconsistencies (HIGH priority).
        
        Returns:
            Dict containing series inconsistency detection results
        """
        logger.info("🔍 Detecting series inconsistencies...")
        
        if self.data is None:
            return {}
        
        # Check series data
        series_books = self.data['series_id'].notna().sum()
        standalone_books = len(self.data) - series_books
        
        result = {
            'series_books': series_books,
            'standalone_books': standalone_books,
            'series_coverage': (series_books / len(self.data)) * 100,
            'inconsistencies_found': 0,
            'status': 'PASS',
            'priority': 'HIGH',
            'details': {}
        }
        
        # Check for series count inconsistencies
        if 'series_works_count' in self.data.columns:
            series_inconsistencies = []
            
            # Sample series for analysis (to avoid performance issues)
            series_sample = self.data[self.data['series_id'].notna()]['series_id'].drop_duplicates().head(1000)
            
            for series_id in series_sample:
                series_books = self.data[self.data['series_id'] == series_id]
                if len(series_books) > 0:
                    expected_count = series_books['series_works_count'].iloc[0]
                    actual_count = len(series_books)
                    
                    if pd.notna(expected_count) and expected_count != actual_count:
                        series_inconsistencies.append({
                            'series_id': series_id,
                            'expected': expected_count,
                            'actual': actual_count,
                            'difference': actual_count - expected_count
                        })
            
            result['inconsistencies_found'] = len(series_inconsistencies)
            result['details']['series_inconsistencies'] = series_inconsistencies[:10]  # First 10
            
            if series_inconsistencies:
                logger.warning(f"Found {len(series_inconsistencies)} series count inconsistencies")
                result['status'] = 'WARNING'
                
                # Show examples
                print(f"  📚 Series count inconsistencies (sample):")
                for inc in series_inconsistencies[:5]:
                    print(f"    - Series {inc['series_id']:.0f}: expected {inc['expected']}, found {inc['actual']} (diff: {inc['difference']:+.0f})")
            else:
                logger.info("✅ Series counts appear consistent")
        
        self.detection_results['series_inconsistencies'] = result
        return result
    
    def detect_author_consistency(self) -> Dict[str, Any]:
        """
        Detect author attribution inconsistencies (MEDIUM priority).
        
        Returns:
            Dict containing author consistency detection results
        """
        logger.info("🔍 Detecting author consistency issues...")
        
        if self.data is None:
            return {}
        
        # Analyze author distribution
        author_unique = self.data['author_id'].nunique()
        total_books = len(self.data)
        avg_books_per_author = total_books / author_unique
        
        # Check for authors with unusually many books
        author_book_counts = self.data['author_id'].value_counts()
        prolific_authors = author_book_counts[author_book_counts > 50]  # More than 50 books
        
        result = {
            'unique_authors': author_unique,
            'total_books': total_books,
            'avg_books_per_author': avg_books_per_author,
            'prolific_authors': len(prolific_authors),
            'max_books_by_author': author_book_counts.max(),
            'status': 'PASS',
            'priority': 'MEDIUM',
            'details': {}
        }
        
        # Check for potential issues
        if avg_books_per_author > 10:
            result['status'] = 'WARNING'
            result['details']['high_avg_books'] = f"High average books per author: {avg_books_per_author:.1f}"
        
        if len(prolific_authors) > 0:
            result['details']['prolific_authors'] = prolific_authors.head(10).to_dict()
            print(f"  👤 Prolific authors (50+ books):")
            for author_id, count in prolific_authors.head(5).items():
                author_name = self.data[self.data['author_id'] == author_id]['author_name'].iloc[0]
                print(f"    - {author_name} (ID: {author_id}): {count} books")
        
        logger.info(f"Author analysis: {author_unique:,} authors, avg {avg_books_per_author:.1f} books/author")
        
        self.detection_results['author_consistency'] = result
        return result
    
    def detect_genre_patterns(self) -> Dict[str, Any]:
        """
        Detect genre classification patterns and inconsistencies (MEDIUM priority).
        
        Returns:
            Dict containing genre pattern detection results
        """
        logger.info("🔍 Detecting genre classification patterns...")
        
        if self.data is None:
            return {}
        
        # Analyze genre distribution
        genre_counts = self.data['genres'].value_counts()
        unique_genres = len(genre_counts)
        top_genres = genre_counts.head(10)
        
        # Check for genre consistency patterns
        genre_patterns = {}
        for genre_combo in genre_counts.index:
            genre_list = [g.strip() for g in genre_combo.split(',')]
            genre_patterns[genre_combo] = {
                'count': genre_counts[genre_combo],
                'genre_count': len(genre_list),
                'has_romance': 'romance' in [g.lower() for g in genre_list]
            }
        
        result = {
            'unique_genre_combinations': unique_genres,
            'total_books': len(self.data),
            'top_genres': top_genres.to_dict(),
            'genre_patterns': genre_patterns,
            'status': 'PASS',
            'priority': 'MEDIUM',
            'details': {}
        }
        
        # Check for potential issues
        romance_missing = sum(1 for pattern in genre_patterns.values() if not pattern['has_romance'])
        if romance_missing > 0:
            result['status'] = 'WARNING'
            result['details']['missing_romance'] = f"{romance_missing} genre combinations missing 'romance'"
        
        logger.info(f"Genre analysis: {unique_genres:,} unique combinations")
        
        # Show top genres
        print(f"  🏷️  Top genre combinations:")
        for i, (genre, count) in enumerate(top_genres.head(5).items(), 1):
            print(f"    {i}. {genre:50} | {count:5,} books")
        
        self.detection_results['genre_patterns'] = result
        return result
    
    def validate_publication_years(self) -> Dict[str, Any]:
        """
        Validate publication year ranges (MEDIUM priority).
        
        Returns:
            Dict containing publication year validation results
        """
        logger.info("🔍 Validating publication years...")
        
        if self.data is None:
            return {}
        
        # Check year range
        min_year = self.data['publication_year'].min()
        max_year = self.data['publication_year'].max()
        expected_min, expected_max = 2000, 2020
        
        # Check for out-of-range years
        out_of_range = self.data[~self.data['publication_year'].between(expected_min, expected_max)]
        out_of_range_count = len(out_of_range)
        
        result = {
            'min_year': min_year,
            'max_year': max_year,
            'expected_range': (expected_min, expected_max),
            'out_of_range_count': out_of_range_count,
            'status': 'PASS' if out_of_range_count == 0 else 'WARNING',
            'priority': 'MEDIUM',
            'details': {}
        }
        
        if out_of_range_count > 0:
            result['details']['out_of_range_years'] = sorted(out_of_range['publication_year'].unique())
            logger.warning(f"Found {out_of_range_count} books outside expected year range {expected_min}-{expected_max}")
            
            print(f"  📅 Out-of-range publication years:")
            for year in sorted(out_of_range['publication_year'].unique()):
                count = (out_of_range['publication_year'] == year).sum()
                print(f"    - {year}: {count} books")
        else:
            logger.info(f"✅ Publication years within expected range: {min_year}-{max_year}")
        
        self.detection_results['publication_year_validation'] = result
        return result
    
    def run_full_detection(self) -> Dict[str, Any]:
        """
        Run all duplicate and inconsistency detection checks.
        
        Returns:
            Dict containing all detection results
        """
        logger.info("🚀 Starting comprehensive duplicate and inconsistency detection...")
        
        # Load dataset
        if not self.load_cleaned_dataset():
            return {}
        
        print(f"🔍 STEP 2: Duplicate and Inconsistency Detection")
        print(f"=" * 60)
        print(f"Analyzing {len(self.data):,} books for data quality issues")
        print(f"=" * 60)
        
        # Run all detection methods
        self.detect_work_id_duplicates()
        self.detect_title_duplicates()
        self.detect_series_inconsistencies()
        self.detect_author_consistency()
        self.detect_genre_patterns()
        self.validate_publication_years()
        
        # Generate summary
        summary = self.generate_detection_summary()
        print(summary)
        
        # Save detailed report
        report_path = self.save_detection_report()
        
        logger.info("Duplicate and inconsistency detection completed")
        
        return self.detection_results
    
    def generate_detection_summary(self) -> str:
        """Generate a comprehensive detection summary."""
        if not self.detection_results:
            return "No detection results available"
        
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("STEP 2: DUPLICATE AND INCONSISTENCY DETECTION SUMMARY")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Dataset: {len(self.data):,} books")
        summary_lines.append("")
        
        # Summary by priority
        for priority in ['CRITICAL', 'HIGH', 'MEDIUM']:
            priority_results = {k: v for k, v in self.detection_results.items() 
                              if v.get('priority') == priority}
            
            if priority_results:
                summary_lines.append(f"{priority} PRIORITY ISSUES:")
                summary_lines.append("-" * 30)
                
                for check_name, result in priority_results.items():
                    status_icon = "✅" if result['status'] == 'PASS' else "⚠️" if result['status'] == 'WARNING' else "❌"
                    summary_lines.append(f"  {status_icon} {check_name.replace('_', ' ').title()}: {result['status']}")
                    
                    if result['status'] != 'PASS':
                        if 'duplicates_found' in result and result['duplicates_found'] > 0:
                            summary_lines.append(f"    - Duplicates: {result['duplicates_found']:,}")
                        if 'inconsistencies_found' in result and result['inconsistencies_found'] > 0:
                            summary_lines.append(f"    - Inconsistencies: {result['inconsistencies_found']:,}")
                
                summary_lines.append("")
        
        # Overall assessment
        total_issues = sum(1 for r in self.detection_results.values() if r['status'] != 'PASS')
        summary_lines.append(f"OVERALL ASSESSMENT:")
        summary_lines.append(f"  Total issues found: {total_issues}")
        
        if total_issues == 0:
            summary_lines.append("  🎉 Dataset passes all quality checks!")
        else:
            summary_lines.append("  ⚠️  Dataset has quality issues that need attention")
        
        summary_lines.append("")
        summary_lines.append("NEXT STEPS:")
        summary_lines.append("  1. Review detailed detection report")
        summary_lines.append("  2. Plan data cleaning strategies for identified issues")
        summary_lines.append("  3. Proceed to Step 3: Outlier Detection and Treatment")
        
        return "\n".join(summary_lines)
    
    def save_detection_report(self) -> str:
        """Save detailed detection report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"step2_duplicate_inconsistency_detection_report_{timestamp}.txt"
        
        output_path = Path("../../data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                f.write(self.generate_detection_summary())
                f.write("\n\n" + "="*60 + "\n")
                f.write("DETAILED DETECTION RESULTS\n")
                f.write("="*60 + "\n\n")
                
                for check_name, result in self.detection_results.items():
                    f.write(f"{check_name.upper().replace('_', ' ')}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(json.dumps(result, indent=2, default=str))
                    f.write("\n\n")
            
            logger.info(f"Detection report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save detection report: {e}")
            return ""


def main():
    """Main function to run duplicate and inconsistency detection."""
    print("🔍 Step 2: Duplicate and Inconsistency Detection")
    print("=" * 60)
    
    # Initialize detection
    detector = DuplicateInconsistencyDetection()
    
    # Run full detection
    results = detector.run_full_detection()
    
    if results:
        print(f"\n✅ Detection completed successfully!")
        print(f"📋 Report saved to: {results.get('report_path', 'Unknown')}")
    else:
        print(f"\n❌ Detection failed!")
    
    return results


if __name__ == "__main__":
    main()
