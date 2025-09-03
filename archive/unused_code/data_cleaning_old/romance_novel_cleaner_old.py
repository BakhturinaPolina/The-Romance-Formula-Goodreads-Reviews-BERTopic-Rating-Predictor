#!/usr/bin/env python3
"""
Data Cleaning Pipeline for Romance Novel Dataset
Implements cleaning steps identified in the EDA analysis.
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RomanceNovelDataCleaner:
    """
    Comprehensive data cleaner for romance novel dataset.
    
    Implements cleaning steps identified in EDA:
    1. Title normalization (series extraction, numbering standardization)
    2. Author name normalization and deduplication
    3. Description text cleaning (HTML removal, whitespace normalization)
    4. Series information standardization
    5. Subgenre classification from popular shelves
    6. Data quality improvements
    """
    
    def __init__(self, input_path: str, output_dir: str = "data/processed"):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Improved series patterns for title cleaning - more conservative
        self.series_patterns = [
            r'\b(\d+)\s*[:\-]\s*',  # Number followed by : or -
            r'\b(Book|Volume|Part)\s+(\d+)\b',  # Book 1, Volume 2, etc.
            r'\b(\d+)\s*(?:st|nd|rd|th)\s*',  # 1st, 2nd, 3rd, etc.
            r'\b(\d+)\s*$',  # Number at end
            r'\b(\d+)\s*\('  # Number followed by parenthesis
        ]
        
        # Target subgenres for classification - expanded list
        self.target_subgenres = [
            'contemporary romance', 'historical romance', 'paranormal romance',
            'romantic suspense', 'romantic fantasy', 'science fiction romance',
            'chick lit', 'chick-lit', 'chicklit', 'new adult', 'new-adult',
            'billionaire romance', 'military romance', 'cowboy romance',
            'vampire romance', 'werewolf romance', 'shifter romance',
            'medieval romance', 'regency romance', 'victorian romance',
            'highland romance', 'scottish romance', 'irish romance'
        ]
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset for cleaning."""
        logger.info(f"Loading dataset from {self.input_path}")
        df = pd.read_csv(self.input_path)
        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
        return df
    
    def clean_titles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean book titles by removing series information and standardizing numbering.
        Uses conservative approach to avoid removing legitimate content.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with cleaned titles and extracted series information
        """
        logger.info("Cleaning book titles with conservative approach...")
        
        # Create new columns for extracted information
        df['title_original'] = df['title'].copy()
        df['series_number_extracted'] = None
        df['title_cleaned'] = df['title'].copy()
        
        # Extract series numbers from titles using more specific patterns
        series_numbers_found = 0
        
        for idx, row in df.iterrows():
            title = str(row['title'])
            series_number = None
            
            # Pattern 1: "Book 1: Title" or "Volume 2 - Title"
            match = re.search(r'\b(Book|Volume|Part)\s+(\d+)\s*[:\-]\s*(.+)', title, re.IGNORECASE)
            if match:
                series_number = int(match.group(2))
                # Only clean if we have a substantial title left
                remaining_title = match.group(3).strip()
                if len(remaining_title) > 3:  # Ensure we have meaningful content
                    df.loc[idx, 'title_cleaned'] = remaining_title
                    series_numbers_found += 1
            
            # Pattern 2: "Title (Book 1)" or "Title (Volume 2)"
            elif re.search(r'\b(Book|Volume|Part)\s+\d+\s*\)', title, re.IGNORECASE):
                match = re.search(r'\b(Book|Volume|Part)\s+(\d+)', title, re.IGNORECASE)
                if match:
                    series_number = int(match.group(2))
                    # Remove the parenthetical series info
                    cleaned_title = re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()
                    if cleaned_title and len(cleaned_title) > 3:
                        df.loc[idx, 'title_cleaned'] = cleaned_title
                        series_numbers_found += 1
            
            # Pattern 3: "Title #1" or "Title #2" (only if # is clearly series marker)
            elif re.search(r'\s#\d+\s*$', title):
                match = re.search(r'(.+?)\s#(\d+)\s*$', title)
                if match:
                    series_number = int(match.group(2))
                    # Only clean if the remaining title is substantial
                    remaining_title = match.group(1).strip()
                    if len(remaining_title) > 3:
                        df.loc[idx, 'title_cleaned'] = remaining_title
                        series_numbers_found += 1
            
            # Store extracted series number
            if series_number is not None:
                df.loc[idx, 'series_number_extracted'] = series_number
        
        # Remove series titles embedded in book titles (only exact matches)
        series_books = df[df['series_title'].notna()].copy()
        titles_cleaned_from_series = 0
        
        for idx, row in series_books.iterrows():
            if pd.notna(row['series_title']) and pd.notna(row['title']):
                series_title = str(row['series_title']).strip()
                book_title = str(row['title'])
                
                # Only remove if series title is at the beginning and followed by separator
                if book_title.lower().startswith(series_title.lower()):
                    # Check for common separators
                    separators = [' - ', ': ', ' (', ' [', ' #']
                    for sep in separators:
                        if sep in book_title[len(series_title):len(series_title)+3]:
                            # Extract title after separator
                            parts = book_title.split(sep, 1)
                            if len(parts) > 1:
                                remaining_title = parts[1].strip()
                                # Only clean if we have substantial content left
                                if len(remaining_title) > 3:
                                    df.loc[idx, 'title_cleaned'] = remaining_title
                                    titles_cleaned_from_series += 1
                                    break
        
        logger.info(f"Title cleaning completed conservatively:")
        logger.info(f"  - Series numbers extracted: {series_numbers_found:,}")
        logger.info(f"  - Titles cleaned from series info: {titles_cleaned_from_series:,}")
        logger.info(f"  - Total titles modified: {(df['title_original'] != df['title_cleaned']).sum():,}")
        
        return df
    
    def clean_author_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize author names and identify potential duplicates.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with normalized author names and duplicate flags
        """
        logger.info("Cleaning author names...")
        
        # Create normalized author name column
        df['author_name_normalized'] = df['author_name'].str.strip().str.title()
        
        # Identify potential duplicate authors (same name, different IDs)
        author_name_to_ids = defaultdict(list)
        for _, row in df.iterrows():
            author_name_to_ids[row['author_name_normalized']].append(row['author_id'])
        
        # Flag potential duplicates
        df['author_potential_duplicate'] = df['author_name_normalized'].apply(
            lambda x: len(author_name_to_ids[x]) > 1
        )
        
        duplicate_count = df['author_potential_duplicate'].sum()
        logger.info(f"Identified {duplicate_count:,} books with potentially duplicate author names")
        
        return df
    
    def clean_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean book descriptions by removing HTML and normalizing text.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with cleaned descriptions
        """
        logger.info("Cleaning book descriptions...")
        
        # Create new columns
        df['description_original'] = df['description'].copy()
        df['description_cleaned'] = df['description'].copy()
        
        # Apply cleaning to non-null descriptions
        mask = df['description'].notna()
        
        # Remove HTML tags
        df.loc[mask, 'description_cleaned'] = df.loc[mask, 'description_cleaned'].apply(
            lambda x: re.sub(r'<[^>]+>', '', str(x))
        )
        
        # Remove HTML entities
        df.loc[mask, 'description_cleaned'] = df.loc[mask, 'description_cleaned'].apply(
            lambda x: re.sub(r'&[a-zA-Z]+;', ' ', str(x))
        )
        
        # Normalize whitespace
        df.loc[mask, 'description_cleaned'] = df.loc[mask, 'description_cleaned'].apply(
            lambda x: re.sub(r'\s+', ' ', str(x))
        )
        
        # Remove line breaks and tabs
        df.loc[mask, 'description_cleaned'] = df.loc[mask, 'description_cleaned'].apply(
            lambda x: re.sub(r'[\r\n\t]+', ' ', str(x))
        )
        
        # Clean up and strip
        df.loc[mask, 'description_cleaned'] = df.loc[mask, 'description_cleaned'].apply(
            lambda x: str(x).strip()
        )
        
        # Calculate cleaning statistics
        original_lengths = df.loc[mask, 'description_original'].str.len()
        cleaned_lengths = df.loc[mask, 'description_cleaned'].str.len()
        length_reduction = ((original_lengths - cleaned_lengths) / original_lengths * 100).mean()
        
        logger.info(f"Description cleaning completed. Average length reduction: {length_reduction:.1f}%")
        return df
    
    def standardize_series_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize series information and create clean series titles.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with standardized series information
        """
        logger.info("Standardizing series information...")
        
        # Create new columns
        df['series_title_original'] = df['series_title'].copy()
        df['series_title_cleaned'] = df['series_title'].copy()
        
        # Clean series titles
        mask = df['series_title'].notna()
        df.loc[mask, 'series_title_cleaned'] = df.loc[mask, 'series_title_cleaned'].apply(
            lambda x: str(x).strip().title()
        )
        
        # Create series position column if not exists
        if 'series_position' not in df.columns:
            df['series_position'] = None
        
        # Fill missing series information where possible
        # Priority 1: Use extracted series numbers from titles
        series_mask = (df['series_id'].notna()) & (df['series_number_extracted'].notna())
        if series_mask.sum() > 0:
            # Ensure series_position is numeric
            df['series_position'] = pd.to_numeric(df['series_position'], errors='coerce')
            df.loc[series_mask, 'series_position'] = df.loc[series_mask, 'series_number_extracted']
        
        # Priority 2: Try to extract from existing series_position if it's a string
        if 'series_position' in df.columns:
            string_positions = df[df['series_position'].notna() & (df['series_position'].astype(str).str.contains(r'\d+', na=False))]
            for idx in string_positions.index:
                pos_str = str(df.loc[idx, 'series_position'])
                # Extract first number found
                match = re.search(r'(\d+)', pos_str)
                if match:
                    df.loc[idx, 'series_position'] = int(match.group(1))
        
        logger.info(f"Series standardization completed:")
        logger.info(f"  - Books with position information: {df['series_position'].notna().sum():,}")
        logger.info(f"  - Series titles cleaned: {mask.sum():,}")
        
        return df
    
    def classify_subgenres(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify books into subgenres based on popular shelves.
        Handles comma-separated format properly.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with subgenre classifications
        """
        logger.info("Classifying subgenres from popular shelves...")
        
        # Initialize subgenre columns
        for subgenre in self.target_subgenres:
            df[f'subgenre_{subgenre.replace(" ", "_")}'] = False
        
        # Parse popular shelves and classify
        subgenre_counts = defaultdict(int)
        total_classified = 0
        
        for idx, row in df.iterrows():
            if pd.notna(row['popular_shelves']):
                shelves = str(row['popular_shelves'])
                
                # Handle comma-separated format (which is what we have)
                shelves_list = [s.strip().lower() for s in shelves.split(',')]
                
                # Classify based on shelf content
                book_subgenres = []
                for shelf in shelves_list:
                    for subgenre in self.target_subgenres:
                        if subgenre in shelf:
                            df.loc[idx, f'subgenre_{subgenre.replace(" ", "_")}'] = True
                            subgenre_counts[subgenre] += 1
                            if subgenre not in book_subgenres:
                                book_subgenres.append(subgenre)
                
                if book_subgenres:
                    total_classified += 1
        
        # Create primary subgenre column
        df['subgenre_primary'] = None
        for idx, row in df.iterrows():
            primary_subgenres = []
            for subgenre in self.target_subgenres:
                if df.loc[idx, f'subgenre_{subgenre.replace(" ", "_")}']:
                    primary_subgenres.append(subgenre)
            
            if primary_subgenres:
                df.loc[idx, 'subgenre_primary'] = primary_subgenres[0]  # Take first match
        
        # Log classification results
        logger.info(f"Subgenre classification completed:")
        logger.info(f"  - Total books classified: {total_classified:,}")
        logger.info(f"  - Classification rate: {(total_classified/len(df)*100):.1f}%")
        
        for subgenre, count in subgenre_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  - {subgenre}: {count:,} books ({percentage:.1f}%)")
        
        return df
    
    def improve_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Improve overall data quality by handling missing values and inconsistencies.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with improved data quality
        """
        logger.info("Improving overall data quality...")
        
        # Handle missing median_publication_year
        missing_median_year = df['median_publication_year'].isnull().sum()
        if missing_median_year > 0:
            # Use publication_year as fallback
            df.loc[df['median_publication_year'].isnull(), 'median_publication_year'] = \
                df.loc[df['median_publication_year'].isnull(), 'publication_year']
            logger.info(f"Filled {missing_median_year:,} missing median_publication_year values")
        
        # Create comprehensive data quality score
        quality_columns = [
            'title', 'description', 'author_name', 'series_title',
            'publication_year', 'average_rating_weighted_mean', 'ratings_count',
            'series_id', 'series_position'
        ]
        
        # Filter to columns that actually exist
        existing_quality_columns = [col for col in quality_columns if col in df.columns]
        
        df['data_quality_score'] = 0.0
        for col in existing_quality_columns:
            if col in df.columns:
                # Add 1 point for each non-null value
                df['data_quality_score'] += df[col].notna().astype(int)
        
        # Normalize to 0-1 scale
        df['data_quality_score'] = df['data_quality_score'] / len(existing_quality_columns)
        
        # Calculate quality statistics
        avg_quality = df['data_quality_score'].mean()
        quality_distribution = df['data_quality_score'].value_counts().sort_index()
        
        logger.info(f"Data quality improvement completed:")
        logger.info(f"  - Average quality score: {avg_quality:.3f}")
        logger.info(f"  - Quality distribution:")
        for score, count in quality_distribution.items():
            percentage = count/len(df)*100
            logger.info(f"    {score:.3f}: {count:,} records ({percentage:.1f}%)")
        
        return df
    
    def validate_cleaning_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the cleaning results and provide detailed statistics.
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating cleaning results...")
        
        validation_results = {
            'total_records': len(df),
            'title_cleaning': {},
            'author_cleaning': {},
            'description_cleaning': {},
            'series_cleaning': {},
            'subgenre_classification': {},
            'data_quality': {}
        }
        
        # Title cleaning validation
        titles_cleaned = (df['title_original'] != df['title_cleaned']).sum()
        validation_results['title_cleaning'] = {
            'titles_modified': titles_cleaned,
            'modification_rate': titles_cleaned / len(df) * 100,
            'series_numbers_extracted': df['series_number_extracted'].notna().sum()
        }
        
        # Author cleaning validation
        author_duplicates = df['author_potential_duplicate'].sum()
        validation_results['author_cleaning'] = {
            'potential_duplicates': author_duplicates,
            'duplicate_rate': author_duplicates / len(df) * 100
        }
        
        # Description cleaning validation
        descriptions_cleaned = (df['description_original'] != df['description_cleaned']).sum()
        validation_results['description_cleaning'] = {
            'descriptions_modified': descriptions_cleaned,
            'modification_rate': descriptions_cleaned / len(df) * 100
        }
        
        # Series cleaning validation
        series_positions = df['series_position'].notna().sum()
        validation_results['series_cleaning'] = {
            'books_with_position': series_positions,
            'position_coverage': series_positions / len(df) * 100
        }
        
        # Subgenre classification validation
        subgenre_cols = [col for col in df.columns if col.startswith('subgenre_') and not col.endswith('_primary')]
        total_classifications = df[subgenre_cols].sum().sum()
        books_classified = (df[subgenre_cols].sum(axis=1) > 0).sum()
        
        validation_results['subgenre_classification'] = {
            'total_classifications': total_classifications,
            'books_classified': books_classified,
            'classification_rate': books_classified / len(df) * 100
        }
        
        # Data quality validation
        avg_quality = df['data_quality_score'].mean()
        validation_results['data_quality'] = {
            'average_score': avg_quality,
            'quality_distribution': df['data_quality_score'].value_counts().sort_index().to_dict()
        }
        
        # Log validation summary
        logger.info("Cleaning validation completed:")
        logger.info(f"  - Title modification rate: {validation_results['title_cleaning']['modification_rate']:.1f}%")
        logger.info(f"  - Author duplicate rate: {validation_results['author_cleaning']['duplicate_rate']:.1f}%")
        logger.info(f"  - Description modification rate: {validation_results['description_cleaning']['modification_rate']:.1f}%")
        logger.info(f"  - Series position coverage: {validation_results['series_cleaning']['position_coverage']:.1f}%")
        logger.info(f"  - Subgenre classification rate: {validation_results['subgenre_classification']['classification_rate']:.1f}%")
        logger.info(f"  - Average data quality score: {validation_results['data_quality']['average_score']:.3f}")
        
        return validation_results
    
    def run_cleaning_pipeline(self, output_filename: str = None) -> str:
        """
        Run the complete cleaning pipeline.
        
        Args:
            output_filename: Optional custom output filename
            
        Returns:
            Path to the cleaned dataset
        """
        logger.info("Starting comprehensive data cleaning pipeline...")
        
        # Load dataset
        df = self.load_dataset()
        
        # Apply cleaning steps
        df = self.clean_titles(df)
        df = self.clean_author_names(df)
        df = self.clean_descriptions(df)
        df = self.standardize_series_info(df)
        df = self.classify_subgenres(df)
        df = self.improve_data_quality(df)
        
        # Validate cleaning results
        validation_results = self.validate_cleaning_results(df)
        
        # Generate output filename
        if output_filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"romance_novels_cleaned_{timestamp}.csv"
        
        output_path = self.output_dir / output_filename
        
        # Save cleaned dataset
        logger.info(f"Saving cleaned dataset to {output_path}")
        df.to_csv(output_path, index=False)
        
        # Generate cleaning report
        self._generate_cleaning_report(df, output_path, validation_results)
        
        logger.info("Data cleaning pipeline completed successfully!")
        return str(output_path)
    
    def _generate_cleaning_report(self, df: pd.DataFrame, output_path: Path, validation_results: Dict[str, Any]):
        """Generate a summary report of the cleaning process."""
        report_path = output_path.parent / f"{output_path.stem}_cleaning_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("ROMANCE NOVEL DATASET CLEANING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.input_path.name}\n")
            f.write(f"Output: {output_path.name}\n")
            f.write(f"Records: {len(df):,}\n")
            f.write(f"Columns: {len(df.columns)}\n\n")
            
            f.write("CLEANING SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            # Title cleaning
            titles_cleaned = (df['title_original'] != df['title_cleaned']).sum()
            f.write(f"Titles cleaned: {titles_cleaned:,}\n")
            
            # Series numbers extracted
            series_numbers = df['series_number_extracted'].notna().sum()
            f.write(f"Series numbers extracted: {series_numbers:,}\n")
            
            # Author duplicates identified
            author_duplicates = df['author_potential_duplicate'].sum()
            f.write(f"Potential author duplicates: {author_duplicates:,}\n")
            
            # Descriptions cleaned
            descriptions_cleaned = (df['description_original'] != df['description_cleaned']).sum()
            f.write(f"Descriptions cleaned: {descriptions_cleaned:,}\n")
            
            # Subgenre classifications
            subgenre_cols = [col for col in df.columns if col.startswith('subgenre_') and not col.endswith('_primary')]
            total_classifications = df[subgenre_cols].sum().sum()
            f.write(f"Subgenre classifications: {total_classifications:,}\n")
            
            # Data quality
            avg_quality = df['data_quality_score'].mean()
            f.write(f"Average data quality score: {avg_quality:.3f}\n")

            # Validation Summary
            f.write("\nVALIDATION SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Records: {validation_results['total_records']:,}\n")
            f.write(f"Title Modification Rate: {validation_results['title_cleaning']['modification_rate']:.1f}%\n")
            f.write(f"Author Duplicate Rate: {validation_results['author_cleaning']['duplicate_rate']:.1f}%\n")
            f.write(f"Description Modification Rate: {validation_results['description_cleaning']['modification_rate']:.1f}%\n")
            f.write(f"Series Position Coverage: {validation_results['series_cleaning']['position_coverage']:.1f}%\n")
            f.write(f"Subgenre Classification Rate: {validation_results['subgenre_classification']['classification_rate']:.1f}%\n")
            f.write(f"Average Data Quality Score: {validation_results['data_quality']['average_score']:.3f}\n")
        
        logger.info(f"Cleaning report saved to {report_path}")

def main():
    """Run the data cleaning pipeline."""
    input_file = "data/processed/final_books_2000_2020_en_20250901_024106.csv"
    
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Initialize cleaner
    cleaner = RomanceNovelDataCleaner(input_file)
    
    # Run cleaning pipeline
    output_path = cleaner.run_cleaning_pipeline()
    
    print(f"\n✅ Data cleaning completed successfully!")
    print(f"📁 Cleaned dataset saved to: {output_path}")
    print(f"📊 Check the cleaning report for detailed statistics")

if __name__ == "__main__":
    main()
