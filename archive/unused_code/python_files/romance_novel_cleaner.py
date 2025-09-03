#!/usr/bin/env python3
"""
Improved Data Cleaning Pipeline for Romance Novel Dataset
Implements cleaning steps with comprehensive type hints, rollback mechanisms, and performance monitoring.
"""

import pandas as pd
import numpy as np
import re
import time
import psutil
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a cleaning step."""
    step_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    memory_delta: float
    records_processed: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class CleaningCheckpoint:
    """Checkpoint for rollback functionality."""
    step_name: str
    timestamp: float
    dataframe_shape: Tuple[int, int]
    memory_usage: float

class RomanceNovelDataCleaner:
    """
    Comprehensive data cleaner for romance novel dataset with advanced features.
    
    Features:
    - Comprehensive type hints
    - Rollback mechanisms for failed cleaning steps
    - Performance monitoring and memory tracking
    - Text cleaning and normalization (no subgenre classification)
    - Checkpoint-based error recovery
    """
    
    def __init__(self, input_path: Union[str, Path], output_dir: Union[str, Path] = "data/processed") -> None:
        """
        Initialize the data cleaner.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory for output files
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []
        self.checkpoints: List[CleaningCheckpoint] = []
        
        # Series patterns for title cleaning - more conservative
        self.series_patterns: List[str] = [
            r'\b(\d+)\s*[:\-]\s*',  # Number followed by : or -
            r'\b(Book|Volume|Part)\s+(\d+)\b',  # Book 1, Volume 2, etc.
            r'\b(\d+)\s*(?:st|nd|rd|th)\s*',  # 1st, 2nd, 3rd, etc.
            r'\b(\d+)\s*$',  # Number at end
            r'\b(\d+)\s*\('  # Number followed by parenthesis
        ]
        
        # Text cleaning patterns for popular_shelves
        self.text_cleaning_patterns: Dict[str, str] = {
            'multiple_spaces': r'\s+',
            'leading_trailing_spaces': r'^\s+|\s+$',
            'multiple_commas': r',+',
            'empty_entries': r'^,+$|^,+|,+$'
        }
        
        # Setup logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the cleaner."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _monitor_performance(self, step_name: str, df: pd.DataFrame, func: callable, *args, **kwargs) -> pd.DataFrame:
        """
        Monitor performance of a cleaning step.
        
        Args:
            step_name: Name of the cleaning step
            df: DataFrame being processed
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Processed DataFrame
        """
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        records_processed = len(df)
        
        try:
            result = func(df, *args, **kwargs)
            success = True
            error_message = None
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            metrics = PerformanceMetrics(
                step_name=step_name,
                execution_time=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_after - memory_before,
                records_processed=records_processed,
                success=success,
                error_message=error_message
            )
            
            self.performance_metrics.append(metrics)
            
            self.logger.info(
                f"Step '{step_name}' completed: "
                f"Time: {metrics.execution_time:.2f}s, "
                f"Memory: {metrics.memory_delta:+.1f}MB, "
                f"Success: {success}"
            )
        
        return result
    
    def _create_checkpoint(self, step_name: str, df: pd.DataFrame) -> CleaningCheckpoint:
        """
        Create a checkpoint for potential rollback.
        
        Args:
            step_name: Name of the cleaning step
            df: DataFrame to checkpoint
            
        Returns:
            CleaningCheckpoint object
        """
        checkpoint = CleaningCheckpoint(
            step_name=step_name,
            timestamp=time.time(),
            dataframe_shape=df.shape,
            memory_usage=psutil.Process().memory_info().rss / 1024 / 1024
        )
        
        self.checkpoints.append(checkpoint)
        self.logger.info(f"Checkpoint created for step '{step_name}'")
        
        return checkpoint
    
    def _rollback_to_checkpoint(self, checkpoint: CleaningCheckpoint, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rollback to a specific checkpoint.
        
        Args:
            checkpoint: Checkpoint to rollback to
            df: Current DataFrame
            
        Returns:
            DataFrame rolled back to checkpoint state
        """
        self.logger.warning(f"Rolling back to checkpoint '{checkpoint.step_name}'")
        
        # Remove columns created after this checkpoint
        checkpoint_index = next(i for i, cp in enumerate(self.checkpoints) if cp == checkpoint)
        
        # Define columns created by each step
        step_columns = {
            "title_cleaning": ["title_original", "title_cleaned", "series_number_extracted"],
            "author_cleaning": ["author_name_normalized", "author_potential_duplicate"],
            "description_cleaning": ["description_original", "description_cleaned"],
            "series_standardization": ["series_title_original", "series_title_cleaned"],
            "text_normalization": ["popular_shelves_original", "popular_shelves_cleaned"],
            "data_quality_improvement": ["data_quality_score"]
        }
        
        # Remove columns from steps after checkpoint
        for step_name in list(step_columns.keys())[checkpoint_index:]:
            columns_to_remove = step_columns[step_name]
            existing_columns = [col for col in columns_to_remove if col in df.columns]
            if existing_columns:
                df = df.drop(columns=existing_columns)
                self.logger.info(f"Removed columns: {existing_columns}")
        
        return df
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the dataset for cleaning.
        
        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading dataset from {self.input_path}")
        
        try:
            df = pd.read_csv(self.input_path)
            self.logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def clean_titles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean book titles by removing series information and standardizing numbering.
        Uses conservative approach to avoid removing legitimate content.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned titles and extracted series information
        """
        self.logger.info("Cleaning book titles with conservative approach...")
        
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
        
        self.logger.info(f"Title cleaning completed conservatively:")
        self.logger.info(f"  - Series numbers extracted: {series_numbers_found:,}")
        self.logger.info(f"  - Titles cleaned from series info: {titles_cleaned_from_series:,}")
        self.logger.info(f"  - Total titles modified: {(df['title_original'] != df['title_cleaned']).sum():,}")
        
        return df
    
    def clean_author_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize author names and identify potential duplicates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized author names and duplicate flags
        """
        self.logger.info("Cleaning author names...")
        
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
        self.logger.info(f"Identified {duplicate_count:,} books with potentially duplicate author names")
        
        return df
    
    def clean_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean book descriptions by removing HTML and normalizing text.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned descriptions
        """
        self.logger.info("Cleaning book descriptions...")
        
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
        
        self.logger.info(f"Description cleaning completed. Average length reduction: {length_reduction:.1f}%")
        return df
    
    def standardize_series_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize series information and create clean series titles.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized series information
        """
        self.logger.info("Standardizing series information...")
        
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
        
        self.logger.info(f"Series standardization completed:")
        self.logger.info(f"  - Books with position information: {df['series_position'].notna().sum():,}")
        self.logger.info(f"  - Series titles cleaned: {mask.sum():,}")
        
        return df
    
    def normalize_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize text in popular_shelves and other text columns.
        Focuses on text cleaning and normalization, not classification.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized text columns
        """
        self.logger.info("Normalizing text columns...")
        
        # Create backup columns
        df['popular_shelves_original'] = df['popular_shelves'].copy()
        df['popular_shelves_cleaned'] = df['popular_shelves'].copy()
        
        # Apply text cleaning to popular_shelves
        mask = df['popular_shelves'].notna()
        
        if mask.sum() > 0:
            # Clean text using patterns
            df.loc[mask, 'popular_shelves_cleaned'] = df.loc[mask, 'popular_shelves_cleaned'].apply(
                lambda x: self._clean_text_content(str(x))
            )
            
            # Calculate cleaning statistics
            original_lengths = df.loc[mask, 'popular_shelves_original'].str.len()
            cleaned_lengths = df.loc[mask, 'popular_shelves_cleaned'].str.len()
            length_reduction = ((original_lengths - cleaned_lengths) / original_lengths * 100).mean()
            
            self.logger.info(f"Text normalization completed. Average length reduction: {length_reduction:.1f}%")
        
        return df
    
    def _clean_text_content(self, text: str) -> str:
        """
        Clean text content using predefined patterns.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return text
        
        # Apply cleaning patterns
        cleaned_text = text
        
        # Remove multiple spaces
        cleaned_text = re.sub(self.text_cleaning_patterns['multiple_spaces'], ' ', cleaned_text)
        
        # Remove leading/trailing spaces
        cleaned_text = re.sub(self.text_cleaning_patterns['leading_trailing_spaces'], '', cleaned_text)
        
        # Clean multiple commas
        cleaned_text = re.sub(self.text_cleaning_patterns['multiple_commas'], ',', cleaned_text)
        
        # Remove empty entries
        cleaned_text = re.sub(self.text_cleaning_patterns['empty_entries'], '', cleaned_text)
        
        # Final cleanup
        cleaned_text = cleaned_text.strip()
        if cleaned_text.startswith(','):
            cleaned_text = cleaned_text[1:]
        if cleaned_text.endswith(','):
            cleaned_text = cleaned_text[:-1]
        
        return cleaned_text
    
    def improve_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Improve overall data quality by handling missing values and inconsistencies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with improved data quality
        """
        self.logger.info("Improving overall data quality...")
        
        # Handle missing median_publication_year
        missing_median_year = df['median_publication_year'].isnull().sum()
        if missing_median_year > 0:
            # Use publication_year as fallback
            df.loc[df['median_publication_year'].isnull(), 'median_publication_year'] = \
                df.loc[df['median_publication_year'].isnull(), 'publication_year']
            self.logger.info(f"Filled {missing_median_year:,} missing median_publication_year values")
        
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
        
        self.logger.info(f"Data quality improvement completed:")
        self.logger.info(f"  - Average quality score: {avg_quality:.3f}")
        self.logger.info(f"  - Quality distribution:")
        for score, count in quality_distribution.items():
            percentage = count/len(df)*100
            self.logger.info(f"    {score:.3f}: {count:,} records ({percentage:.1f}%)")
        
        return df
    
    def create_clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a clean dataset with original columns, applying all cleaning operations.
        
        Args:
            df: DataFrame with cleaning metadata columns
            
        Returns:
            Clean DataFrame with original column structure
        """
        self.logger.info("Creating clean dataset with original column structure...")
        
        # Start with original columns (excluding cleaning metadata)
        original_columns = [
            'work_id', 'book_id_list_en', 'title', 'publication_year', 
            'median_publication_year', 'language_codes_en', 'num_pages_median',
            'description', 'popular_shelves', 'author_id', 'author_name',
            'author_average_rating', 'author_ratings_count', 'series_id',
            'series_title', 'series_works_count', 'ratings_count_sum',
            'text_reviews_count_sum', 'average_rating_weighted_mean',
            'average_rating_weighted_median'
        ]
        
        # Filter to columns that exist in the input
        existing_columns = [col for col in original_columns if col in df.columns]
        
        # Create clean dataset
        clean_df = df[existing_columns].copy()
        
        # Apply cleaned values where available
        if 'title_cleaned' in df.columns:
            clean_df['title'] = df['title_cleaned']
        
        if 'description_cleaned' in df.columns:
            clean_df['description'] = df['description_cleaned']
        
        if 'series_title_cleaned' in df.columns:
            clean_df['series_title'] = df['series_title_cleaned']
        
        if 'popular_shelves_cleaned' in df.columns:
            clean_df['popular_shelves'] = df['popular_shelves_cleaned']
        
        if 'author_name_normalized' in df.columns:
            clean_df['author_name'] = df['author_name_normalized']
        
        # Fill missing median_publication_year with publication_year
        if 'median_publication_year' in clean_df.columns and 'publication_year' in clean_df.columns:
            clean_df['median_publication_year'] = clean_df['median_publication_year'].fillna(
                clean_df['publication_year']
            )
        
        self.logger.info(f"Clean dataset created with {len(clean_df.columns)} columns")
        return clean_df

    def run_cleaning_pipeline(self, output_filename: Optional[str] = None, 
                            create_separate_clean: bool = True) -> str:
        """
        Run the complete cleaning pipeline with error handling and rollback.
        
        Args:
            output_filename: Optional custom output filename
            create_separate_clean: If True, create separate clean dataset with original columns
            
        Returns:
            Path to the cleaned dataset
            
        Raises:
            RuntimeError: If cleaning pipeline fails and rollback is not possible
        """
        self.logger.info("Starting comprehensive data cleaning pipeline...")
        
        try:
            # Load dataset
            df = self.load_dataset()
            
            # Define cleaning steps
            cleaning_steps = [
                ("title_cleaning", self.clean_titles),
                ("author_cleaning", self.clean_author_names),
                ("description_cleaning", self.clean_descriptions),
                ("series_standardization", self.standardize_series_info),
                ("text_normalization", self.normalize_text_columns),
                ("data_quality_improvement", self.improve_data_quality)
            ]
            
            # Apply cleaning steps with error handling
            for step_name, step_func in cleaning_steps:
                try:
                    # Create checkpoint before each step
                    checkpoint = self._create_checkpoint(step_name, df)
                    
                    # Execute cleaning step with performance monitoring
                    df = self._monitor_performance(step_name, df, step_func)
                    
                    self.logger.info(f"✅ Step '{step_name}' completed successfully")
                    
                except Exception as e:
                    self.logger.error(f"❌ Step '{step_name}' failed: {e}")
                    
                    # Attempt rollback
                    if self.checkpoints:
                        last_checkpoint = self.checkpoints[-1]
                        df = self._rollback_to_checkpoint(last_checkpoint, df)
                        self.logger.info(f"Rolled back to checkpoint '{last_checkpoint.step_name}'")
                    else:
                        self.logger.error("No checkpoints available for rollback")
                        raise RuntimeError(f"Cleaning pipeline failed at step '{step_name}' and rollback not possible")
                    
                    # Re-raise the error
                    raise
            
            # Generate output filenames
            if output_filename is None:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"romance_novels_cleaned_{timestamp}.csv"
            
            # Save full dataset with cleaning metadata
            full_output_path = self.output_dir / output_filename
            self.logger.info(f"Saving full dataset with cleaning metadata to {full_output_path}")
            df.to_csv(full_output_path, index=False)
            
            # Create separate clean dataset if requested
            if create_separate_clean:
                clean_df = self.create_clean_dataset(df)
                clean_filename = f"romance_novels_clean_only_{timestamp}.csv"
                clean_output_path = self.output_dir / clean_filename
                self.logger.info(f"Saving clean dataset (original columns) to {clean_output_path}")
                clean_df.to_csv(clean_output_path, index=False)
                
                # Update output path to return the clean dataset path
                output_path = clean_output_path
            else:
                output_path = full_output_path
            
            # Generate reports
            self._generate_cleaning_report(df, full_output_path)
            self._generate_performance_report(full_output_path)
            
            self.logger.info("Data cleaning pipeline completed successfully!")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Cleaning pipeline failed: {e}")
            raise
    
    def _generate_cleaning_report(self, df: pd.DataFrame, output_path: Path) -> None:
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
            
            # Text normalization
            text_modified = (df['popular_shelves_original'] != df['popular_shelves_cleaned']).sum()
            f.write(f"Text columns normalized: {text_modified:,}\n")
            
            # Data quality
            avg_quality = df['data_quality_score'].mean()
            f.write(f"Average data quality score: {avg_quality:.3f}\n")
        
        self.logger.info(f"Cleaning report saved to {report_path}")
    
    def _generate_performance_report(self, output_path: Path) -> None:
        """Generate a performance report for the cleaning pipeline."""
        report_path = output_path.parent / f"{output_path.stem}_performance_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("CLEANING PIPELINE PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("PERFORMANCE METRICS BY STEP:\n")
            f.write("-" * 30 + "\n")
            
            total_time = 0
            total_memory_delta = 0
            
            for metrics in self.performance_metrics:
                f.write(f"\nStep: {metrics.step_name}\n")
                f.write(f"  Execution Time: {metrics.execution_time:.2f} seconds\n")
                f.write(f"  Memory Before: {metrics.memory_before:.1f} MB\n")
                f.write(f"  Memory After: {metrics.memory_after:.1f} MB\n")
                f.write(f"  Memory Delta: {metrics.memory_delta:+.1f} MB\n")
                f.write(f"  Records Processed: {metrics.records_processed:,}\n")
                f.write(f"  Success: {metrics.success}\n")
                
                if metrics.error_message:
                    f.write(f"  Error: {metrics.error_message}\n")
                
                total_time += metrics.execution_time
                total_memory_delta += metrics.memory_delta
            
            f.write(f"\nTOTAL PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Execution Time: {total_time:.2f} seconds\n")
            f.write(f"Total Memory Delta: {total_memory_delta:+.1f} MB\n")
            f.write(f"Average Time per Step: {total_time/len(self.performance_metrics):.2f} seconds\n")
            f.write(f"Steps Completed: {len([m for m in self.performance_metrics if m.success])}/{len(self.performance_metrics)}\n")
        
        self.logger.info(f"Performance report saved to {report_path}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.performance_metrics:
            return {}
        
        successful_steps = [m for m in self.performance_metrics if m.success]
        failed_steps = [m for m in self.performance_metrics if not m.success]
        
        total_time = sum(m.execution_time for m in self.performance_metrics)
        total_memory_delta = sum(m.memory_delta for m in self.performance_metrics)
        
        return {
            'total_steps': len(self.performance_metrics),
            'successful_steps': len(successful_steps),
            'failed_steps': len(failed_steps),
            'total_execution_time': total_time,
            'total_memory_delta': total_memory_delta,
            'average_time_per_step': total_time / len(self.performance_metrics),
            'success_rate': len(successful_steps) / len(self.performance_metrics) * 100
        }
