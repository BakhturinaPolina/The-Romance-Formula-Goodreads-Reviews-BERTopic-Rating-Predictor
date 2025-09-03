"""
Pipeline Runner for Data Processing
Orchestrates the entire data processing workflow.
"""

import yaml
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List
import pandas as pd

from .config_loader import ConfigLoader
from .data_loader import DataLoader
from .quality_filters import QualityFilters
from .data_integrator import DataIntegrator
from .pipeline_validator import PipelineValidator

class PipelineRunner:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = self._setup_logger()
        
        # Load configurations
        self.config_loader = ConfigLoader(config_dir)
        self.variable_selection = self.config_loader.get_variable_selection()
        self.sampling_policy = self.config_loader.get_sampling_policy()
        self.fields_required = self.config_loader.get_fields_required()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.quality_filters = QualityFilters(self.sampling_policy)
        self.data_integrator = DataIntegrator()
        self.validator = PipelineValidator()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for pipeline operations."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def run_pipeline(self, sample_size: int = None) -> None:
        """
        Run the complete data processing pipeline with type conversion.
        
        NEW STRUCTURE:
        1. Clean entire dataset with quality thresholds
        2. Save full cleaned datasets (all books that pass thresholds)
        3. Apply random sampling to create sampled versions
        
        Args:
            sample_size: If provided, create sampled versions with this many books. 
                        If None, only create full cleaned datasets.
        """
        if sample_size:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting data processing pipeline (full cleaning + {sample_size:,} book sample)...")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting data processing pipeline (full cleaning only)...")
        
        start_time = time.time()
        
        try:
            # PHASE 1: FULL DATASET CLEANING
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧹 PHASE 1: Cleaning entire dataset...")
            
            # Step 1: Load books data with type conversion
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Step 1: Loading books data...")
            books_data = self.data_loader.load_books_data(self.variable_selection)
            
            # Validate books data after loading
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating books data after loading...")
            self.validator.validate_books_data(books_data, "after_loading")
            
            # Step 2: Apply quality filters to ENTIRE dataset
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Step 2: Applying quality filters to entire dataset...")
            filtered_books = self.quality_filters.filter_books(books_data)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Quality filtering complete: {len(filtered_books):,} books passed thresholds")
            
            # Validate books data after filtering
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating books data after filtering...")
            self.validator.validate_books_data(filtered_books, "after_filtering")
            
            # Step 3: Integrate books data with edition aggregations
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 Step 3: Integrating books data...")
            books_df = self.data_integrator.integrate_books_data(filtered_books, self.variable_selection)
            
            # Validate books data after integration
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating books data after integration...")
            self.validator.validate_books_data(books_df.to_dict('records'), "after_integration")
            
            # Step 3.5: Integrate works data for better publication years
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Step 3.5: Integrating works data for better publication years...")
            books_with_works = self.data_integrator.integrate_works_data(
                books_df.to_dict('records'), self.variable_selection
            )
            books_df = pd.DataFrame(books_with_works)
            
            # Validate books data after works integration
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating books data after works integration...")
            self.validator.validate_books_data(books_df.to_dict('records'), "after_works_integration")
            
            # Step 3.6: Fix problematic publication years using median calculation
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Step 3.6: Fixing problematic publication years...")
            books_fixed = self.data_integrator.fix_problematic_publication_years(
                books_df.to_dict('records')
            )
            books_df = pd.DataFrame(books_fixed)
            
            # Validate books data after publication year fixing
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating books data after publication year fixing...")
            self.validator.validate_books_data(books_df.to_dict('records'), "after_publication_year_fixing")
            
            # Step 4: Integrate author data
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 👤 Step 4: Integrating author data...")
            books_with_authors = self.data_integrator.integrate_author_data(
                books_df.to_dict('records'), self.variable_selection
            )
            books_df = pd.DataFrame(books_with_authors)
            
            # Validate books data after author integration
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating books data after author integration...")
            self.validator.validate_books_data(books_df.to_dict('records'), "after_author_integration")
            
            # Step 4.5: Integrate series data
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📚 Step 4.5: Integrating series data...")
            books_with_series = self.data_integrator.integrate_series_data(
                books_df.to_dict('records'), self.variable_selection
            )
            books_df = pd.DataFrame(books_with_series)
            
            # Validate books data after series integration
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating books data after series integration...")
            self.validator.validate_books_data(books_df.to_dict('records'), "after_series_integration")
            
            # Step 5: Load ALL reviews for cleaned books
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Step 5: Loading ALL reviews for cleaned books...")
            reviews_data = self.data_loader.load_reviews_data(self.variable_selection)
            
            # Validate reviews data after loading
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating reviews data after loading...")
            self.validator.validate_reviews_data(reviews_data, "after_loading")
            
            # Get ALL reviews for cleaned books (no sampling yet)
            book_ids = books_df['book_id'].tolist()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 Filtering reviews to {len(book_ids):,} cleaned books...")
            all_reviews_df = self.data_integrator.filter_reviews_for_books(reviews_data, book_ids)
            
            # Validate reviews data after filtering
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating reviews data after filtering...")
            self.validator.validate_reviews_data(all_reviews_df.to_dict('records'), "after_filtering")
            
            # Step 5.5: Integrate reviews data for aggregated metrics
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Step 5.5: Integrating reviews data for aggregated metrics...")
            books_with_reviews = self.data_integrator.integrate_reviews_data(
                books_df.to_dict('records'), all_reviews_df.to_dict('records')
            )
            books_df = pd.DataFrame(books_with_reviews)
            
            # Validate books data after reviews integration
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating books data after reviews integration...")
            self.validator.validate_books_data(books_df.to_dict('records'), "after_reviews_integration")
            
            # Step 6: Validate cross-dataset consistency
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Step 6: Validating cross-dataset consistency...")
            self.validator.validate_cross_dataset_consistency(
                books_df.to_dict('records'), 
                all_reviews_df.to_dict('records'),
                "after_full_cleaning"
            )
            
            # Step 7: Create subgenre classification placeholder
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏷️ Step 7: Creating subgenre classification placeholder...")
            subgenre_df = self._create_subgenre_placeholder(books_df)
            
            # Step 8: Save FULL cleaned datasets
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Step 8: Saving FULL cleaned datasets...")
            self.data_integrator.save_full_datasets(books_df, all_reviews_df, subgenre_df)
            
            # PHASE 2: SAMPLING (if requested)
            if sample_size and len(books_df) > sample_size:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 PHASE 2: Creating sampled datasets...")
                
                # Step 9: Sample books from full cleaned dataset
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎲 Step 9: Sampling {sample_size:,} books from {len(books_df):,} cleaned books...")
                import numpy as np
                np.random.seed(42)  # For reproducibility
                sample_indices = np.random.choice(len(books_df), size=sample_size, replace=False)
                sampled_books_df = books_df.iloc[sample_indices].copy()
                
                # Step 10: Get reviews for sampled books
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Step 10: Getting reviews for sampled books...")
                sampled_book_ids = sampled_books_df['book_id'].tolist()
                sampled_reviews_df = self.data_integrator.filter_reviews_for_books(reviews_data, sampled_book_ids)
                
                # Step 11: Apply review sampling to sampled books
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎲 Step 11: Sampling reviews for sampled books...")
                sampled_reviews_df = self.data_integrator.sample_reviews_data(
                    sampled_reviews_df.to_dict('records'), 
                    self.sampling_policy, 
                    sampled_book_ids
                )
                
                # Step 12: Create subgenre classification for sampled books
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏷️ Step 12: Creating subgenre classification for sampled books...")
                sampled_subgenre_df = self._create_subgenre_placeholder(sampled_books_df)
                
                # Step 13: Save sampled datasets
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Step 13: Saving sampled datasets...")
                self.data_integrator.save_sampled_datasets(sampled_books_df, sampled_reviews_df, sampled_subgenre_df, sample_size)
                
                # Validate sampled datasets
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Validating sampled datasets...")
                self.validator.validate_cross_dataset_consistency(
                    sampled_books_df.to_dict('records'), 
                    sampled_reviews_df.to_dict('records'),
                    "after_sampling"
                )
                
                final_books_df = sampled_books_df
                final_reviews_df = sampled_reviews_df
                final_subgenre_df = sampled_subgenre_df
                
            else:
                final_books_df = books_df
                final_reviews_df = all_reviews_df
                final_subgenre_df = subgenre_df
                
                if sample_size and len(books_df) <= sample_size:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Only {len(books_df):,} books available after cleaning (requested {sample_size:,})")
            
            # Step 14: Generate validation summary
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Step 14: Generating validation summary...")
            self.validator.print_validation_summary()
            
            # Save validation report
            validation_report_path = Path("logs/validation") / f"pipeline_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.validator.save_validation_report(validation_report_path)
            
            # Print final summary
            elapsed_time = time.time() - start_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Pipeline completed successfully!")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 Final dataset sizes:")
            print(f"  - Books: {len(final_books_df):,} records")
            print(f"  - Reviews: {len(final_reviews_df):,} records")
            print(f"  - Subgenre classification: {len(final_subgenre_df):,} records")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏱️ Total processing time: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Pipeline failed with error: {e}")
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _create_subgenre_placeholder(self, books_df) -> pd.DataFrame:
        """
        Create a placeholder subgenre classification DataFrame.
        
        Args:
            books_df: Books DataFrame
            
        Returns:
            Subgenre classification DataFrame
        """
        
        subgenre_data = []
        for _, book in books_df.iterrows():
            subgenre_record = {
                'book_id': book.get('book_id'),
                'title': book.get('title', ''),
                'subgenre_primary': '',  # To be filled by classifier
                'subgenre_keywords': '',  # To be filled by classifier
                'subgenre_final': '',     # To be filled by classifier
                'confidence_score': 0.0,  # To be filled by classifier
                'primary_keywords_matched': '',  # To be filled by classifier
                'keyword_keywords_matched': '',  # To be filled by classifier
                'validation_notes': ''    # To be filled by classifier
            }
            subgenre_data.append(subgenre_record)
        
        return pd.DataFrame(subgenre_data)

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline configuration and status.
        
        Returns:
            Dictionary containing pipeline summary
        """
        return {
            "pipeline_config": {
                "variable_selection": self.variable_selection,
                "sampling_policy": self.sampling_policy,
                "fields_required": self.fields_required
            },
            "components": {
                "data_loader": "DataLoader with type conversion",
                "quality_filters": "QualityFilters with author balancing",
                "data_integrator": "DataIntegrator with edition aggregations",
                "pipeline_validator": "PipelineValidator with comprehensive checks"
            },
            "data_types": {
                "converted_fields": [
                    "text_reviews_count (str -> int)",
                    "ratings_count (str -> int)", 
                    "num_pages (str -> int)",
                    "publication_day (str -> int)",
                    "publication_month (str -> int)",
                    "publication_year (str -> int)",
                    "book_id (str -> int)",
                    "work_id (str -> int)",
                    "average_rating (str -> float)",
                    "is_ebook (str -> bool)",
                    "review_id (str -> int)",
                    "user_id (str -> int)",
                    "rating (str -> int)"
                ]
            }
        }
