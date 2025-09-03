"""
Data Quality Assessment for Romance Novel NLP Research
Step 1: Data Quality Assessment and Missing Values Handling

This module implements comprehensive data quality assessment for the final dataset,
focusing on missing values analysis and data type validation as outlined in the EDA plan.

Author: AI Assistant
Date: September 2025
Version: 1.0.0
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


class DataQualityAssessment:
    """
    Comprehensive data quality assessment for romance novel dataset.
    
    Implements Step 1 of the EDA plan:
    - Missing values analysis for key fields
    - Data type validation for critical variables
    - Quality metrics reporting
    - NLP analysis readiness assessment
    """
    
    def __init__(self, data_path: str = "data/processed"):
        """
        Initialize the data quality assessment.
        
        Args:
            data_path: Path to the processed data directory
        """
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
        self.quality_report = {}
        self.validation_errors = []
        self.nlp_ready_flags = {}
        
        # Define critical fields for analysis
        self.critical_fields = [
            'work_id', 'title', 'publication_year', 'description',
            'author_id', 'author_name', 'genres'
        ]
        
        # Define fields for missing values analysis
        self.missing_analysis_fields = [
            'num_pages_median', 'description', 'series_id', 'genres'
        ]
        
        # Define fields for data type validation
        self.type_validation_fields = {
            'publication_year': {'type': 'int', 'range': (2000, 2020)},
            'ratings_count_sum': {'type': 'int', 'min': 0},
            'text_reviews_count_sum': {'type': 'int', 'min': 0},
            'average_rating_weighted_mean': {'type': 'float', 'range': (0, 5)}
        }
        
        logger.info("DataQualityAssessment initialized")
    
    def load_dataset(self, filename: str = None) -> bool:
        """
        Load the dataset for analysis.
        
        Args:
            filename: Specific CSV file to load, or None for latest
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if filename:
                file_path = self.data_path / filename
            else:
                # Find the latest processed dataset
                csv_files = list(self.data_path.glob("final_books_*.csv"))
                if not csv_files:
                    logger.error("No processed CSV files found")
                    return False
                
                # Sort by modification time and get the latest
                csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                file_path = csv_files[0]
            
            logger.info(f"Loading dataset: {file_path}")
            
            # Load with appropriate data types
            self.data = pd.read_csv(file_path)
            
            logger.info(f"Dataset loaded successfully: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values for key fields.
        
        Returns:
            Dict containing missing values analysis results
        """
        if self.data is None:
            logger.error("No dataset loaded")
            return {}
        
        logger.info("Analyzing missing values...")
        
        missing_analysis = {}
        
        for field in self.missing_analysis_fields:
            if field in self.data.columns:
                missing_count = self.data[field].isna().sum()
                missing_percentage = (missing_count / len(self.data)) * 100
                total_count = len(self.data)
                
                missing_analysis[field] = {
                    'missing_count': missing_count,
                    'missing_percentage': missing_percentage,
                    'total_count': total_count,
                    'completeness': 100 - missing_percentage
                }
                
                logger.info(f"{field}: {missing_count:,} missing ({missing_percentage:.1f}%)")
                
                # Flag missing descriptions for NLP analysis exclusion
                if field == 'description':
                    self.nlp_ready_flags['descriptions_complete'] = missing_count == 0
                    self.nlp_ready_flags['descriptions_missing'] = missing_count
                    if missing_count > 0:
                        logger.warning(f"Missing descriptions will be excluded from NLP analysis")
        
        # Handle missing series data (67% coverage expected)
        if 'series_id' in self.data.columns:
            series_coverage = missing_analysis.get('series_id', {}).get('completeness', 0)
            expected_coverage = 67.0
            coverage_status = "above" if series_coverage > expected_coverage else "below"
            logger.info(f"Series coverage: {series_coverage:.1f}% ({coverage_status} expected {expected_coverage}%)")
        
        self.quality_report['missing_values'] = missing_analysis
        return missing_analysis
    
    def validate_data_types(self) -> Dict[str, Any]:
        """
        Validate data types for critical variables.
        
        Returns:
            Dict containing validation results
        """
        if self.data is None:
            logger.error("No dataset loaded")
            return {}
        
        logger.info("Validating data types...")
        
        validation_results = {}
        
        for field, validation_rules in self.type_validation_fields.items():
            if field not in self.data.columns:
                logger.warning(f"Field {field} not found in dataset")
                continue
            
            field_validation = {
                'field': field,
                'expected_type': validation_rules['type'],
                'actual_type': str(self.data[field].dtype),
                'validation_passed': True,
                'errors': []
            }
            
            # Type validation
            if validation_rules['type'] == 'int':
                if not pd.api.types.is_integer_dtype(self.data[field]):
                    field_validation['validation_passed'] = False
                    field_validation['errors'].append("Field is not integer type")
                    
                    # Check for non-integer values
                    non_int_mask = ~pd.to_numeric(self.data[field], errors='coerce').notna()
                    non_int_count = non_int_mask.sum()
                    if non_int_count > 0:
                        field_validation['errors'].append(f"{non_int_count} non-integer values found")
            
            elif validation_rules['type'] == 'float':
                if not pd.api.types.is_float_dtype(self.data[field]):
                    field_validation['validation_passed'] = False
                    field_validation['errors'].append("Field is not float type")
            
            # Range validation
            if 'range' in validation_rules:
                min_val, max_val = validation_rules['range']
                out_of_range = self.data[field][
                    (self.data[field] < min_val) | (self.data[field] > max_val)
                ]
                out_of_range_count = len(out_of_range)
                
                if out_of_range_count > 0:
                    field_validation['validation_passed'] = False
                    field_validation['errors'].append(
                        f"{out_of_range_count} values outside range [{min_val}, {max_val}]"
                    )
                    
                    # Log specific out-of-range values for investigation
                    if out_of_range_count <= 10:  # Only log if manageable number
                        logger.warning(f"Out-of-range values in {field}: {out_of_range.tolist()}")
            
            # Minimum value validation
            if 'min' in validation_rules:
                min_val = validation_rules['min']
                below_min = self.data[field][self.data[field] < min_val]
                below_min_count = len(below_min)
                
                if below_min_count > 0:
                    field_validation['validation_passed'] = False
                    field_validation['errors'].append(
                        f"{below_min_count} values below minimum {min_val}"
                    )
            
            validation_results[field] = field_validation
            
            if field_validation['validation_passed']:
                logger.info(f"✓ {field}: Validation passed")
            else:
                logger.warning(f"✗ {field}: Validation failed - {field_validation['errors']}")
                self.validation_errors.append(field_validation)
        
        self.quality_report['data_type_validation'] = validation_results
        return validation_results
    
    def assess_nlp_readiness(self) -> Dict[str, Any]:
        """
        Assess dataset readiness for NLP analysis.
        
        Returns:
            Dict containing NLP readiness assessment
        """
        if self.data is None:
            logger.error("No dataset loaded")
            return {}
        
        logger.info("Assessing NLP analysis readiness...")
        
        nlp_assessment = {
            'total_books': len(self.data),
            'ready_for_nlp': 0,
            'excluded_from_nlp': 0,
            'exclusion_reasons': {}
        }
        
        # Check description quality for NLP
        if 'description' in self.data.columns:
            # Books with valid descriptions
            valid_descriptions = self.data['description'].notna() & (self.data['description'] != '')
            valid_description_count = valid_descriptions.sum()
            
            # Check description length
            if valid_description_count > 0:
                description_lengths = self.data.loc[valid_descriptions, 'description'].str.len()
                min_length = description_lengths.min()
                max_length = description_lengths.max()
                median_length = description_lengths.median()
                
                nlp_assessment['description_stats'] = {
                    'valid_count': valid_description_count,
                    'min_length': min_length,
                    'max_length': max_length,
                    'median_length': median_length
                }
                
                # Flag very short descriptions (potential NLP issues)
                very_short = description_lengths < 50
                very_short_count = very_short.sum()
                if very_short_count > 0:
                    nlp_assessment['exclusion_reasons']['very_short_descriptions'] = very_short_count
                    logger.warning(f"{very_short_count} descriptions are very short (<50 chars)")
            
            nlp_assessment['ready_for_nlp'] = valid_description_count
            nlp_assessment['excluded_from_nlp'] = len(self.data) - valid_description_count
        
        # Check genre coverage for subgenre analysis
        if 'genres' in self.data.columns:
            valid_genres = self.data['genres'].notna() & (self.data['genres'] != '')
            valid_genre_count = valid_genres.sum()
            nlp_assessment['genre_coverage'] = {
                'valid_count': valid_genre_count,
                'coverage_percentage': (valid_genre_count / len(self.data)) * 100
            }
        
        self.quality_report['nlp_readiness'] = nlp_assessment
        return nlp_assessment
    
    def generate_quality_summary(self) -> str:
        """
        Generate a comprehensive quality summary report.
        
        Returns:
            str: Formatted quality summary
        """
        if not self.quality_report:
            return "No quality analysis performed yet"
        
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("ROMANCE NOVEL DATASET QUALITY ASSESSMENT REPORT")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Dataset overview
        if self.data is not None:
            summary_lines.append("DATASET OVERVIEW:")
            summary_lines.append(f"  Total records: {len(self.data):,}")
            summary_lines.append(f"  Total columns: {len(self.data.columns)}")
            summary_lines.append("")
        
        # Missing values summary
        if 'missing_values' in self.quality_report:
            summary_lines.append("MISSING VALUES ANALYSIS:")
            for field, stats in self.quality_report['missing_values'].items():
                summary_lines.append(f"  {field}:")
                summary_lines.append(f"    Missing: {stats['missing_count']:,} ({stats['missing_percentage']:.1f}%)")
                summary_lines.append(f"    Completeness: {stats['completeness']:.1f}%")
            summary_lines.append("")
        
        # Data type validation summary
        if 'data_type_validation' in self.quality_report:
            summary_lines.append("DATA TYPE VALIDATION:")
            passed_count = 0
            total_count = len(self.quality_report['data_type_validation'])
            
            for field, validation in self.quality_report['data_type_validation'].items():
                status = "✓ PASS" if validation['validation_passed'] else "✗ FAIL"
                summary_lines.append(f"  {field}: {status}")
                if not validation['validation_passed']:
                    for error in validation['errors']:
                        summary_lines.append(f"    Error: {error}")
                else:
                    passed_count += 1
            
            summary_lines.append(f"  Overall: {passed_count}/{total_count} fields passed validation")
            summary_lines.append("")
        
        # NLP readiness summary
        if 'nlp_readiness' in self.quality_report:
            nlp_stats = self.quality_report['nlp_readiness']
            summary_lines.append("NLP ANALYSIS READINESS:")
            summary_lines.append(f"  Books ready for NLP: {nlp_stats['ready_for_nlp']:,}")
            summary_lines.append(f"  Books excluded from NLP: {nlp_stats['excluded_from_nlp']:,}")
            
            if 'description_stats' in nlp_stats:
                desc_stats = nlp_stats['description_stats']
                summary_lines.append(f"  Description length range: {desc_stats['min_length']} - {desc_stats['max_length']} chars")
                summary_lines.append(f"  Median description length: {desc_stats['median_length']:.0f} chars")
            
            if 'genre_coverage' in nlp_stats:
                genre_stats = nlp_stats['genre_coverage']
                summary_lines.append(f"  Genre coverage: {genre_stats['coverage_percentage']:.1f}%")
            summary_lines.append("")
        
        # Validation errors summary
        if self.validation_errors:
            summary_lines.append("VALIDATION ERRORS SUMMARY:")
            summary_lines.append(f"  Total validation errors: {len(self.validation_errors)}")
            for error in self.validation_errors:
                summary_lines.append(f"  - {error['field']}: {', '.join(error['errors'])}")
            summary_lines.append("")
        
        # Recommendations
        summary_lines.append("RECOMMENDATIONS:")
        if 'missing_values' in self.quality_report:
            missing_stats = self.quality_report['missing_values']
            if missing_stats.get('description', {}).get('missing_count', 0) > 0:
                summary_lines.append("  - Missing descriptions should be excluded from NLP analysis")
            if missing_stats.get('series_id', {}).get('missing_count', 0) > 0:
                summary_lines.append("  - Series data coverage is below expected 67%")
        
        if self.validation_errors:
            summary_lines.append("  - Data type validation errors should be addressed before analysis")
        
        summary_lines.append("  - Dataset is ready for Step 2: Duplicate and Inconsistency Detection")
        
        return "\n".join(summary_lines)
    
    def save_quality_report(self, output_path: str = None) -> str:
        """
        Save the quality report to file.
        
        Args:
            output_path: Path to save the report, or None for default
            
        Returns:
            str: Path to saved report
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"data/processed/data_quality_report_{timestamp}.txt"
        
        # Convert to absolute path relative to project root
        if Path(output_path).is_absolute():
            output_file = Path(output_path)
        else:
            # Find project root (look for src directory)
            current_dir = Path.cwd()
            while current_dir.parent != current_dir:
                if (current_dir / "src").exists():
                    break
                current_dir = current_dir.parent
            
            output_file = current_dir / output_path
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                f.write(self.generate_quality_summary())
            
            logger.info(f"Quality report saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")
            return ""
    
    def run_full_assessment(self, dataset_filename: str = None) -> Dict[str, Any]:
        """
        Run the complete data quality assessment.
        
        Args:
            dataset_filename: Specific dataset file to analyze
            
        Returns:
            Dict: Complete quality assessment results
        """
        logger.info("Starting comprehensive data quality assessment...")
        
        # Load dataset
        if not self.load_dataset(dataset_filename):
            return {}
        
        # Run all assessments
        missing_analysis = self.analyze_missing_values()
        type_validation = self.validate_data_types()
        nlp_readiness = self.assess_nlp_readiness()
        
        # Generate summary
        summary = self.generate_quality_summary()
        print(summary)
        
        # Save report
        report_path = self.save_quality_report()
        
        logger.info("Data quality assessment completed")
        
        return {
            'missing_values': missing_analysis,
            'data_type_validation': type_validation,
            'nlp_readiness': nlp_readiness,
            'summary': summary,
            'report_path': report_path
        }


def main():
    """Main function to run data quality assessment."""
    print("🔍 Romance Novel Dataset - Data Quality Assessment")
    print("=" * 60)
    
    # Initialize assessment
    assessor = DataQualityAssessment()
    
    # Run full assessment
    results = assessor.run_full_assessment()
    
    if results:
        print(f"\n✅ Assessment completed successfully!")
        print(f"📋 Report saved to: {results.get('report_path', 'Unknown')}")
    else:
        print("\n❌ Assessment failed!")
    
    return results


if __name__ == "__main__":
    main()
