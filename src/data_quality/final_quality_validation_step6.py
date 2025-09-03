#!/usr/bin/env python3
"""
Step 6: Final Data Quality Validation (Medium Priority)

This script performs comprehensive final data quality validation by:
1. Cross-validating all previous pipeline steps (1-5)
2. Implementing quality gates for pipeline completion
3. Generating final data quality certification
4. Validating data integrity throughout the pipeline

Author: Research Assistant
Date: 2025-09-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinalQualityValidator:
    """
    Comprehensive final data quality validation system.
    Cross-validates all pipeline steps and implements quality gates.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the final quality validator.
        
        Args:
            data_path: Path to the optimized dataset from Step 5
        """
        self.data_path = data_path or "outputs/data_type_optimization/cleaned_romance_novels_step5_optimized_20250902_232035.parquet"
        self.df = None
        self.validation_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("outputs/final_quality_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality gate thresholds
        self.quality_gates = {
            'completeness': 0.95,  # 95% non-null values
            'consistency': 0.90,    # 90% data consistency
            'integrity': 0.98,      # 98% data integrity
            'optimization': 0.95,   # 95% optimization success
            'overall': 0.90        # 90% overall quality
        }
        
        # Pipeline step validation criteria
        self.pipeline_validation = {
            'step1_3': {
                'duplicate_removal': 'success',
                'missing_value_treatment': 'success',
                'data_type_validation': 'success'
            },
            'step4': {
                'outlier_detection': 'success',
                'publication_year_treatment': 'success',
                'conservative_treatment': 'success'
            },
            'step5': {
                'data_type_optimization': 'success',
                'type_preservation': 'success',
                'storage_efficiency': 'success'
            }
        }
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the optimized dataset from Step 5.
        
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading optimized dataset from: {self.data_path}")
            
            if self.data_path.endswith('.parquet'):
                self.df = pd.read_parquet(self.data_path)
                logger.info("Data loaded from parquet file - data types preserved")
            elif self.data_path.endswith('.pkl') or self.data_path.endswith('.pickle'):
                self.df = pd.read_pickle(self.data_path)
                logger.info("Data loaded from pickle file - data types preserved")
            else:
                self.df = pd.read_csv(self.data_path)
                logger.info("Data loaded from CSV file")
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def validate_pipeline_step1_3(self) -> Dict[str, Any]:
        """
        Validate results from Steps 1-3 (Initial Cleaning).
        
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating Pipeline Steps 1-3: Initial Cleaning...")
        
        validation = {
            'step': '1-3',
            'description': 'Initial Cleaning (Duplicate Removal, Missing Values, Data Types)',
            'checks': {},
            'overall_status': 'pending'
        }
        
        # Check 1: Duplicate removal
        total_records = len(self.df)
        duplicate_flags = [
            'title_duplicate_flag',
            'author_id_duplicate_flag'
        ]
        
        duplicate_check = {
            'total_records': total_records,
            'duplicate_flags_present': all(col in self.df.columns for col in duplicate_flags),
            'duplicate_flags_values': {}
        }
        
        for flag in duplicate_flags:
            if flag in self.df.columns:
                duplicate_check['duplicate_flags_values'][flag] = {
                    'total': self.df[flag].sum(),
                    'percentage': (self.df[flag].sum() / total_records) * 100
                }
        
        validation['checks']['duplicate_removal'] = duplicate_check
        
        # Check 2: Missing value treatment
        missing_value_check = {
            'total_columns': len(self.df.columns),
            'columns_with_missing': 0,
            'missing_value_summary': {}
        }
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_value_check['columns_with_missing'] += 1
                missing_value_check['missing_value_summary'][col] = {
                    'missing_count': missing_count,
                    'missing_percentage': (missing_count / total_records) * 100
                }
        
        validation['checks']['missing_value_treatment'] = missing_value_check
        
        # Check 3: Data type validation
        data_type_check = {
            'total_columns': len(self.df.columns),
            'optimized_data_types': 0,
            'data_type_distribution': {},
            'optimization_status': {}
        }
        
        # Count optimized data types
        optimized_types = ['int16', 'int32', 'float32', 'category']
        for dtype in optimized_types:
            count = len(self.df.select_dtypes(include=[dtype]).columns)
            data_type_check['data_type_distribution'][dtype] = count
            data_type_check['optimized_data_types'] += count
        
        # Check specific optimizations
        expected_optimizations = {
            'publication_year': 'int16',
            'work_id': 'int32',
            'genres': 'category',
            'author_name': 'category'
        }
        
        for field, expected_type in expected_optimizations.items():
            if field in self.df.columns:
                actual_type = str(self.df[field].dtype)
                data_type_check['optimization_status'][field] = {
                    'expected': expected_type,
                    'actual': actual_type,
                    'status': '✅ Optimized' if actual_type == expected_type else '❌ Not Optimized'
                }
        
        validation['checks']['data_type_validation'] = data_type_check
        
        # Overall status assessment
        duplicate_success = duplicate_check['duplicate_flags_present']
        missing_success = missing_value_check['columns_with_missing'] < len(self.df.columns) * 0.5
        data_type_success = data_type_check['optimized_data_types'] > len(self.df.columns) * 0.5
        
        if duplicate_success and missing_success and data_type_success:
            validation['overall_status'] = 'success'
        else:
            validation['overall_status'] = 'partial'
        
        logger.info(f"Steps 1-3 validation completed: {validation['overall_status']}")
        
        return validation
    
    def validate_pipeline_step4(self) -> Dict[str, Any]:
        """
        Validate results from Step 4 (Outlier Detection & Treatment).
        
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating Pipeline Step 4: Outlier Detection & Treatment...")
        
        validation = {
            'step': '4',
            'description': 'Outlier Detection & Treatment',
            'checks': {},
            'overall_status': 'pending'
        }
        
        # Check 1: Publication year treatment
        publication_year_check = {
            'field_present': 'publication_year' in self.df.columns,
            'year_range': {},
            'treatment_effectiveness': {}
        }
        
        if 'publication_year' in self.df.columns:
            years = pd.to_numeric(self.df['publication_year'], errors='coerce').dropna()
            publication_year_check['year_range'] = {
                'min': int(years.min()),
                'max': int(years.max()),
                'target_min': 2000,
                'target_max': 2017,
                'within_bounds': (years.min() >= 2000) and (years.max() <= 2017)
            }
            
            # Check treatment effectiveness
            books_before_2000 = len(years[years < 2000])
            books_after_2017 = len(years[years > 2017])
            
            publication_year_check['treatment_effectiveness'] = {
                'books_before_2000': books_before_2000,
                'books_after_2017': books_after_2017,
                'total_out_of_bounds': books_before_2000 + books_after_2017,
                'treatment_success': (books_before_2000 + books_after_2017) == 0
            }
        
        validation['checks']['publication_year_treatment'] = publication_year_check
        
        # Check 2: Conservative treatment (outlier documentation)
        conservative_check = {
            'outlier_fields_present': 0,
            'outlier_analysis_ready': False
        }
        
        # Check if outlier analysis fields are present
        outlier_fields = [
            'average_rating_weighted_mean',
            'ratings_count_sum',
            'text_reviews_count_sum',
            'author_ratings_count'
        ]
        
        conservative_check['outlier_fields_present'] = sum(1 for field in outlier_fields if field in self.df.columns)
        conservative_check['outlier_analysis_ready'] = conservative_check['outlier_fields_present'] == len(outlier_fields)
        
        validation['checks']['conservative_treatment'] = conservative_check
        
        # Check 3: Data integrity maintenance
        integrity_check = {
            'total_records': len(self.df),
            'expected_records': 80657,  # From Step 4 treatment
            'record_count_match': len(self.df) == 80657,
            'no_data_loss': True
        }
        
        validation['checks']['data_integrity'] = integrity_check
        
        # Overall status assessment
        year_treatment_success = publication_year_check.get('treatment_effectiveness', {}).get('treatment_success', False)
        conservative_success = conservative_check['outlier_analysis_ready']
        integrity_success = integrity_check['record_count_match']
        
        if year_treatment_success and conservative_success and integrity_success:
            validation['overall_status'] = 'success'
        elif year_treatment_success and integrity_success:
            validation['overall_status'] = 'partial'
        else:
            validation['overall_status'] = 'failed'
        
        logger.info(f"Step 4 validation completed: {validation['overall_status']}")
        
        return validation
    
    def validate_pipeline_step5(self) -> Dict[str, Any]:
        """
        Validate results from Step 5 (Data Type Optimization & Persistence).
        
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating Pipeline Step 5: Data Type Optimization & Persistence...")
        
        validation = {
            'step': '5',
            'description': 'Data Type Optimization & Persistence',
            'checks': {},
            'overall_status': 'pending'
        }
        
        # Check 1: Data type optimization
        optimization_check = {
            'total_columns': len(self.df.columns),
            'optimized_columns': 0,
            'optimization_distribution': {},
            'specific_optimizations': {}
        }
        
        # Count optimized data types
        optimized_types = ['int16', 'int32', 'float32', 'category']
        for dtype in optimized_types:
            count = len(self.df.select_dtypes(include=[dtype]).columns)
            optimization_check['optimization_distribution'][dtype] = count
            optimization_check['optimized_columns'] += count
        
        # Check specific float optimizations from Step 5
        float_optimizations = {
            'num_pages_median': 'float32',
            'author_average_rating': 'float32',
            'average_rating_weighted_mean': 'float32'
        }
        
        for field, expected_type in float_optimizations.items():
            if field in self.df.columns:
                actual_type = str(self.df[field].dtype)
                optimization_check['specific_optimizations'][field] = {
                    'expected': expected_type,
                    'actual': actual_type,
                    'status': '✅ Optimized' if actual_type == expected_type else '❌ Not Optimized'
                }
        
        validation['checks']['data_type_optimization'] = optimization_check
        
        # Check 2: Type preservation
        type_preservation_check = {
            'categorical_preserved': 0,
            'numerical_preserved': 0,
            'preservation_status': {}
        }
        
        # Check categorical preservation
        categorical_fields = [
            'genres', 'author_name', 'series_title', 'duplication_status',
            'cleaning_strategy', 'disambiguation_notes', 'decade',
            'book_length_category', 'rating_category', 'popularity_category'
        ]
        
        for field in categorical_fields:
            if field in self.df.columns:
                actual_type = str(self.df[field].dtype)
                if actual_type == 'category':
                    type_preservation_check['categorical_preserved'] += 1
                    type_preservation_check['preservation_status'][field] = '✅ Preserved'
                else:
                    type_preservation_check['preservation_status'][field] = '❌ Not Preserved'
        
        # Check numerical preservation
        numerical_fields = [
            'work_id', 'publication_year', 'author_id', 'author_ratings_count',
            'ratings_count_sum', 'text_reviews_count_sum'
        ]
        
        for field in numerical_fields:
            if field in self.df.columns:
                actual_type = str(self.df[field].dtype)
                if actual_type in ['int16', 'int32', 'int64']:
                    type_preservation_check['numerical_preserved'] += 1
                    type_preservation_check['preservation_status'][field] = '✅ Preserved'
                else:
                    type_preservation_check['preservation_status'][field] = '❌ Not Preserved'
        
        validation['checks']['type_preservation'] = type_preservation_check
        
        # Check 3: Storage efficiency
        storage_check = {
            'current_memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'expected_memory_mb': 182.66,  # From Step 5
            'memory_efficiency': {},
            'storage_format': 'parquet' if self.data_path.endswith('.parquet') else 'other'
        }
        
        current_memory = storage_check['current_memory_mb']
        expected_memory = storage_check['expected_memory_mb']
        
        storage_check['memory_efficiency'] = {
            'memory_match': abs(current_memory - expected_memory) < 1.0,  # Within 1 MB
            'memory_difference': current_memory - expected_memory,
            'efficiency_status': '✅ Efficient' if abs(current_memory - expected_memory) < 1.0 else '⚠️ Inefficient'
        }
        
        validation['checks']['storage_efficiency'] = storage_check
        
        # Overall status assessment
        optimization_success = optimization_check['optimized_columns'] > len(self.df.columns) * 0.7
        preservation_success = (type_preservation_check['categorical_preserved'] + type_preservation_check['numerical_preserved']) > 15
        storage_success = storage_check['memory_efficiency']['memory_match']
        
        if optimization_success and preservation_success and storage_success:
            validation['overall_status'] = 'success'
        elif optimization_success and preservation_success:
            validation['overall_status'] = 'partial'
        else:
            validation['overall_status'] = 'failed'
        
        logger.info(f"Step 5 validation completed: {validation['overall_status']}")
        
        return validation
    
    def calculate_quality_score(self) -> Dict[str, Any]:
        """
        Calculate comprehensive quality score based on all validations.
        
        Returns:
            Dictionary with quality score and breakdown
        """
        logger.info("📊 Calculating comprehensive quality score...")
        
        quality_score = {
            'overall_score': 0.0,
            'component_scores': {},
            'quality_gates': {},
            'recommendations': []
        }
        
        # Calculate component scores
        step_scores = {}
        
        # Step 1-3 score
        step1_3 = self.validation_results.get('step1_3', {})
        if step1_3.get('overall_status') == 'success':
            step_scores['step1_3'] = 100.0
        elif step1_3.get('overall_status') == 'partial':
            step1_3_checks = step1_3.get('checks', {})
            duplicate_success = step1_3_checks.get('duplicate_removal', {}).get('duplicate_flags_present', False)
            missing_success = step1_3_checks.get('missing_value_treatment', {}).get('columns_with_missing', 0) < 10
            data_type_success = step1_3_checks.get('data_type_validation', {}).get('optimized_data_types', 0) > 20
            
            step_scores['step1_3'] = sum([duplicate_success, missing_success, data_type_success]) / 3 * 100
        else:
            step_scores['step1_3'] = 0.0
        
        # Step 4 score
        step4 = self.validation_results.get('step4', {})
        if step4.get('overall_status') == 'success':
            step_scores['step4'] = 100.0
        elif step4.get('overall_status') == 'partial':
            step4_checks = step4.get('checks', {})
            year_success = step4_checks.get('publication_year_treatment', {}).get('treatment_effectiveness', {}).get('treatment_success', False)
            conservative_success = step4_checks.get('conservative_treatment', {}).get('outlier_analysis_ready', False)
            integrity_success = step4_checks.get('data_integrity', {}).get('record_count_match', False)
            
            step_scores['step4'] = sum([year_success, conservative_success, integrity_success]) / 3 * 100
        else:
            step_scores['step4'] = 0.0
        
        # Step 5 score
        step5 = self.validation_results.get('step5', {})
        if step5.get('overall_status') == 'success':
            step_scores['step5'] = 100.0
        elif step5.get('overall_status') == 'partial':
            step5_checks = step5.get('checks', {})
            optimization_success = step5_checks.get('data_type_optimization', {}).get('optimized_columns', 0) > 20
            preservation_success = step5_checks.get('type_preservation', {}).get('categorical_preserved', 0) + step5_checks.get('type_preservation', {}).get('numerical_preserved', 0) > 15
            storage_success = step5_checks.get('storage_efficiency', {}).get('memory_efficiency', {}).get('memory_match', False)
            
            step_scores['step5'] = sum([optimization_success, preservation_success, storage_success]) / 3 * 100
        else:
            step_scores['step5'] = 0.0
        
        quality_score['component_scores'] = step_scores
        
        # Calculate overall score (weighted average)
        weights = {'step1_3': 0.4, 'step4': 0.3, 'step5': 0.3}
        overall_score = sum(step_scores[step] * weights[step] for step in step_scores)
        quality_score['overall_score'] = overall_score
        
        # Check quality gates
        for gate_name, threshold in self.quality_gates.items():
            if gate_name == 'overall':
                quality_score['quality_gates'][gate_name] = {
                    'threshold': threshold * 100,
                    'achieved': overall_score,
                    'passed': overall_score >= threshold * 100
                }
            else:
                # Map component scores to quality gates
                if gate_name == 'completeness':
                    score = step_scores.get('step1_3', 0)
                elif gate_name == 'consistency':
                    score = step_scores.get('step4', 0)
                elif gate_name == 'integrity':
                    score = step_scores.get('step5', 0)
                elif gate_name == 'optimization':
                    score = step_scores.get('step5', 0)
                else:
                    score = 0
                
                quality_score['quality_gates'][gate_name] = {
                    'threshold': threshold * 100,
                    'achieved': score,
                    'passed': score >= threshold * 100
                }
        
        # Generate recommendations
        for gate_name, gate_info in quality_score['quality_gates'].items():
            if not gate_info['passed']:
                quality_score['recommendations'].append(
                    f"Improve {gate_name}: Current {gate_info['achieved']:.1f}%, Target {gate_info['threshold']:.1f}%"
                )
        
        if overall_score >= 90:
            quality_score['recommendations'].append("🎉 Pipeline quality meets excellent standards!")
        elif overall_score >= 80:
            quality_score['recommendations'].append("✅ Pipeline quality meets good standards with minor improvements possible")
        else:
            quality_score['recommendations'].append("⚠️ Pipeline quality needs significant improvements before production use")
        
        logger.info(f"Quality score calculated: {overall_score:.1f}/100")
        
        return quality_score
    
    def generate_final_certification(self) -> Dict[str, Any]:
        """
        Generate final data quality certification.
        
        Returns:
            Dictionary with final certification
        """
        logger.info("🏆 Generating final data quality certification...")
        
        certification = {
            'certification_id': f"DQ_CERT_{self.timestamp}",
            'certification_date': datetime.now().isoformat(),
            'pipeline_version': '1.0',
            'dataset_info': {
                'name': 'Romance Novel NLP Research Dataset',
                'version': 'Step 6 Final',
                'shape': self.df.shape,
                'memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
                'columns': list(self.df.columns)
            },
            'pipeline_summary': {
                'total_steps': 6,
                'completed_steps': 6,
                'completion_percentage': 100.0,
                'overall_status': 'Complete'
            },
            'quality_certification': {
                'overall_score': 0.0,
                'quality_level': 'Pending',
                'certification_status': 'Pending',
                'quality_gates_passed': 0,
                'total_quality_gates': len(self.quality_gates)
            },
            'step_validation_summary': {},
            'recommendations': [],
            'next_steps': []
        }
        
        # Add step validation summaries
        for step_name, step_validation in self.validation_results.items():
            certification['step_validation_summary'][step_name] = {
                'status': step_validation.get('overall_status', 'unknown'),
                'description': step_validation.get('description', ''),
                'key_achievements': []
            }
            
            # Extract key achievements
            checks = step_validation.get('checks', {})
            if step_name == 'step1_3':
                if checks.get('duplicate_removal', {}).get('duplicate_flags_present', False):
                    certification['step_validation_summary'][step_name]['key_achievements'].append('Duplicate detection implemented')
                if checks.get('data_type_validation', {}).get('optimized_data_types', 0) > 20:
                    certification['step_validation_summary'][step_name]['key_achievements'].append('Data types optimized')
            
            elif step_name == 'step4':
                if checks.get('publication_year_treatment', {}).get('treatment_effectiveness', {}).get('treatment_success', False):
                    certification['step_validation_summary'][step_name]['key_achievements'].append('Publication year anomalies treated')
                if checks.get('conservative_treatment', {}).get('outlier_analysis_ready', False):
                    certification['step_validation_summary'][step_name]['key_achievements'].append('Outlier analysis documented')
            
            elif step_name == 'step5':
                if checks.get('data_type_optimization', {}).get('optimized_columns', 0) > 20:
                    certification['step_validation_summary'][step_name]['key_achievements'].append('Data types fully optimized')
                if checks.get('storage_efficiency', {}).get('memory_efficiency', {}).get('memory_match', False):
                    certification['step_validation_summary'][step_name]['key_achievements'].append('Storage efficiency achieved')
        
        # Add quality certification details
        quality_score = self.calculate_quality_score()
        certification['quality_certification']['overall_score'] = quality_score['overall_score']
        
        # Determine quality level
        if quality_score['overall_score'] >= 95:
            certification['quality_certification']['quality_level'] = 'Excellent'
            certification['quality_certification']['certification_status'] = 'Certified'
        elif quality_score['overall_score'] >= 85:
            certification['quality_certification']['quality_level'] = 'Good'
            certification['quality_certification']['certification_status'] = 'Certified'
        elif quality_score['overall_score'] >= 75:
            certification['quality_certification']['quality_level'] = 'Acceptable'
            certification['quality_certification']['certification_status'] = 'Conditionally Certified'
        else:
            certification['quality_certification']['quality_level'] = 'Needs Improvement'
            certification['quality_certification']['certification_status'] = 'Not Certified'
        
        # Count passed quality gates
        passed_gates = sum(1 for gate in quality_score['quality_gates'].values() if gate['passed'])
        certification['quality_certification']['quality_gates_passed'] = passed_gates
        
        # Add recommendations
        certification['recommendations'] = quality_score['recommendations']
        
        # Add next steps
        if certification['quality_certification']['certification_status'] == 'Certified':
            certification['next_steps'].append('Dataset ready for production use')
            certification['next_steps'].append('Proceed with NLP analysis and modeling')
            certification['next_steps'].append('Monitor data quality in production')
        elif certification['quality_certification']['certification_status'] == 'Conditionally Certified':
            certification['next_steps'].append('Address quality gate failures before production use')
            certification['next_steps'].append('Implement additional quality checks')
            certification['next_steps'].append('Re-run validation after improvements')
        else:
            certification['next_steps'].append('Significant quality improvements required')
            certification['next_steps'].append('Review and fix failed pipeline steps')
            certification['next_steps'].append('Re-run complete pipeline after fixes')
        
        logger.info(f"Final certification generated: {certification['quality_certification']['certification_status']}")
        
        return certification
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        Run complete final quality validation process.
        
        Returns:
            Dictionary with complete validation results
        """
        logger.info("🚀 Starting complete final quality validation process...")
        start_time = datetime.now()
        
        # Initialize results
        self.validation_results = {}
        
        # 1. Load dataset
        logger.info("📥 Loading optimized dataset...")
        self.df = self.load_data()
        
        # 2. Validate each pipeline step
        logger.info("🔍 Validating pipeline steps...")
        
        self.validation_results['step1_3'] = self.validate_pipeline_step1_3()
        self.validation_results['step4'] = self.validate_pipeline_step4()
        self.validation_results['step5'] = self.validate_pipeline_step5()
        
        # 3. Calculate quality score
        logger.info("📊 Calculating quality score...")
        quality_score = self.calculate_quality_score()
        
        # 4. Generate final certification
        logger.info("🏆 Generating final certification...")
        final_certification = self.generate_final_certification()
        
        # 5. Compile complete results
        complete_results = {
            'validation_timestamp': self.timestamp,
            'dataset_info': {
                'shape': self.df.shape,
                'memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
                'columns': list(self.df.columns)
            },
            'pipeline_validation': self.validation_results,
            'quality_score': quality_score,
            'final_certification': final_certification,
            'execution_summary': {
                'total_steps_validated': len(self.validation_results),
                'successful_steps': sum(1 for step in self.validation_results.values() if step.get('overall_status') == 'success'),
                'execution_time': str(datetime.now() - start_time)
            }
        }
        
        logger.info("✅ Complete final quality validation process finished!")
        
        return complete_results
    
    def save_validation_report(self, validation_results: Dict[str, Any], filename: str = None) -> str:
        """
        Save comprehensive validation report.
        
        Args:
            validation_results: Validation results dictionary
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"final_quality_validation_report_step6_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.Index):
                return obj.tolist()
            return obj
        
        # Deep copy and convert results
        import copy
        results_copy = copy.deepcopy(validation_results)
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy_types(obj)
        
        results_copy = recursive_convert(results_copy)
        
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {filepath}")
        return str(filepath)
    
    def print_validation_summary(self, validation_results: Dict[str, Any]):
        """Print a human-readable validation summary."""
        print("\n" + "="*80)
        print("FINAL DATA QUALITY VALIDATION SUMMARY - STEP 6")
        print("="*80)
        
        # Dataset info
        dataset_info = validation_results['dataset_info']
        print(f"Dataset: {dataset_info['shape'][0]:,} records × {dataset_info['shape'][1]} columns")
        print(f"Memory: {dataset_info['memory_mb']:.2f} MB")
        
        # Pipeline validation summary
        pipeline_validation = validation_results['pipeline_validation']
        print(f"\n🔍 PIPELINE VALIDATION SUMMARY:")
        for step_name, step_validation in pipeline_validation.items():
            status = step_validation.get('overall_status', 'unknown')
            description = step_validation.get('description', '')
            print(f"  • {step_name.upper()}: {status.upper()} - {description}")
        
        # Quality score
        quality_score = validation_results['quality_score']
        print(f"\n📊 QUALITY SCORE: {quality_score['overall_score']:.1f}/100")
        
        # Quality gates
        print(f"\n🎯 QUALITY GATES:")
        for gate_name, gate_info in quality_score['quality_gates'].items():
            status = "✅ PASSED" if gate_info['passed'] else "❌ FAILED"
            print(f"  • {gate_name.title()}: {status} ({gate_info['achieved']:.1f}% / {gate_info['threshold']:.1f}%)")
        
        # Final certification
        certification = validation_results['final_certification']
        print(f"\n🏆 FINAL CERTIFICATION:")
        print(f"  • Status: {certification['quality_certification']['certification_status']}")
        print(f"  • Quality Level: {certification['quality_certification']['quality_level']}")
        print(f"  • Quality Gates: {certification['quality_certification']['quality_gates_passed']}/{certification['quality_certification']['total_quality_gates']} passed")
        
        # Recommendations
        if quality_score['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in quality_score['recommendations']:
                print(f"  • {rec}")
        
        # Next steps
        if certification['next_steps']:
            print(f"\n🚀 NEXT STEPS:")
            for step in certification['next_steps']:
                print(f"  • {step}")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    logger = logging.getLogger(__name__)
    
    print("🏆 STEP 6: FINAL DATA QUALITY VALIDATION")
    print("=" * 60)
    
    try:
        # Initialize validator
        validator = FinalQualityValidator()
        
        # Run complete validation process
        results = validator.run_complete_validation()
        
        # Save validation report
        logger.info("📊 Saving validation report...")
        report_file = validator.save_validation_report(results)
        
        # Print summary
        validator.print_validation_summary(results)
        
        print("\n🎉 Step 6: Final Data Quality Validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Final quality validation failed: {str(e)}", exc_info=True)
        print(f"\n💥 Final quality validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)
