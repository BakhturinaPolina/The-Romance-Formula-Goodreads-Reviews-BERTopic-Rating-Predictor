#!/usr/bin/env python3
"""
EDA Analysis for Title Cleaning Outputs

This script analyzes the outputs from the title cleaning process to:
1. Assess data quality and completeness
2. Understand the cleaning strategies applied
3. Evaluate integration readiness with main dataset
4. Recommend next steps for the romance novel NLP research pipeline

Author: AI Assistant
Date: 2025-09-02
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/title_cleaning_eda_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TitleCleaningEDA:
    """EDA analysis for title cleaning outputs."""
    
    def __init__(self, data_dir: str = "outputs/title_cleaning"):
        """Initialize the EDA analyzer."""
        self.data_dir = Path(data_dir)
        self.cleaned_data = None
        self.cleaning_report = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load the cleaned titles data and cleaning report."""
        try:
            # Find the most recent cleaned titles file
            cleaned_files = list(self.data_dir.glob("cleaned_titles_*.csv"))
            if not cleaned_files:
                raise FileNotFoundError("No cleaned titles CSV files found")
            
            latest_file = max(cleaned_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading cleaned titles from: {latest_file}")
            
            # Load data with progress tracking
            logger.info("Loading cleaned titles data...")
            self.cleaned_data = pd.read_csv(latest_file)
            logger.info(f"Loaded {len(self.cleaned_data):,} records")
            
            # Load cleaning report
            report_files = list(self.data_dir.glob("title_cleaning_report_*.json"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Loading cleaning report from: {latest_report}")
                with open(latest_report, 'r') as f:
                    self.cleaning_report = json.load(f)
                logger.info("Cleaning report loaded successfully")
            else:
                logger.warning("No cleaning report found")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def basic_data_overview(self):
        """Generate basic data overview statistics."""
        logger.info("Generating basic data overview...")
        
        if self.cleaned_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        overview = {
            'total_records': len(self.cleaned_data),
            'total_columns': len(self.cleaned_data.columns),
            'memory_usage_mb': self.cleaned_data.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_titles': self.cleaned_data['duplication_status'].value_counts().to_dict(),
            'cleaning_strategies': self.cleaned_data['cleaning_strategy'].value_counts().to_dict()
        }
        
        # Add column info
        overview['columns'] = list(self.cleaned_data.columns)
        overview['data_types'] = {str(k): str(v) for k, v in self.cleaned_data.dtypes.to_dict().items()}
        
        # Add sample data
        overview['sample_titles'] = self.cleaned_data['title'].head(10).tolist()
        
        self.analysis_results['overview'] = overview
        logger.info(f"Basic overview completed: {overview['total_records']:,} records")
        
        return overview
    
    def quality_assessment(self):
        """Assess data quality and completeness."""
        logger.info("Assessing data quality...")
        
        quality_metrics = {}
        
        # Check for null values
        null_counts = self.cleaned_data.isnull().sum()
        quality_metrics['null_counts'] = {str(k): int(v) for k, v in null_counts.to_dict().items()}
        quality_metrics['completeness_percentage'] = {str(k): float(v) for k, v in ((len(self.cleaned_data) - null_counts) / len(self.cleaned_data) * 100).to_dict().items()}
        
        # Check for empty strings
        empty_strings = {}
        text_columns = ['title', 'description', 'author_name', 'series_title']
        for col in text_columns:
            if col in self.cleaned_data.columns:
                empty_strings[col] = int((self.cleaned_data[col] == '').sum())
        quality_metrics['empty_strings'] = empty_strings
        
        # Check data consistency
        quality_metrics['publication_year_range'] = {
            'min': int(self.cleaned_data['publication_year'].min()),
            'max': int(self.cleaned_data['publication_year'].max()),
            'unique_years': int(self.cleaned_data['publication_year'].nunique())
        }
        
        # Check language codes
        if 'language_codes_en' in self.cleaned_data.columns:
            language_counts = self.cleaned_data['language_codes_en'].value_counts()
            quality_metrics['language_distribution'] = {str(k): int(v) for k, v in language_counts.head(10).to_dict().items()}
        
        self.analysis_results['quality'] = quality_metrics
        logger.info("Quality assessment completed")
        
        return quality_metrics
    
    def duplicate_analysis(self):
        """Analyze duplicate handling and cleaning strategies."""
        logger.info("Analyzing duplicate handling...")
        
        duplicate_analysis = {}
        
        # Analyze duplication status
        dup_status = self.cleaned_data['duplication_status'].value_counts()
        duplicate_analysis['duplication_status_distribution'] = {str(k): int(v) for k, v in dup_status.to_dict().items()}
        
        # Analyze cleaning strategies
        cleaning_strategies = self.cleaned_data['cleaning_strategy'].value_counts()
        duplicate_analysis['cleaning_strategies_distribution'] = {str(k): int(v) for k, v in cleaning_strategies.to_dict().items()}
        
        # Analyze disambiguation notes
        if 'disambiguation_notes' in self.cleaned_data.columns:
            disambig_counts = int(self.cleaned_data['disambiguation_notes'].notna().sum())
            duplicate_analysis['records_with_disambiguation'] = disambig_counts
            
            # Sample disambiguation notes
            sample_disambig = self.cleaned_data[self.cleaned_data['disambiguation_notes'].notna()]['disambiguation_notes'].head(5).tolist()
            duplicate_analysis['sample_disambiguation_notes'] = sample_disambig
        
        # Analyze series data for duplicates
        if 'series_id' in self.cleaned_data.columns:
            series_duplicates = self.cleaned_data.groupby('series_id').size()
            duplicate_analysis['series_size_distribution'] = {
                'single_books': int((series_duplicates == 1).sum()),
                'series_books': int((series_duplicates > 1).sum()),
                'max_series_size': int(series_duplicates.max())
            }
        
        self.analysis_results['duplicate_analysis'] = duplicate_analysis
        logger.info("Duplicate analysis completed")
        
        return duplicate_analysis
    
    def integration_readiness_assessment(self):
        """Assess if data is ready for integration with main dataset."""
        logger.info("Assessing integration readiness...")
        
        readiness = {
            'ready_for_integration': True,
            'issues_found': [],
            'recommendations': []
        }
        
        # Check critical fields
        critical_fields = ['work_id', 'title', 'publication_year', 'author_id', 'author_name']
        missing_critical = []
        for field in critical_fields:
            if field not in self.cleaned_data.columns:
                missing_critical.append(field)
            elif self.cleaned_data[field].isnull().sum() > 0:
                missing_critical.append(field)
        
        if missing_critical:
            readiness['ready_for_integration'] = False
            readiness['issues_found'].append(f"Missing or incomplete critical fields: {missing_critical}")
        
        # Check data consistency
        if 'publication_year' in self.cleaned_data.columns:
            invalid_years = self.cleaned_data[
                (self.cleaned_data['publication_year'] < 1900) | 
                (self.cleaned_data['publication_year'] > 2025)
            ]
            if len(invalid_years) > 0:
                readiness['issues_found'].append(f"Found {len(invalid_years)} records with invalid publication years")
        
        # Check for data type issues
        if 'work_id' in self.cleaned_data.columns:
            try:
                pd.to_numeric(self.cleaned_data['work_id'], errors='raise')
            except:
                readiness['issues_found'].append("work_id contains non-numeric values")
        
        # Generate recommendations
        if readiness['ready_for_integration']:
            readiness['recommendations'].append("Data appears ready for integration with main dataset")
        else:
            readiness['recommendations'].append("Address identified issues before integration")
        
        readiness['recommendations'].append("Consider merging with main processed dataset")
        readiness['recommendations'].append("Validate against original data sources")
        
        self.analysis_results['integration_readiness'] = readiness
        logger.info("Integration readiness assessment completed")
        
        return readiness
    
    def next_steps_recommendation(self):
        """Recommend next steps based on analysis results."""
        logger.info("Generating next steps recommendations...")
        
        recommendations = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_goals': [],
            'priority_order': []
        }
        
        # Based on integration readiness
        if self.analysis_results.get('integration_readiness', {}).get('ready_for_integration', False):
            recommendations['immediate_actions'].append("Integrate cleaned titles with main processed dataset")
            recommendations['immediate_actions'].append("Validate integration results")
        else:
            recommendations['immediate_actions'].append("Fix identified data quality issues")
            recommendations['immediate_actions'].append("Re-run quality assessment")
        
        # Based on duplicate analysis
        dup_analysis = self.analysis_results.get('duplicate_analysis', {})
        if dup_analysis.get('records_with_disambiguation', 0) > 0:
            recommendations['short_term_goals'].append("Implement disambiguation handling in NLP preprocessing")
            recommendations['short_term_goals'].append("Create disambiguation metadata for downstream analysis")
        
        # Based on overall data quality
        quality = self.analysis_results.get('quality', {})
        if quality.get('completeness_percentage', {}).get('description', 0) < 90:
            recommendations['short_term_goals'].append("Address missing descriptions for NLP analysis")
        
        # Long-term goals
        recommendations['long_term_goals'].append("Proceed with topic modeling on cleaned dataset")
        recommendations['long_term_goals'].append("Implement series-level analysis capabilities")
        recommendations['long_term_goals'].append("Begin review sentiment analysis")
        
        # Priority order
        recommendations['priority_order'] = [
            "Data integration and validation",
            "NLP preprocessing pipeline setup",
            "Topic modeling implementation",
            "Series analysis development",
            "Review analysis pipeline"
        ]
        
        self.analysis_results['recommendations'] = recommendations
        logger.info("Next steps recommendations generated")
        
        return recommendations
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        logger.info("Generating summary report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': self.analysis_results.get('overview', {}),
            'quality_summary': self.analysis_results.get('quality', {}),
            'duplicate_summary': self.analysis_results.get('duplicate_analysis', {}),
            'integration_status': self.analysis_results.get('integration_readiness', {}),
            'next_steps': self.analysis_results.get('recommendations', {}),
            'overall_assessment': self._generate_overall_assessment()
        }
        
        # Save report
        report_path = f"outputs/title_cleaning_eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to: {report_path}")
        return report
    
    def _generate_overall_assessment(self):
        """Generate overall assessment of the title cleaning outputs."""
        assessment = {
            'status': 'UNKNOWN',
            'score': 0,
            'strengths': [],
            'weaknesses': [],
            'overall_quality': 'UNKNOWN'
        }
        
        # Calculate quality score
        quality = self.analysis_results.get('quality', {})
        completeness = quality.get('completeness_percentage', {})
        
        if completeness:
            avg_completeness = sum(completeness.values()) / len(completeness)
            assessment['score'] = avg_completeness
            
            if avg_completeness >= 95:
                assessment['overall_quality'] = 'EXCELLENT'
                assessment['status'] = 'READY_FOR_NLP'
            elif avg_completeness >= 90:
                assessment['overall_quality'] = 'GOOD'
                assessment['status'] = 'READY_FOR_NLP_WITH_MINOR_ISSUES'
            elif avg_completeness >= 80:
                assessment['overall_quality'] = 'FAIR'
                assessment['status'] = 'NEEDS_IMPROVEMENT'
            else:
                assessment['overall_quality'] = 'POOR'
                assessment['status'] = 'NOT_READY'
        
        # Identify strengths and weaknesses
        if self.analysis_results.get('integration_readiness', {}).get('ready_for_integration', False):
            assessment['strengths'].append("Data ready for integration")
        else:
            assessment['weaknesses'].append("Integration issues identified")
        
        dup_analysis = self.analysis_results.get('duplicate_analysis', {})
        if dup_analysis.get('records_with_disambiguation', 0) > 0:
            assessment['strengths'].append("Comprehensive duplicate handling")
        
        return assessment
    
    def run_full_analysis(self):
        """Run the complete EDA analysis pipeline."""
        logger.info("Starting full EDA analysis...")
        
        try:
            # Load data
            self.load_data()
            
            # Run analysis components
            self.basic_data_overview()
            self.quality_assessment()
            self.duplicate_analysis()
            self.integration_readiness_assessment()
            self.next_steps_recommendation()
            
            # Generate final report
            report = self.generate_summary_report()
            
            logger.info("Full EDA analysis completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

def main():
    """Main execution function."""
    print("=" * 80)
    print("Title Cleaning Outputs EDA Analysis")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = TitleCleaningEDA()
        
        # Run full analysis
        report = analyzer.run_full_analysis()
        
        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        
        overview = report['data_summary']
        print(f"Total Records: {overview['total_records']:,}")
        print(f"Total Columns: {overview['total_columns']}")
        print(f"Memory Usage: {overview['memory_usage_mb']:.2f} MB")
        
        assessment = report['overall_assessment']
        print(f"\nOverall Quality: {assessment['overall_quality']}")
        print(f"Status: {assessment['status']}")
        print(f"Quality Score: {assessment['score']:.1f}%")
        
        print(f"\nNext Priority: {report['next_steps']['priority_order'][0]}")
        
        print("\n" + "=" * 80)
        print("Analysis complete! Check the generated report for detailed findings.")
        print("=" * 80)
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        logger.error(f"Main execution failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
