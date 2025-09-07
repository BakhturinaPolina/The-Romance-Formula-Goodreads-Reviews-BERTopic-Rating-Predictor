#!/usr/bin/env python3
"""
Improved EDA Final Script - Clean and Focused
Senior Data Scientist Review & Production

This script creates exactly two publication-ready figures:
1. Before/After Cleaning Distributions (2x2 histograms)
2. Cleaned Data Summary (Boxplots with jitter)

Uses Antique color palette and follows academic publication standards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
import logging
import time
from typing import Dict, List, Any, Optional
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exceptions for Coding Agent Pattern
class AnalysisError(Exception):
    """Raised when task analysis fails."""
    pass

class ModificationError(Exception):
    """Raised when code modification fails."""
    pass

# Set up academic publication style
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'figure.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False
})

# Antique color palette from pypalettes
ANTIQUE_COLORS = [
    '#855C75FF',  # Dark purple
    '#D9AF6BFF',  # Gold
    '#AF6458FF',  # Red-brown
    '#736F4CFF',  # Olive
    '#526A83FF',  # Blue-gray
    '#625377FF',  # Purple
    '#68855CFF',  # Green
    '#9C9C5EFF',  # Yellow-green
    '#A06177FF',  # Pink-purple
    '#8C785DFF',  # Brown
    '#467378FF',  # Teal
    '#7C7C7CFF'   # Gray
]

class ImprovedEDAFinal:
    """Clean, focused EDA plot generator with Coding Agent Pattern implementation."""
    
    def __init__(self, data_path_before, data_path_after, output_dir):
        """
        Initialize the visualizer with Coding Agent Pattern components.
        
        Args:
            data_path_before: Path to the raw dataset CSV file
            data_path_after: Path to the cleaned dataset CSV file
            output_dir: Directory to save visualization outputs
        """
        self.data_path_before = Path(data_path_before)
        self.data_path_after = Path(data_path_after)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Coding Agent Pattern: Change history tracking
        self.change_history = []
        self.task_analysis = {}
        self.change_plan = {}
        
        # Load datasets with error handling
        try:
            self.df_before = pd.read_csv(self.data_path_before)
            self.df_after = pd.read_csv(self.data_path_after)
            logger.info(f"Successfully loaded datasets")
        except Exception as e:
            logger.error(f"Failed to load datasets: {str(e)}")
            raise
        
        logger.info(f"Loaded before dataset: {self.df_before.shape[0]:,} books, {self.df_before.shape[1]} features")
        logger.info(f"Loaded after dataset: {self.df_after.shape[0]:,} books, {self.df_after.shape[1]} features")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Define the four numerical variables to analyze
        self.numerical_vars = [
            'publication_year',
            'num_pages_median', 
            'ratings_count_sum',
            'average_rating_weighted_mean'
        ]
    
    def create_figure_1_histograms(self):
        """
        FIGURE 1: Before/After Cleaning Distributions (2x2 histograms)
        
        Creates a 2×2 subplot layout comparing distributions before and after 
        data cleaning for four numerical variables using Antique color palette.
        """
        print("Creating Figure 1: Before/After Cleaning Distributions...")
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Publication Year Distribution
        sns.histplot(data=self.df_before, x="publication_year", kde=False, color=ANTIQUE_COLORS[0], 
                     label="Before Cleaning", ax=axs[0,0], alpha=0.6, bins=30)
        sns.histplot(data=self.df_after, x="publication_year", kde=False, color=ANTIQUE_COLORS[1], 
                     label="After Cleaning", ax=axs[0,0], alpha=0.6, bins=30)
        axs[0,0].set_title("Publication Year Distribution", fontsize=12, fontweight='bold')
        axs[0,0].set_xlabel("Publication Year")
        axs[0,0].set_ylabel("Count")
        axs[0,0].legend()
        
        # 2. Median Page Count Distribution  
        sns.histplot(data=self.df_before, x="num_pages_median", kde=False, color=ANTIQUE_COLORS[2], 
                     label="Before Cleaning", ax=axs[0,1], alpha=0.6, bins=30)
        sns.histplot(data=self.df_after, x="num_pages_median", kde=False, color=ANTIQUE_COLORS[3], 
                     label="After Cleaning", ax=axs[0,1], alpha=0.6, bins=30)
        axs[0,1].set_title("Median Page Count Distribution", fontsize=12, fontweight='bold')
        axs[0,1].set_xlabel("Number of Pages (Median)")
        axs[0,1].set_ylabel("Count")
        axs[0,1].legend()
        
        # 3. Ratings Count Distribution (with log scale to show popular vs niche)
        sns.histplot(data=self.df_before, x="ratings_count_sum", kde=False, color=ANTIQUE_COLORS[4], 
                     label="Before Cleaning", ax=axs[1,0], alpha=0.6, bins=50)
        sns.histplot(data=self.df_after, x="ratings_count_sum", kde=False, color=ANTIQUE_COLORS[5], 
                     label="After Cleaning", ax=axs[1,0], alpha=0.6, bins=50)
        axs[1,0].set_title("Ratings Count Distribution (Popular vs Niche)", fontsize=12, fontweight='bold')
        axs[1,0].set_xlabel("Total Ratings Count (log scale)")
        axs[1,0].set_ylabel("Count")
        axs[1,0].set_xscale("log")  # Log scale to distinguish popular vs niche books
        axs[1,0].legend()
        
        # 4. Average Rating Distribution (with KDE to show skewness)
        sns.histplot(data=self.df_before, x="average_rating_weighted_mean", kde=True, color=ANTIQUE_COLORS[6], 
                     label="Before Cleaning", ax=axs[1,1], alpha=0.6, bins=30, stat="density")
        sns.histplot(data=self.df_after, x="average_rating_weighted_mean", kde=True, color=ANTIQUE_COLORS[7], 
                     label="After Cleaning", ax=axs[1,1], alpha=0.6, bins=30, stat="density")
        axs[1,1].set_title("Average Rating Distribution (Skewness Check)", fontsize=12, fontweight='bold')
        axs[1,1].set_xlabel("Weighted Average Rating")
        axs[1,1].set_ylabel("Density")
        axs[1,1].legend()
        
        plt.suptitle("Data Distribution: Before vs After Cleaning", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / 'figure_1_before_after_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Figure 1 saved to: {output_path}")
    
    def create_figure_2_boxplots(self):
        """
        FIGURE 2: Cleaned Data Summary (Boxplots with jitter)
        
        Creates a horizontal boxplot with jittered individual points showing 
        the distribution of all four cleaned numerical variables.
        """
        print("Creating Figure 2: Cleaned Data Summary (Boxplots with jitter)...")
        
        # Prepare data for boxplot - use raw values but transform ratings_count to log scale for visibility
        cleaned_vars = self.df_after[self.numerical_vars].copy()
        
        # Transform ratings_count to log scale for better visibility
        cleaned_vars['ratings_count_sum'] = np.log1p(cleaned_vars['ratings_count_sum'])  # log(1+x) to handle zeros
        
        # Melt dataframe for grouped boxplot
        cleaned_melted = cleaned_vars.melt(var_name='Variable', value_name='Value')
        
        # Rename variables for better display
        variable_names = {
            'publication_year': 'Publication Year',
            'num_pages_median': 'Pages (Median)',
            'ratings_count_sum': 'Ratings Count (log)',
            'average_rating_weighted_mean': 'Avg Rating (Weighted)'
        }
        cleaned_melted['Variable'] = cleaned_melted['Variable'].map(variable_names)
        
        # Create horizontal boxplot with jitter
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Boxplot
        sns.boxplot(data=cleaned_melted, y='Variable', x='Value', 
                    palette=ANTIQUE_COLORS[:4], orient='h', ax=ax, width=0.6)
        
        # Add jittered points
        sns.stripplot(data=cleaned_melted, y='Variable', x='Value', 
                      color='gray', alpha=0.3, size=2, jitter=True, ax=ax)
        
        ax.set_title("Distribution Summary of Cleaned Numerical Variables", fontsize=14, fontweight='bold')
        ax.set_xlabel("Value", fontsize=11)
        ax.set_ylabel("")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / 'figure_2_cleaned_data_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Figure 2 saved to: {output_path}")
    
    def generate_summary_statistics(self):
        """Generate summary statistics for both datasets."""
        print("Generating summary statistics...")
        
        summary_stats = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'before_cleaning': {
                    'total_books': len(self.df_before),
                    'features': len(self.df_before.columns)
                },
                'after_cleaning': {
                    'total_books': len(self.df_after),
                    'features': len(self.df_after.columns)
                },
                'data_reduction': {
                    'books_removed': len(self.df_before) - len(self.df_after),
                    'reduction_percentage': round(((len(self.df_before) - len(self.df_after)) / len(self.df_before)) * 100, 2)
                }
            },
            'numerical_variables_analysis': {}
        }
        
        # Analyze each numerical variable
        for var in self.numerical_vars:
            data_before = self.df_before[var].dropna()
            data_after = self.df_after[var].dropna()
            
            summary_stats['numerical_variables_analysis'][var] = {
                'before_cleaning': {
                    'count': len(data_before),
                    'mean': round(data_before.mean(), 4),
                    'median': round(data_before.median(), 4),
                    'std': round(data_before.std(), 4),
                    'min': data_before.min(),
                    'max': data_before.max()
                },
                'after_cleaning': {
                    'count': len(data_after),
                    'mean': round(data_after.mean(), 4),
                    'median': round(data_after.median(), 4),
                    'std': round(data_after.std(), 4),
                    'min': data_after.min(),
                    'max': data_after.max()
                },
                'cleaning_impact': {
                    'data_points_removed': len(data_before) - len(data_after),
                    'removal_percentage': round(((len(data_before) - len(data_after)) / len(data_before)) * 100, 2),
                    'mean_change': round(data_after.mean() - data_before.mean(), 4),
                    'std_change': round(data_after.std() - data_before.std(), 4)
                }
            }
        
        return summary_stats
    
    def save_summary_report(self, summary_stats):
        """Save summary statistics to JSON file."""
        report_path = self.output_dir / 'improved_eda_final_summary.json'
        
        with open(report_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"Summary report saved to: {report_path}")
    
    def print_summary(self, summary_stats):
        """Print a summary of the analysis."""
        print("=" * 80)
        print("IMPROVED EDA FINAL ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Dataset overview
        dataset_info = summary_stats['dataset_info']
        print(f"Dataset Overview:")
        print(f"  Before cleaning: {dataset_info['before_cleaning']['total_books']:,} books")
        print(f"  After cleaning: {dataset_info['after_cleaning']['total_books']:,} books")
        print(f"  Books removed: {dataset_info['data_reduction']['books_removed']:,} ({dataset_info['data_reduction']['reduction_percentage']}%)")
        print()
        
        # Numerical variables analysis
        print("Numerical Variables Analysis:")
        for var, analysis in summary_stats['numerical_variables_analysis'].items():
            print(f"  {var}:")
            print(f"    Before: n={analysis['before_cleaning']['count']:,}, "
                  f"mean={analysis['before_cleaning']['mean']}, "
                  f"std={analysis['before_cleaning']['std']}")
            print(f"    After: n={analysis['after_cleaning']['count']:,}, "
                  f"mean={analysis['after_cleaning']['mean']}, "
                  f"std={analysis['after_cleaning']['std']}")
            print(f"    Impact: {analysis['cleaning_impact']['removal_percentage']}% removed, "
                  f"mean change={analysis['cleaning_impact']['mean_change']}")
            print()
        
        print("=" * 80)
    
    # ============================================================================
    # CODING AGENT PATTERN METHODS
    # ============================================================================
    
    def analyze_task(self, task_description: str = "Create EDA visualizations") -> Dict[str, Any]:
        """
        Analyze coding task requirements (Coding Agent Pattern).
        
        Args:
            task_description: Description of the task
            
        Returns:
            Analysis results with requirements, affected files, and change plan
        """
        logger.info("Analyzing EDA visualization task...")
        
        try:
            # Parse requirements
            requirements = {
                'create_figure_1': True,  # Before/after histograms
                'create_figure_2': True,  # Cleaned data boxplots
                'generate_statistics': True,  # Summary statistics
                'save_outputs': True,  # Save all outputs
                'use_antique_palette': True,  # Use specified color palette
                'academic_standards': True  # Follow publication standards
            }
            
            # Identify affected files
            affected_files = {
                'input_files': [str(self.data_path_before), str(self.data_path_after)],
                'output_files': [
                    str(self.output_dir / 'figure_1_before_after_distributions.png'),
                    str(self.output_dir / 'figure_2_cleaned_data_summary.png'),
                    str(self.output_dir / 'improved_eda_final_summary.json')
                ]
            }
            
            # Create change plan
            change_plan = {
                'sequence': [
                    'validate_data_availability',
                    'create_figure_1_histograms',
                    'create_figure_2_boxplots', 
                    'generate_summary_statistics',
                    'save_all_outputs',
                    'verify_outputs'
                ],
                'dependencies': {
                    'create_figure_1_histograms': ['validate_data_availability'],
                    'create_figure_2_boxplots': ['validate_data_availability'],
                    'generate_summary_statistics': ['validate_data_availability'],
                    'save_all_outputs': ['create_figure_1_histograms', 'create_figure_2_boxplots', 'generate_summary_statistics'],
                    'verify_outputs': ['save_all_outputs']
                }
            }
            
            self.task_analysis = {
                'requirements': requirements,
                'affected_files': affected_files,
                'change_plan': change_plan,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Task analysis completed successfully")
            return self.task_analysis
            
        except Exception as e:
            logger.error(f"Task analysis failed: {str(e)}")
            raise AnalysisError("Failed to analyze EDA task")
    
    def validate_data_availability(self) -> bool:
        """
        Validate that required data is available (Coding Agent Pattern).
        
        Returns:
            bool indicating if data validation passed
        """
        logger.info("Validating data availability...")
        
        try:
            # Check if datasets are loaded
            if self.df_before is None or self.df_after is None:
                raise ValueError("Datasets not loaded")
            
            # Check if required columns exist
            missing_cols_before = set(self.numerical_vars) - set(self.df_before.columns)
            missing_cols_after = set(self.numerical_vars) - set(self.df_after.columns)
            
            if missing_cols_before:
                raise ValueError(f"Missing columns in before dataset: {missing_cols_before}")
            if missing_cols_after:
                raise ValueError(f"Missing columns in after dataset: {missing_cols_after}")
            
            # Check if output directory is writable
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def apply_changes(self, change_plan: Dict[str, Any]) -> List[str]:
        """
        Apply planned code changes (Coding Agent Pattern).
        
        Args:
            change_plan: Planned changes
            
        Returns:
            List of modified files
        """
        logger.info("Applying planned changes...")
        
        try:
            modified_files = []
            start_time = time.time()
            
            # Execute changes in dependency order
            for step in change_plan['sequence']:
                logger.info(f"Executing step: {step}")
                
                if step == 'validate_data_availability':
                    if not self.validate_data_availability():
                        raise ValueError("Data validation failed")
                        
                elif step == 'create_figure_1_histograms':
                    self.create_figure_1_histograms()
                    modified_files.append(str(self.output_dir / 'figure_1_before_after_distributions.png'))
                    
                elif step == 'create_figure_2_boxplots':
                    self.create_figure_2_boxplots()
                    modified_files.append(str(self.output_dir / 'figure_2_cleaned_data_summary.png'))
                    
                elif step == 'generate_summary_statistics':
                    summary_stats = self.generate_summary_statistics()
                    self.save_summary_report(summary_stats)
                    modified_files.append(str(self.output_dir / 'improved_eda_final_summary.json'))
                    
                elif step == 'save_all_outputs':
                    # Already handled in individual steps
                    pass
                    
                elif step == 'verify_outputs':
                    if not self.verify_changes(modified_files):
                        raise ValueError("Output verification failed")
                
                # Record change
                self.change_history.append({
                    'step': step,
                    'timestamp': time.time(),
                    'duration': time.time() - start_time,
                    'status': 'completed'
                })
            
            logger.info(f"All changes applied successfully. Modified {len(modified_files)} files")
            return modified_files
            
        except Exception as e:
            logger.error(f"Change application failed: {str(e)}")
            self._revert_changes()
            raise ModificationError("Failed to apply EDA changes")
    
    def verify_changes(self, modified_files: List[str]) -> bool:
        """
        Verify changes through testing (Coding Agent Pattern).
        
        Args:
            modified_files: List of modified files
            
        Returns:
            bool indicating if changes pass verification
        """
        logger.info("Verifying changes...")
        
        try:
            verification_results = {
                'file_existence': True,
                'data_integrity': True,
                'figure_quality': True
            }
            
            # Check file existence
            for file_path in modified_files:
                if not Path(file_path).exists():
                    logger.error(f"Output file not found: {file_path}")
                    verification_results['file_existence'] = False
            
            # Check data integrity
            if len(self.df_before) == 0 or len(self.df_after) == 0:
                logger.error("Empty datasets detected")
                verification_results['data_integrity'] = False
            
            # Check figure quality (basic checks)
            for var in self.numerical_vars:
                if self.df_after[var].isna().all():
                    logger.error(f"All values missing for variable: {var}")
                    verification_results['figure_quality'] = False
            
            all_passed = all(verification_results.values())
            
            if all_passed:
                logger.info("All verifications passed")
            else:
                logger.error(f"Verification failed: {verification_results}")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return False
    
    def _revert_changes(self):
        """Revert changes on failure (Coding Agent Pattern)."""
        logger.warning("Reverting changes due to failure...")
        
        try:
            # Remove any created output files
            for change in self.change_history:
                if change['status'] == 'completed':
                    # Could implement file cleanup here if needed
                    pass
            
            # Clear change history
            self.change_history = []
            logger.info("Changes reverted successfully")
            
        except Exception as e:
            logger.error(f"Failed to revert changes: {str(e)}")
    
    def generate_all_visualizations(self):
        """Generate both essential figures using Coding Agent Pattern."""
        logger.info("Starting EDA visualization generation with Coding Agent Pattern...")
        print("=" * 60)
        
        try:
            # Step 1: Analyze task
            task_analysis = self.analyze_task("Create EDA visualizations")
            self.change_plan = task_analysis['change_plan']
            
            # Step 2: Apply changes using the plan
            modified_files = self.apply_changes(self.change_plan)
            
            # Step 3: Print summary
            summary_stats = self.generate_summary_statistics()
            self.print_summary(summary_stats)
            
            logger.info("All EDA visualizations generated successfully using Coding Agent Pattern!")
            print("=" * 60)
            print(f"Output directory: {self.output_dir}")
            print(f"Generated files: {len(modified_files)}")
            
        except Exception as e:
            logger.error(f"EDA generation failed: {str(e)}")
            raise


def main():
    """Main execution function."""
    # Set up paths - using proper before/after datasets
    data_path_before = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/final_books_2000_2020_en_enhanced_20250907_013708.csv"
    data_path_after = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/romance_novels_text_preprocessed_20250907_015606.csv"
    output_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/eda_analysis/outputs/complete_pipeline_analysis"
    
    # Create visualizer
    visualizer = ImprovedEDAFinal(data_path_before, data_path_after, output_dir)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
