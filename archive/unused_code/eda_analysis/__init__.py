"""
Exploratory Data Analysis Module for Romance Novel NLP Research
Handles EDA scripts and analysis tools for romance novel datasets.

Current Workflow: EDA analysis is performed in notebooks/02_final_dataset_eda_and_cleaning.ipynb
Python version available at: 02_final_dataset_eda_and_cleaning.py
Legacy code has been moved to archive/unused_code/eda_analysis_old/
"""

# Note: EDA analysis is currently performed in Jupyter notebooks
# Legacy standalone scripts have been archived
# Python version available for automated execution

__all__ = ['run_eda_analysis']

def run_eda_analysis():
    """Run the complete EDA analysis from the Python script"""
    from . import final_dataset_eda_and_cleaning
    final_dataset_eda_and_cleaning.main()
