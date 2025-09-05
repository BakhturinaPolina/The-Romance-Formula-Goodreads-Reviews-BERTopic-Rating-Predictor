"""
Source code for Romance Novel NLP Research Project
"""

__version__ = "1.0.0"

# Import working modules only
from . import csv_building
from . import data_quality
from . import nlp_preprocessing

__all__ = [
    'csv_building',
    'data_quality',
    'nlp_preprocessing'
]
