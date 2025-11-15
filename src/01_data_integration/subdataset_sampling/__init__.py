"""
Subdataset Sampling Module

This module contains functionality for creating representative sub-datasets
from the main Goodreads corpus with balanced representation across
popularity tiers and preserved demographic characteristics.
"""

from .create_subdataset_6000 import create_subdataset_6000

__all__ = ['create_subdataset_6000']

