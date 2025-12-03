"""
Data Pipelines Module.

Provides automated SQL data pipelines for ingesting, transforming,
and preparing healthcare data for analysis and forecasting.
"""

from demand_capacity_platform.data_pipelines.data_loader import DataLoader
from demand_capacity_platform.data_pipelines.sql_pipeline import SQLDataPipeline

__all__ = ["SQLDataPipeline", "DataLoader"]
