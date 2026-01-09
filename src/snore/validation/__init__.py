"""
Validation module for comparing SNORE's programmatic detection with machine events.

This module provides batch validation and reporting capabilities.
"""

from snore.validation.batch import BatchValidator
from snore.validation.report import (
    ValidationReport,
    export_report_csv,
    export_report_json,
)

__all__ = [
    "BatchValidator",
    "ValidationReport",
    "export_report_csv",
    "export_report_json",
]
