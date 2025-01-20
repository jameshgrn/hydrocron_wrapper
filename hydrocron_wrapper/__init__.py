"""
A lightweight Python wrapper for the Hydrocron API.
"""

from .client import (
    HydrocronClient, HydrocronError,
    HydrocronValidationError, HydrocronAPIError
)
from .types import (
    HydrocronConfig, FeatureType, OutputFormat, 
    HydrocronField, TimeseriesRequest,
    ResponseFormat, TimeseriesResponse,
    GeoJSONResponse, CSVResponse
)

__all__ = [
    'HydrocronClient',
    'HydrocronConfig',
    'FeatureType',
    'OutputFormat',
    'HydrocronField',
    'TimeseriesRequest',
    'ResponseFormat',
    'TimeseriesResponse',
    'GeoJSONResponse', 
    'CSVResponse',
    'HydrocronError',
    'HydrocronValidationError',
    'HydrocronAPIError'
]

__version__ = "0.1.0"
