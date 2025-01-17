from datetime import datetime
from typing import Optional, Union, Dict, Any
import logging
from io import StringIO
from urllib.parse import quote_plus

import requests
import pandas as pd
from pydantic import ValidationError

from .types import (
    HydrocronConfig, TimeseriesRequest, 
    FeatureType, OutputFormat, HydrocronField
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HydrocronError(Exception):
    """Base exception for Hydrocron API errors"""
    pass

class HydrocronValidationError(HydrocronError):
    """Raised when request validation fails"""
    pass

class HydrocronAPIError(HydrocronError):
    """Raised when API request fails"""
    pass

class HydrocronClient:
    """Client for interacting with the Hydrocron API"""
    
    def __init__(self, config: Optional[HydrocronConfig] = None):
        """Initialize client with optional configuration"""
        try:
            self.config = config or HydrocronConfig()
        except ValidationError as e:
            raise HydrocronValidationError(f"Invalid configuration: {e}")
        
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers including optional API key"""
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["x-hydrocron-key"] = self.config.api_key
        return headers
    
    @staticmethod
    def _to_iso_time(time: Union[str, datetime]) -> str:
        """Convert time to ISO 8601 format"""
        if isinstance(time, datetime):
            return time.isoformat()
        return time
    
    def get_timeseries(
        self,
        feature: Union[FeatureType, str],
        feature_id: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        fields: Optional[list[str]] = None,
        output_format: OutputFormat = OutputFormat.CSV,
        include_geometry: bool = False
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Get time series data for a specific feature
        
        Args:
            feature: Feature type (Reach, Node, or PriorLake)
            feature_id: ID of the feature
            start_time: Start time for query
            end_time: End time for query
            fields: List of fields to return (default: basic fields)
            output_format: Format of the response (CSV or GeoJSON)
            include_geometry: Whether to include geometry in response
            
        Returns:
            DataFrame for CSV output, dict for GeoJSON output
            
        Raises:
            HydrocronValidationError: If request parameters are invalid
            HydrocronAPIError: If API request fails
        """
        try:
            # Convert feature type if string
            if isinstance(feature, str):
                feature = FeatureType(feature)
                
            # Use default fields if none provided
            fields = fields or HydrocronField.default_fields()
            if include_geometry and HydrocronField.GEOMETRY.value not in fields:
                fields.append(HydrocronField.GEOMETRY.value)
                
            # Build and validate request
            request = TimeseriesRequest(
                feature=feature,
                feature_id=feature_id,
                start_time=quote_plus(self._to_iso_time(start_time)),
                end_time=quote_plus(self._to_iso_time(end_time)),
                fields=fields,
                output=output_format
            )
            
        except (ValidationError, ValueError) as e:
            raise HydrocronValidationError(f"Invalid request parameters: {e}")
        
        # Make request
        try:
            response = requests.get(
                f"{self.config.base_url}/timeseries",
                headers=self._build_headers(),
                params=request.model_dump(),
                timeout=self.config.timeout
            )
            
            # Handle errors
            if response.status_code == 400:
                error_msg = f"API Error: {response.text}"
                logger.warning(error_msg)
                raise HydrocronAPIError(error_msg)
                
            response.raise_for_status()
            
            # Parse response
            if output_format == OutputFormat.CSV:
                json_data = response.json()
                csv_data = json_data.get('results', {}).get('csv', '')
                if not csv_data:
                    logger.warning(f"No data found for feature {feature_id}")
                    return pd.DataFrame()
                return pd.read_csv(StringIO(csv_data))
            else:
                return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            logger.error(error_msg)
            raise HydrocronAPIError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            raise HydrocronError(error_msg) 