from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class FeatureType(str, Enum):
    """Valid feature types for Hydrocron API"""
    REACH = "Reach"
    NODE = "Node"
    PRIOR_LAKE = "PriorLake"

class OutputFormat(str, Enum):
    """Valid output formats for Hydrocron API"""
    CSV = "csv"
    GEOJSON = "geojson"

class HydrocronField(str, Enum):
    """Common fields available in the Hydrocron API"""
    REACH_ID = "reach_id"
    TIME = "time_str"
    WSE = "wse"
    WIDTH = "width"
    SLOPE = "slope"
    GEOMETRY = "geometry"
    
    @classmethod
    def default_fields(cls) -> List[str]:
        """Returns commonly used fields"""
        return [cls.REACH_ID.value, cls.TIME.value, cls.WSE.value, cls.WIDTH.value]

class HydrocronConfig(BaseModel):
    """Configuration for Hydrocron API client"""
    api_key: Optional[str] = Field(default=None, description="Optional API key for authentication")
    base_url: str = Field(
        default="https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1",
        description="Base URL for the Hydrocron API"
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
        gt=0
    )

    @field_validator('timeout')
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive"""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

class TimeseriesRequest(BaseModel):
    """Parameters for a timeseries request"""
    feature: FeatureType
    feature_id: str = Field(..., min_length=1)
    start_time: str
    end_time: str
    fields: List[str] = Field(..., min_length=1)
    output: OutputFormat = OutputFormat.CSV
    compact: Optional[bool] = None

    @field_validator('fields')
    def validate_fields(cls, v: List[str]) -> List[str]:
        """Validate fields list is not empty"""
        if not v:
            raise ValueError("At least one field must be specified")
        return v

    @field_validator('start_time', 'end_time')
    def validate_time_format(cls, v: str) -> str:
        """Validate time string format"""
        try:
            # Try to parse as ISO format
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("Time must be in ISO format (YYYY-MM-DDTHH:MM:SSZ)") 