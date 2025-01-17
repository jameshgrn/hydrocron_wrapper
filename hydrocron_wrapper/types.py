from enum import Enum
from typing import Optional, List, Set, Any, TypeVar
from datetime import datetime
import re
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo

# Type variable for model validators
ModelT = TypeVar('ModelT', bound=BaseModel)

class FeatureType(str, Enum):
    """Valid feature types for Hydrocron API"""
    REACH = "Reach"
    NODE = "Node"
    PRIOR_LAKE = "PriorLake"

    @classmethod
    def get_id_pattern(cls, feature_type: "FeatureType") -> str:
        """Get regex pattern for feature ID validation"""
        patterns = {
            cls.REACH: r"^\d{11}$",  # CBBBBBRRRRT format
            cls.NODE: r"^\d{14}$",   # CBBBBBRRRRNNNT format
            cls.PRIOR_LAKE: r"^\d{10}$"  # CBBNNNNNNT format
        }
        return patterns[feature_type]

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
    
    @classmethod
    def get_all_fields(cls) -> Set[str]:
        """Get all valid field names"""
        return {field.value for field in cls}

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
    @classmethod
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

    @field_validator('feature_id')
    @classmethod
    def validate_feature_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate feature ID format based on feature type"""
        feature_type = info.data.get('feature')
        if not feature_type:
            raise ValueError("Feature type must be specified before feature ID")
            
        pattern = FeatureType.get_id_pattern(feature_type)
        if not re.match(pattern, v):
            raise ValueError(
                f"Invalid {feature_type.value} ID format. "
                f"Must match pattern: {pattern}"
            )
        return v

    @field_validator('fields')
    @classmethod
    def validate_fields(cls, v: List[str]) -> List[str]:
        """Validate fields list is not empty and contains valid fields"""
        if not v:
            raise ValueError("At least one field must be specified")
            
        valid_fields = HydrocronField.get_all_fields()
        invalid_fields = [f for f in v if f not in valid_fields]
        if invalid_fields:
            raise ValueError(
                f"Invalid fields: {invalid_fields}. "
                f"Valid fields are: {sorted(valid_fields)}"
            )
        return v

    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate time string format"""
        try:
            # Try to parse as ISO format
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("Time must be in ISO format (YYYY-MM-DDTHH:MM:SSZ)")
            
    @model_validator(mode='after')
    @classmethod
    def validate_time_range(cls, data: "TimeseriesRequest") -> "TimeseriesRequest":
        """Validate that start_time is before end_time"""
        start = datetime.fromisoformat(data.start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(data.end_time.replace('Z', '+00:00'))
        
        if start >= end:
            raise ValueError("start_time must be before end_time")
            
        return data 