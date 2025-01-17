from datetime import datetime
import pytest
from pydantic import ValidationError

from hydrocron_wrapper import (
    HydrocronConfig, TimeseriesRequest,
    FeatureType, OutputFormat, HydrocronField
)

def test_hydrocron_config():
    """Test HydrocronConfig validation"""
    # Test default config
    config = HydrocronConfig()
    assert config.api_key is None
    assert config.base_url == "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1"
    assert config.timeout == 30

    # Test custom config
    config = HydrocronConfig(
        api_key="test_key",
        base_url="https://test.url",
        timeout=60
    )
    assert config.api_key == "test_key"
    assert config.base_url == "https://test.url"
    assert config.timeout == 60

    # Test invalid timeout
    with pytest.raises(ValidationError):
        HydrocronConfig(timeout=-1)

def test_feature_type():
    """Test FeatureType enum"""
    assert FeatureType.REACH.value == "Reach"
    assert FeatureType.NODE.value == "Node"
    assert FeatureType.PRIOR_LAKE.value == "PriorLake"

    # Test conversion from string
    assert FeatureType("Reach") == FeatureType.REACH
    assert FeatureType("Node") == FeatureType.NODE
    assert FeatureType("PriorLake") == FeatureType.PRIOR_LAKE

    # Test invalid value
    with pytest.raises(ValueError):
        FeatureType("InvalidType")

def test_output_format():
    """Test OutputFormat enum"""
    assert OutputFormat.CSV.value == "csv"
    assert OutputFormat.GEOJSON.value == "geojson"

    # Test conversion from string
    assert OutputFormat("csv") == OutputFormat.CSV
    assert OutputFormat("geojson") == OutputFormat.GEOJSON

    # Test invalid value
    with pytest.raises(ValueError):
        OutputFormat("invalid")

def test_hydrocron_field():
    """Test HydrocronField enum"""
    assert HydrocronField.REACH_ID.value == "reach_id"
    assert HydrocronField.TIME.value == "time_str"
    assert HydrocronField.WSE.value == "wse"
    assert HydrocronField.WIDTH.value == "width"
    assert HydrocronField.SLOPE.value == "slope"
    assert HydrocronField.GEOMETRY.value == "geometry"

    # Test default fields
    default_fields = HydrocronField.default_fields()
    assert len(default_fields) == 4
    assert HydrocronField.REACH_ID.value in default_fields
    assert HydrocronField.TIME.value in default_fields
    assert HydrocronField.WSE.value in default_fields
    assert HydrocronField.WIDTH.value in default_fields

    # Test all fields
    all_fields = HydrocronField.get_all_fields()
    assert len(all_fields) == 6
    assert all(f.value in all_fields for f in HydrocronField)

def test_timeseries_request():
    """Test TimeseriesRequest validation"""
    # Test valid request
    request = TimeseriesRequest(
        feature=FeatureType.REACH,
        feature_id="12345678901",  # Valid REACH ID format
        start_time="2024-02-01T00:00:00Z",
        end_time="2024-02-08T00:00:00Z",
        fields=["reach_id", "time_str", "wse"]
    )
    assert request.feature == FeatureType.REACH
    assert request.feature_id == "12345678901"
    assert request.output == OutputFormat.CSV  # default value
    assert request.compact is None  # default value

    # Test invalid feature ID format
    with pytest.raises(ValidationError) as exc_info:
        TimeseriesRequest(
            feature=FeatureType.REACH,
            feature_id="invalid",  # Wrong format
            start_time="2024-02-01T00:00:00Z",
            end_time="2024-02-08T00:00:00Z",
            fields=["reach_id", "time_str", "wse"]
        )
    assert "Invalid Reach ID format" in str(exc_info.value)

    # Test invalid fields
    with pytest.raises(ValidationError) as exc_info:
        TimeseriesRequest(
            feature=FeatureType.REACH,
            feature_id="12345678901",
            start_time="2024-02-01T00:00:00Z",
            end_time="2024-02-08T00:00:00Z",
            fields=["invalid_field"]
        )
    assert "Invalid fields" in str(exc_info.value)

    # Test invalid time format
    with pytest.raises(ValidationError) as exc_info:
        TimeseriesRequest(
            feature=FeatureType.REACH,
            feature_id="12345678901",
            start_time="invalid_time",
            end_time="2024-02-08T00:00:00Z",
            fields=["reach_id", "time_str", "wse"]
        )
    assert "Time must be in ISO format" in str(exc_info.value)

    # Test invalid time range
    with pytest.raises(ValidationError) as exc_info:
        TimeseriesRequest(
            feature=FeatureType.REACH,
            feature_id="12345678901",
            start_time="2024-02-08T00:00:00Z",  # Later than end_time
            end_time="2024-02-01T00:00:00Z",
            fields=["reach_id", "time_str", "wse"]
        )
    assert "start_time must be before end_time" in str(exc_info.value)

    # Test node ID format
    request = TimeseriesRequest(
        feature=FeatureType.NODE,
        feature_id="12345678901234",  # Valid NODE ID format
        start_time="2024-02-01T00:00:00Z",
        end_time="2024-02-08T00:00:00Z",
        fields=["reach_id", "time_str", "wse"]
    )
    assert request.feature == FeatureType.NODE
    assert request.feature_id == "12345678901234"

    # Test prior lake ID format
    request = TimeseriesRequest(
        feature=FeatureType.PRIOR_LAKE,
        feature_id="1234567890",  # Valid PRIOR_LAKE ID format
        start_time="2024-02-01T00:00:00Z",
        end_time="2024-02-08T00:00:00Z",
        fields=["reach_id", "time_str", "wse"]
    )
    assert request.feature == FeatureType.PRIOR_LAKE
    assert request.feature_id == "1234567890" 