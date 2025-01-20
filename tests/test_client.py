from datetime import datetime, timezone
import json
from typing import Dict, Any

import pytest
import responses
import pandas as pd
from requests.exceptions import RequestException
import geopandas as gpd
from shapely.geometry import Point, LineString
from pydantic import ValidationError

from hydrocron_wrapper import (
    HydrocronClient, HydrocronConfig, 
    FeatureType, OutputFormat, HydrocronError,
    HydrocronValidationError, HydrocronAPIError,
    ResponseFormat
)

@pytest.fixture
def client():
    """Create a test client with default config"""
    return HydrocronClient()

@pytest.fixture
def mock_csv_response() -> Dict[str, Any]:
    """Create a mock CSV response"""
    return {
        "status": "200 OK",
        "time": 844.614,
        "hits": 2,
        "results": {
            "csv": "reach_id,time_str,wse,width,wse_units,width_units\n63470800171,2024-02-01T02:26:50Z,3386.9332,383.19271200000003,m,m\n63470800171,2024-02-08T13:48:41Z,1453.4136,501.616464,m,m\n",
            "geojson": {}
        }
    }

@pytest.fixture
def mock_geojson_response() -> Dict[str, Any]:
    """Create a mock GeoJSON response"""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "0",
                "type": "Feature",
                "properties": {
                    "reach_id": "63470800171",
                    "time_str": "2024-02-01T02:26:50Z",
                    "wse": "3386.9332",
                    "wse_units": "m"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-45.845445, -16.166559],
                        [-45.845500, -16.166600]
                    ]
                }
            }
        ]
    }

@pytest.fixture
def mock_client():
    """Create a test client with mocked requests"""
    client = HydrocronClient()
    # Mock the _make_request method
    client._make_request = lambda *args, **kwargs: {}
    return client

def test_client_initialization():
    """Test client initialization with different configs"""
    # Test default config
    client = HydrocronClient()
    assert client.config.api_key is None
    assert client.config.base_url == "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1"
    assert client.config.timeout == 30

    # Test custom config
    config = HydrocronConfig(
        api_key="test_key",
        base_url="https://test.url",
        timeout=60
    )
    client = HydrocronClient(config)
    assert client.config.api_key == "test_key"
    assert client.config.base_url == "https://test.url"
    assert client.config.timeout == 60

    # Test invalid config
    with pytest.raises(ValidationError):
        HydrocronConfig(timeout=-1)

@responses.activate
def test_get_timeseries_csv(client, mock_csv_response):
    """Test getting time series data in CSV format"""
    # Setup mock response
    responses.add(
        responses.GET,
        f"{client.config.base_url}/timeseries",
        json=mock_csv_response,
        status=200
    )

    # Make request
    result = client.get_timeseries(
        feature=FeatureType.REACH,
        feature_id="63470800171",
        start_time=datetime(2024, 2, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 2, 8, tzinfo=timezone.utc)
    )

    # Verify response
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ['reach_id', 'time_str', 'wse', 'width', 'wse_units', 'width_units']
    assert result.iloc[0]['reach_id'] == "63470800171"

@responses.activate
def test_get_timeseries_geojson(client, mock_geojson_response):
    """Test getting time series data in GeoJSON format"""
    # Print the mock response for debugging
    print(f"\nMock GeoJSON response: {mock_geojson_response}")
    
    # Setup mock response with GeoJSON directly in results
    mock_full_response = {
        'status': '200 OK',
        'time': 123.45,
        'hits': 1,
        'results': {
            'geojson': mock_geojson_response
        }
    }
    print(f"\nFull mock response: {mock_full_response}")
    
    responses.add(
        responses.GET,
        f"{client.config.base_url}/timeseries",
        json=mock_full_response,
        status=200
    )

    # Make request
    result = client.get_timeseries(
        feature=FeatureType.REACH,
        feature_id="63470800171",
        start_time=datetime(2024, 2, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 2, 8, tzinfo=timezone.utc),
        output_format=OutputFormat.GEOJSON
    )

    # Print result for debugging
    print(f"\nParsed result: {result}")

    # Verify response
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1
    assert result.iloc[0]['reach_id'] == "63470800171"
    assert 'wse' in result.columns
    assert 'time_str' in result.columns
    assert isinstance(result.geometry.iloc[0], LineString)

@responses.activate
def test_get_timeseries_error_handling(client):
    """Test error handling in get_timeseries"""
    # Test 400 error
    responses.add(
        responses.GET,
        f"{client.config.base_url}/timeseries",
        json={"error": "Bad Request"},
        status=400
    )

    with pytest.raises(HydrocronAPIError):
        client.get_timeseries(
            feature=FeatureType.REACH,
            feature_id="invalid_id",
            start_time=datetime(2024, 2, 1),
            end_time=datetime(2024, 2, 8)
        )

    # Test network error
    responses.remove(responses.GET, f"{client.config.base_url}/timeseries")
    responses.add(
        responses.GET,
        f"{client.config.base_url}/timeseries",
        body=RequestException()
    )

    with pytest.raises(HydrocronAPIError):
        client.get_timeseries(
            feature=FeatureType.REACH,
            feature_id="63470800171",
            start_time=datetime(2024, 2, 1),
            end_time=datetime(2024, 2, 8)
        )

def test_invalid_feature_type(client):
    """Test handling of invalid feature type"""
    with pytest.raises(HydrocronValidationError):
        client.get_timeseries(
            feature="InvalidType",
            feature_id="63470800171",
            start_time=datetime(2024, 2, 1),
            end_time=datetime(2024, 2, 8)
        )

def test_invalid_time_format(client):
    """Test handling of invalid time format"""
    with pytest.raises(HydrocronValidationError):
        client.get_timeseries(
            feature=FeatureType.REACH,
            feature_id="63470800171",
            start_time="invalid_time",
            end_time="invalid_time"
        ) 

def test_csv_response_parsing(mock_client):
    """Test parsing CSV response"""
    mock_response = {
        'status': '200 OK',
        'time': 123.45,
        'hits': 2,
        'results': {
            'csv': 'reach_id,time_str,wse\n123,2024-01-01,100.5\n123,2024-01-02,101.2'
        }
    }
    
    df = mock_client._parse_csv_response(mock_response)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ['reach_id', 'time_str', 'wse']

def test_geojson_response_parsing(mock_client):
    """Test parsing GeoJSON response"""
    mock_response = {
        'status': '200 OK',
        'time': 123.45,
        'hits': 1,
        'results': {
            'geojson': {
                'type': 'FeatureCollection',
                'features': [{
                    'type': 'Feature',
                    'properties': {
                        'reach_id': '123',
                        'wse': 100.5
                    },
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [0, 0]
                    }
                }]
            }
        }
    }
    
    gdf = mock_client._parse_geojson_response(mock_response)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.iloc[0]['reach_id'] == '123'
    assert isinstance(gdf.geometry.iloc[0], Point) 