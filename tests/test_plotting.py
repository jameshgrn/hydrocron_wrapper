import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import plotly.graph_objects as go

from hydrocron_wrapper import HydrocronClient
from hydrocron_wrapper.utils import get_planet_basemap_url

@pytest.fixture
def mock_reaches_gdf():
    """Create mock reaches GeoDataFrame"""
    return gpd.GeoDataFrame(
        {
            'reach_id': ['123', '456'],
            'geometry': [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)])
            ]
        }
    )

@pytest.fixture
def mock_nodes_gdf():
    """Create mock nodes GeoDataFrame"""
    return gpd.GeoDataFrame(
        {
            'node_id': ['1', '2'],
            'dist_out': [100.0, 200.0],
            'geometry': [Point(0, 0), Point(1, 1)]
        }
    )

@pytest.fixture
def mock_timeseries_df():
    """Create mock timeseries DataFrame"""
    return pd.DataFrame({
        'time_str': ['2024-01-01', '2024-01-02'],
        'wse': [100.0, 101.0],
        'width': [50.0, 51.0],
        'quality': [80, 90]
    })

def test_plot_river_network(mock_reaches_gdf, mock_nodes_gdf):
    """Test river network plotting"""
    client = HydrocronClient()
    fig = client.plot_river_network(
        reaches_gdf=mock_reaches_gdf,
        nodes_gdf=mock_nodes_gdf
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # 2 reaches + 1 nodes trace
    
    # Test with Planet basemap
    fig = client.plot_river_network(
        reaches_gdf=mock_reaches_gdf,
        nodes_gdf=mock_nodes_gdf,
        use_planet=True,
        planet_key="test_key",
        planet_quarter="2024Q1"
    )
    
    assert isinstance(fig, go.Figure)
    assert 'mapbox' in fig.layout
    assert 'layers' in fig.layout.mapbox

def test_plot_timeseries(mock_timeseries_df):
    """Test timeseries plotting"""
    client = HydrocronClient()
    fig = client.plot_timeseries(
        df=mock_timeseries_df,
        quality_threshold=85,
        show_uncertainty=True,
        feature_type="Reach",
        feature_id="123"
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # At least WSE and width traces
    
    # Test without uncertainty
    fig = client.plot_timeseries(
        df=mock_timeseries_df,
        show_uncertainty=False
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Just WSE and width traces

def test_plot_river_network_edge_cases(mock_reaches_gdf, mock_nodes_gdf):
    """Test edge cases in river network plotting"""
    client = HydrocronClient()
    
    # Test empty GeoDataFrames
    fig = client.plot_river_network(
        reaches_gdf=gpd.GeoDataFrame(),
        nodes_gdf=gpd.GeoDataFrame()
    )
    assert isinstance(fig, go.Figure)
    
    # Test invalid geometries
    bad_reaches = mock_reaches_gdf.copy()
    bad_reaches.loc[0, 'geometry'] = None
    fig = client.plot_river_network(
        reaches_gdf=bad_reaches,
        nodes_gdf=mock_nodes_gdf
    )
    assert isinstance(fig, go.Figure)

def test_plot_timeseries_edge_cases(mock_timeseries_df):
    """Test edge cases in timeseries plotting"""
    client = HydrocronClient()
    
    # Test empty DataFrame
    fig = client.plot_timeseries(df=pd.DataFrame())
    assert isinstance(fig, go.Figure)
    
    # Test missing columns
    bad_df = mock_timeseries_df.drop(columns=['quality'])
    fig = client.plot_timeseries(
        df=bad_df,
        quality_threshold=85
    )
    assert isinstance(fig, go.Figure)
    
    # Test all null values
    null_df = mock_timeseries_df.copy()
    null_df['wse'] = None
    fig = client.plot_timeseries(df=null_df)
    assert isinstance(fig, go.Figure) 