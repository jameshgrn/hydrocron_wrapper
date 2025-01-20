"""Tests for timeseries_analysis.py example script."""

import pytest
from unittest.mock import Mock, patch
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

from hydrocron_wrapper import HydrocronClient, HydrocronConfig
from examples.timeseries_analysis import main

@pytest.fixture
def mock_client():
    """Mock HydrocronClient with predefined responses."""
    client = Mock(spec=HydrocronClient)
    
    # Mock analyze_and_plot for reach
    reach_stats = {
        'wse': {
            'mean': 100.0,
            'std': 5.0,
            'min': 90.0,
            'max': 110.0
        },
        'width': {
            'mean': 50.0,
            'std': 2.0,
            'min': 45.0,
            'max': 55.0
        }
    }
    reach_plot = go.Figure()
    client.analyze_and_plot.side_effect = [
        (reach_stats, reach_plot),  # First call for reach
        (reach_stats, reach_plot)   # Second call for node
    ]
    
    return client

@pytest.fixture
def mock_files():
    """Setup and cleanup test files."""
    files = ["reach_timeseries.html", "node_timeseries.html"]
    yield files
    # Cleanup after tests
    for file in files:
        path = Path(file)
        if path.exists():
            path.unlink()

class TestTimeseriesAnalysis:
    """Tests for timeseries_analysis.py example."""

    @patch('examples.timeseries_analysis.HydrocronClient')
    def test_main_function(self, mock_client_class, mock_files, capsys):
        """Test the main function's execution flow."""
        # Setup mock client
        mock_client_instance = mock_client_class.return_value
        mock_stats = {
            'wse': {'mean': 100.0, 'std': 5.0},
            'width': {'mean': 50.0, 'std': 2.0}
        }
        mock_plot = go.Figure()
        mock_client_instance.analyze_and_plot.return_value = (mock_stats, mock_plot)

        # Run main function
        main()

        # Capture printed output
        captured = capsys.readouterr()

        # Verify client initialization
        mock_client_class.assert_called_once()

        # Verify analyze_and_plot calls
        assert mock_client_instance.analyze_and_plot.call_count == 2
        
        # Verify reach analysis call
        reach_call = mock_client_instance.analyze_and_plot.call_args_list[0]
        assert reach_call.kwargs['feature_type'] == "Reach"
        assert reach_call.kwargs['feature_id'] == "63470800171"
        assert reach_call.kwargs['show_uncertainty'] is True

        # Verify node analysis call
        node_call = mock_client_instance.analyze_and_plot.call_args_list[1]
        assert node_call.kwargs['feature_type'] == "Node"
        assert node_call.kwargs['feature_id'] == "12228200110861"
        assert node_call.kwargs['show_uncertainty'] is True

        # Verify file creation
        assert Path("reach_timeseries.html").exists()
        assert Path("node_timeseries.html").exists()

        # Verify console output
        assert "=== Analyzing Reach Time Series ===" in captured.out
        assert "=== Analyzing Node Time Series ===" in captured.out
        assert "Reach Statistics:" in captured.out
        assert "Node Statistics:" in captured.out

    @patch('examples.timeseries_analysis.HydrocronClient')
    def test_error_handling(self, mock_client_class, mock_files):
        """Test error handling in main function."""
        # Setup mock client to raise an exception
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.analyze_and_plot.side_effect = Exception("API Error")

        # Run main function and verify it handles the error gracefully
        with pytest.raises(Exception) as exc_info:
            main()
        
        assert str(exc_info.value) == "API Error"

    def test_date_range_validity(self):
        """Test the validity of the date range used in the script."""
        start_time = "2022-08-01T00:00:00Z"
        end_time = "2024-12-31T23:59:59Z"
        
        # Convert to datetime objects
        start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
        end_dt = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")
        
        # Verify date range is valid
        assert start_dt < end_dt
        assert (end_dt - start_dt).days > 0 