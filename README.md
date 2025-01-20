# Hydrocron Wrapper

A Python wrapper for the Hydrocron API, providing easy access to river and lake water surface elevation data.

## Installation

```bash
pip install hydrocron-wrapper
```

Or with Poetry:

```bash
poetry add hydrocron-wrapper
```

## Basic Usage

```python
from hydrocron_wrapper import HydrocronClient, FeatureType
from datetime import datetime, timezone

# Initialize client
client = HydrocronClient()

# Get time series data
df = client.get_timeseries(
    feature=FeatureType.REACH,
    feature_id="63470800171",
    start_time=datetime(2024, 2, 1, tzinfo=timezone.utc),
    end_time=datetime(2024, 2, 8, tzinfo=timezone.utc)
)

# Analyze and plot data
stats, fig = client.analyze_and_plot(
    feature_type=FeatureType.REACH,
    feature_id="63470800171",
    start_time=datetime(2024, 2, 1, tzinfo=timezone.utc),
    end_time=datetime(2024, 2, 8, tzinfo=timezone.utc),
    quality_threshold=85,
    show_uncertainty=True
)

# Display plot
fig.show()
```

## Features

- Time series data retrieval
- Data analysis and statistics
- Interactive plotting
- River network visualization
- Response caching
- Rate limiting
- Error handling

## Advanced Usage

See [examples](examples/) directory for more detailed usage examples:
- Basic data retrieval
- Time series analysis
- River network plotting
- Error handling

## API Documentation

For detailed API documentation, see [api_documentation.md](api_documentation.md).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests with `poetry run pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details
