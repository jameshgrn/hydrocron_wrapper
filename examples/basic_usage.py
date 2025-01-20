from datetime import datetime, timedelta
import os
from typing import Union, Dict, Any
import pandas as pd
from dotenv import load_dotenv

from hydrocron_wrapper.client import HydrocronClient, ResponseFormat
from hydrocron_wrapper.types import HydrocronConfig, FeatureType, OutputFormat

def main():
    """Example usage of the Hydrocron API client"""
    # Load environment variables
    load_dotenv()
    
    # Create client with optional API key
    config = HydrocronConfig(api_key=os.getenv('HYDROCRON_API_KEY'))
    client = HydrocronClient(config)
    
    # Example reach ID from documentation
    reach_id = "63470800171"
    
    # Get data for last 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Fetching data for reach {reach_id}")
    print(f"Time range: {start_time} to {end_time}")
    
    # Get time series data
    result = client.get_timeseries(
        feature=FeatureType.REACH,
        feature_id=reach_id,
        start_time=start_time,
        end_time=end_time,
        output_format=OutputFormat.CSV
    )
    
    if isinstance(result, pd.DataFrame):
        if not result.empty:
            print("\nData retrieved successfully!")
            print("\nFirst few rows:")
            print(result.head())
            print(f"\nTotal records: {len(result)}")
        else:
            print("\nNo data found for the specified parameters")
    else:
        print("\nReceived GeoJSON response")
        print(result)

if __name__ == "__main__":
    main() 