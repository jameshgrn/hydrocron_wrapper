"""Example of time series analysis and visualization"""
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
from example_implementation import HydrocronAPI, HydrocronConfig, FeatureType
import matplotlib.pyplot as plt

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize API client
    api = HydrocronAPI()
    
    # Use older date range
    start_time = datetime(2022, 2, 1, tzinfo=timezone.utc)
    end_time = datetime(2024, 3, 1, tzinfo=timezone.utc)
    
    print("\n=== Analyzing Reach Time Series ===")
    print(f"Reach ID: 63470800171")
    print(f"Time range: {start_time.strftime('%Y-%m-%dT%H:%M:%SZ')} to {end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    
    try:
        # Get time series data directly using the working implementation
        fields = ['reach_id', 'time_str', 'wse', 'width', 'wse_u', 'width_u']
        df = api.get_reach_timeseries(
            reach_id="63470800171",
            start_time=start_time,
            end_time=end_time,
            fields=fields
        )
        
        if df.empty:
            print("No data returned from API")
            return
            
        # Print first few rows
        print("\nFirst few rows of data:")
        print(df.head())
        
        # Create plot using matplotlib as in the example
        print("\nCreating plot...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot water surface elevation
        ax1.plot(df.time_str, df.wse, marker='o')
        ax1.set_ylabel('Water Surface Elevation (m)')
        ax1.set_xlabel('Date')
        ax1.grid(True)
        ax1.set_title('Water Surface Elevation')
        
        # Plot width
        ax2.plot(df.time_str, df.width, marker='o', color='green')
        ax2.set_ylabel('River Width (m)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        ax2.set_title('River Width')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 