from __future__ import annotations
import dis

import requests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import folium
from typing import (
    Tuple, Dict, List, Optional, Union, Any, 
    Literal, TypedDict, cast, overload
)
import xarray as xr
from dataclasses import dataclass
from enum import Enum
import logging
from urllib.parse import quote_plus
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os
import geopandas as gpd
import uuid
from shapely.geometry import LineString, Point
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
import tqdm
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.animation as animation
import time
from pathlib import Path
import json
import traceback

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)



@dataclass
class HydrocronConfig:
    """Configuration for Hydrocron API"""
    def __init__(self):
        """Initialize with default values"""
        self.api_key = os.getenv('HYDROCRON_API_KEY')
        self.base_url = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1"
        self.timeout = 30
        self.default_quality_threshold = 1
        self.db_params = None

class FeatureType(str, Enum):
    """Valid feature types for Hydrocron API"""
    REACH = "Reach"
    NODE = "Node" 
    PRIOR_LAKE = "PriorLake"


def create_db_engine(db_params: Dict[str, Optional[str]]):
    """Create SQLAlchemy engine from database parameters"""
    # Filter out None values before creating connection string
    valid_params = {k: v for k, v in db_params.items() if v is not None}
    return create_engine(
        f"postgresql://{valid_params['user']}:{valid_params['password']}@"
        f"{valid_params['host']}:{valid_params['port']}/{valid_params['dbname']}"
    )

def get_dataframe(engine, query: str) -> gpd.GeoDataFrame:
    """Execute SQL query and return results as GeoDataFrame"""
    try:
        gdf = gpd.read_postgis(query, engine, geom_col='geometry')
        if isinstance(gdf, gpd.GeoDataFrame):
            return gdf
        # If it's a generator, materialize it
        return gpd.GeoDataFrame(gdf)
    except Exception as e:
        logger.error(f"Database query failed: {str(e)}")
        raise

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
        return [cls.REACH_ID.value, cls.TIME.value, 
                cls.WSE.value, cls.WIDTH.value]

class HydrocronError(Exception):
    """Base exception for Hydrocron API errors"""
    pass

class HydrocronAPI:
    """Enhanced interface for the PO.DAAC Hydrocron API for SWOT river data"""
    
    def __init__(self, config: Optional[HydrocronConfig] = None):
        """Initialize with optional configuration"""
        load_dotenv()
        self.config = config or HydrocronConfig()
        self.planet_api_key = os.getenv('PLANET_API_KEY')
        
        # Set up database connection if parameters provided
        if not self.config.db_params:
            self.config.db_params = {
                'dbname': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT')
            }
        
        if all(self.config.db_params.values()):
            self.engine = create_db_engine(self.config.db_params)
        else:
            self.engine = None
            logger.warning("Database configuration not provided or incomplete")

        # Initialize headers
        self.headers = {
            'Accept': 'application/json',  # API returns JSON wrapper around CSV
            'Content-Type': 'application/json'
        }
        
        # Add authorization if API key is provided
        if self.config.api_key:
            self.headers['Authorization'] = f'Bearer {self.config.api_key}'

    def clean_reach_id(self, reach_id: Union[str, float, int]) -> str:
        """Convert reach ID to proper integer string format"""
        return str(int(float(str(reach_id))))

    def traverse_river_db(
        self,
        start_reach_id: str,
        direction: Literal['up', 'down'] = 'down',
        max_reaches: int = 5000,  # Increased default
        min_facc: Optional[float] = 1e3  # Lowered default threshold
    ) -> List[str]:
        """Traverse the river network using database queries"""
        if not self.engine:
            raise HydrocronError("Database connection not configured")
        
        start_reach_id = self.clean_reach_id(start_reach_id)
        logger.info(f"Starting {direction}stream traversal from reach {start_reach_id}")
        
        # First, verify the start reach exists
        verify_query = f"""
        SELECT reach_id, rch_id_up, rch_id_dn, facc 
        FROM sword_reaches_v16 
        WHERE reach_id = '{start_reach_id}'
        """
        
        start_df = pd.read_sql(verify_query, self.engine)
        if start_df.empty:
            logger.error(f"Start reach {start_reach_id} not found in database")
            return []
        
        current_facc = float(start_df.iloc[0]['facc'])
        logger.info(f"Start reach FACC: {current_facc}")
        
        visited = set()
        reach_ids = []
        queue = [(start_reach_id, current_facc)]
        
        while queue and len(reach_ids) < max_reaches:
            current_reach_id, current_facc = queue.pop(0)
            
            if current_reach_id in visited:
                continue
                
            visited.add(current_reach_id)
            
            # More lenient flow accumulation check
            if min_facc is None or current_facc >= min_facc:
                reach_ids.append(current_reach_id)
                logger.debug(f"Added reach {current_reach_id} (FACC: {current_facc})")
                
                # Get next reaches with a more expansive query
                next_reaches_field = 'rch_id_up' if direction == 'up' else 'rch_id_dn'
                query = f"""
                SELECT reach_id, {next_reaches_field}, facc 
                FROM sword_reaches_v16 
                WHERE reach_id = '{current_reach_id}'
                """
                
                try:
                    df = pd.read_sql(query, self.engine)
                    if not df.empty:
                        next_reaches = str(df.iloc[0][next_reaches_field]).split()
                        next_reaches = [r for r in next_reaches if r and r.strip()]
                        
                        if next_reaches:
                            logger.debug(f"Found {len(next_reaches)} {direction}stream reaches: {next_reaches}")
                            
                            # Get flow accumulation for next reaches
                            next_reaches_str = ','.join([f"'{self.clean_reach_id(r)}'" for r in next_reaches])
                            facc_query = f"""
                            SELECT reach_id, facc 
                            FROM sword_reaches_v16 
                            WHERE reach_id IN ({next_reaches_str})
                            ORDER BY facc DESC
                            """
                            
                            next_df = pd.read_sql(facc_query, self.engine)
                            
                            for _, row in next_df.iterrows():
                                next_id = self.clean_reach_id(row['reach_id'])
                                next_facc = float(row['facc'])
                                
                                # More lenient conditions for adding reaches
                                if direction == 'down':
                                    # Allow some increase in FACC for downstream
                                    if next_facc <= current_facc * 1.2:  # 20% tolerance
                                        queue.append((next_id, next_facc))
                                        logger.debug(f"Queued downstream reach {next_id} (FACC: {next_facc})")
                                else:
                                    # For upstream, explore all branches
                                    queue.append((next_id, next_facc))
                                    logger.debug(f"Queued upstream reach {next_id} (FACC: {next_facc})")
                        
                except Exception as e:
                    logger.error(f"Error processing reach {current_reach_id}: {str(e)}")
                    continue
        
        logger.info(f"{direction.capitalize()} traversal complete. Found {len(reach_ids)} reaches")
        return reach_ids

    def get_reach_network(
        self,
        start_reach_id: str,
        start_dist_out: Optional[float] = None,
        end_dist_out: Optional[float] = None,
        max_reaches: int = 500,
        min_facc: Optional[float] = None
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Get the river network and nodes for a given reach"""
        if not self.engine:
            raise HydrocronError("Database connection not configured")
        
        # First, verify the start reach exists and get its data
        verify_query = f"""
        SELECT reach_id, rch_id_up, rch_id_dn, facc 
        FROM sword_reaches_v16 
        WHERE reach_id = '{self.clean_reach_id(start_reach_id)}'
        """
        
        start_df = pd.read_sql(verify_query, self.engine)
        if start_df.empty:
            raise HydrocronError(f"Start reach {start_reach_id} not found in database")
        
        logger.info(f"Found start reach. FACC: {start_df.iloc[0]['facc']}")
        logger.info(f"Upstream reaches: {start_df.iloc[0]['rch_id_up']}")
        logger.info(f"Downstream reaches: {start_df.iloc[0]['rch_id_dn']}")
        
        # If we found the start reach, at minimum include it in our results
        reach_ids = [self.clean_reach_id(start_reach_id)]
        
        # Get additional reaches through traversal
        downstream_reaches = self.traverse_river_db(start_reach_id, 'down', max_reaches=max_reaches, min_facc=min_facc)
        upstream_reaches = self.traverse_river_db(start_reach_id, 'up', max_reaches=max_reaches, min_facc=min_facc)
        
        # Add traversal results to our list
        reach_ids.extend(downstream_reaches)
        reach_ids.extend(upstream_reaches)
        
        # Remove duplicates while preserving order
        reach_ids = list(dict.fromkeys(reach_ids))
        
        logger.info(f"Total reaches found: {len(reach_ids)}")
        logger.info(f"Sample reaches: {reach_ids[:5]}")
        
        # Create the reaches query
        reach_ids_str = ','.join([f"'{r}'" for r in reach_ids])
        reaches_query = f"""
        SELECT * FROM sword_reaches_v16 
        WHERE reach_id IN ({reach_ids_str})
        """
        
        reaches_gdf = get_dataframe(self.engine, reaches_query)
        
        # Build nodes query
        nodes_query = f"""
        SELECT * FROM sword_nodes_v16 
        WHERE reach_id IN ({reach_ids_str})
        """
        
        if start_dist_out is not None and end_dist_out is not None:
            nodes_query += f"""
            AND dist_out BETWEEN {end_dist_out} AND {start_dist_out}
            """
        
        nodes_gdf = get_dataframe(self.engine, nodes_query)
        
        logger.info(f"Retrieved {len(reaches_gdf)} reaches and {len(nodes_gdf)} nodes")
        
        return reaches_gdf, nodes_gdf

    def plot_river_network(
        self,
        reaches_gdf: gpd.GeoDataFrame,
        nodes_gdf: gpd.GeoDataFrame,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot a network of connected river reaches with Planet imagery
        
        Args:
            reaches_gdf: GeoDataFrame containing reach geometries
            nodes_gdf: GeoDataFrame containing node geometries
            title: Optional title for the plot
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add reaches
        for _, reach in reaches_gdf.iterrows():
            try:
                coords = list(reach.geometry.coords)
                fig.add_trace(
                    go.Scattermapbox(
                        lat=[coord[1] for coord in coords],
                        lon=[coord[0] for coord in coords],
                        mode='lines',
                        line=dict(width=2),
                        name=f'Reach {reach.reach_id}'
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to plot reach {reach.reach_id}: {str(e)}")
        
        # Add nodes
        fig.add_trace(
            go.Scattermapbox(
                lat=nodes_gdf.geometry.y,
                lon=nodes_gdf.geometry.x,
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Nodes',
                text=[f"Node {node.node_id}<br>Dist: {node.dist_out:.1f}m" for _, node in nodes_gdf.iterrows()],
                hoverinfo='text'
            )
        )
        
        # Calculate center point and bounds
        bounds = nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Calculate zoom level based on bounds
        lat_range = bounds[3] - bounds[1]
        lon_range = bounds[2] - bounds[0]
        zoom = min(10, max(5, int(np.log2(360 / max(lat_range, lon_range)))))
        
        # Get current quarter for Planet imagery
        current_quarter = self._get_planet_quarter(datetime.now())
        
        # Set layout with Planet imagery
        fig.update_layout(
            mapbox=dict(
                style="white-bg",
                zoom=zoom,
                center=dict(lat=center_lat, lon=center_lon),
                layers=[{
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [self._get_planet_basemap_url(current_quarter)],
                    "type": "raster",
                    "opacity": 1
                }] if self.planet_api_key else []
            ),
            title=title or "River Network with Nodes",
            height=800,
            margin={"r":0,"t":30,"l":0,"b":0},
            showlegend=True
        )
        
        return fig

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers including optional API key"""
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["x-hydrocron-key"] = self.config.api_key
        return headers

    @overload
    def get_reach_timeseries(
        self, reach_id: str, 
        start_time: Union[str, datetime], 
        end_time: Union[str, datetime],
        fields: Optional[List[str]] = None,
        include_geometry: Literal[False] = False
    ) -> pd.DataFrame: ...

    @overload
    def get_reach_timeseries(
        self, reach_id: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        fields: Optional[List[str]] = None,
        include_geometry: Literal[True] = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]: ...

    def get_reach_timeseries(
        self,
        reach_id: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        fields: Optional[List[str]] = None,
        include_geometry: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Get time series data for a specific river reach with enhanced error handling
        and data validation.
        """
        # Convert datetimes to ISO format with proper URL encoding
        start_iso = quote_plus(self._to_iso_time(start_time))
        end_iso = quote_plus(self._to_iso_time(end_time))
        
        # Use default fields if none provided
        fields = fields or HydrocronField.default_fields()
        if include_geometry and HydrocronField.GEOMETRY.value not in fields:
            fields.append(HydrocronField.GEOMETRY.value)
            
        # Build request parameters
        params = {
            "feature": FeatureType.REACH.value,
            "feature_id": reach_id,
            "start_time": start_iso,
            "end_time": end_iso,
            "output": "csv",  # Always request CSV format
            "fields": ",".join(fields)
        }
        
        headers = {
            "Accept": "application/json",  # Request JSON wrapper
            **self._build_headers()
        }
        
        try:
            print("\nMaking API request...")
            response = requests.get(
                f"{self.config.base_url}/timeseries",
                headers=headers,
                params=params,
                timeout=self.config.timeout
            )
            
            print(f"\nAPI Response Status: {response.status_code}")
            
            # Handle specific error cases
            if response.status_code == 400:
                error_msg = f"API Error for reach {reach_id}: {response.text}"
                logger.warning(error_msg)
                return pd.DataFrame() if not include_geometry else (pd.DataFrame(), {})
            
            response.raise_for_status()
            
            # Parse JSON response
            json_data = response.json()
            csv_data = json_data.get('results', {}).get('csv', '')
            
            if not csv_data:
                logger.warning(f"No data found for reach {reach_id}")
                return pd.DataFrame() if not include_geometry else (pd.DataFrame(), {})
            
            print("\nAttempting to parse CSV data...")
            df = pd.read_csv(StringIO(csv_data))
            print(f"Parsed DataFrame shape: {df.shape}")
            
            if include_geometry:
                return df, json_data.get('results', {}).get('geojson', {})
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for reach {reach_id}: {str(e)}")
            return pd.DataFrame() if not include_geometry else (pd.DataFrame(), {})
        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error(f"Failed to parse API response for reach {reach_id}: {str(e)}")
            return pd.DataFrame() if not include_geometry else (pd.DataFrame(), {})

    @staticmethod
    def _to_iso_time(time: Union[str, datetime]) -> str:
        """Convert time to ISO 8601 format"""
        if isinstance(time, datetime):
            # Ensure timezone awareness
            if time.tzinfo is None:
                time = time.replace(tzinfo=timezone.utc)
            return time.isoformat()
        return time

    def analyze_reach_stats(
        self, df: pd.DataFrame,
        fields: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate enhanced statistics for reach measurements
        
        Args:
            df: DataFrame with reach measurements
            fields: Optional list of fields to analyze
            
        Returns:
            Dictionary of statistics
        """
        if fields is None:
            fields = ['wse', 'width']
            
        stats: Dict[str, float] = {}
        
        for field in fields:
            if field not in df.columns:
                continue
                
            series = df[field].dropna()
            if len(series) == 0:
                continue
                
            stats.update({
                f"{field}_mean": float(series.mean()),
                f"{field}_std": float(series.std()),
                f"{field}_min": float(series.min()),
                f"{field}_max": float(series.max()),
                f"{field}_median": float(series.median()),
                f"{field}_iqr": float(series.quantile(0.75) - series.quantile(0.25))
            })
            
        stats['n_observations'] = len(df)
        return stats

    def plot_reach_timeseries(
        self, df: pd.DataFrame,
        title: Optional[str] = None,
        quality_threshold: Optional[int] = None
    ) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """Enhanced plotting of reach time series"""
        if quality_threshold is None:
            quality_threshold = self.config.default_quality_threshold
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Apply quality filtering if available
        plot_df = df
        if 'wse_qual' in df.columns:
            plot_df = df[df.wse_qual < quality_threshold].copy()
            
        # Plot water surface elevation
        ax1.plot(plot_df.time_str, plot_df.wse, marker='o')
        ax1.set_ylabel('Water Surface Elevation (m)')
        ax1.set_xlabel('Date')
        ax1.grid(True)
        if title:
            ax1.set_title(f'Water Surface Elevation - {title}')
            
        # Add confidence intervals if uncertainty available
        if 'wse_u' in plot_df.columns:
            ax1.fill_between(
                plot_df.time_str,
                plot_df.wse - plot_df.wse_u,
                plot_df.wse + plot_df.wse_u,
                alpha=0.2
            )
        
        # Plot width
        ax2.plot(plot_df.time_str, plot_df.width, marker='o', color='green')
        ax2.set_ylabel('River Width (m)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        if title:
            ax2.set_title(f'River Width - {title}')
            
        # Add confidence intervals if uncertainty available
        if 'width_u' in plot_df.columns:
            ax2.fill_between(
                plot_df.time_str,
                plot_df.width - plot_df.width_u,
                plot_df.width + plot_df.width_u,
                alpha=0.2
            )
        
        plt.tight_layout()
        return fig, (ax1, ax2)

    def _get_planet_quarter(self, date: datetime) -> str:
        """Convert date to Planet quarterly basemap format (e.g., '2024Q1')"""
        year = date.year
        quarter = (date.month - 1) // 3 + 1
        return f"{year}Q{quarter}"

    def _get_planet_basemap_url(self, quarter: str = "2024Q1") -> str:
        """Construct Planet basemap URL with API key"""
        if not self.planet_api_key:
            raise HydrocronError("Planet API key not found in environment variables")
        return f"https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_{quarter}_mosaic/{{z}}/{{x}}/{{y}}.png?api_key={self.planet_api_key}"

    def _get_base_mapbox_layout(self, center_lat: float, center_lon: float, zoom: int = 12, quarter: Optional[str] = None) -> Dict[str, Any]:
        """Return common mapbox layout settings"""
        layout = {
            "mapbox": {
                "style": "white-bg",
                "zoom": zoom,
                "center": {"lat": center_lat, "lon": center_lon},
            }
        }
        if self.planet_api_key and quarter:
            layout["mapbox"]["layers"] = [{
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Planet Imagery",
                "source": [self._get_planet_basemap_url(quarter)],
                "type": "raster"
            }]
        return layout

    def plot_reach_map(
        self,
        df: pd.DataFrame,
        geojson: Dict[str, Any],
        title: Optional[str] = None,
        use_planet: bool = True
    ) -> go.Figure:
        """
        Create an interactive map of the reach using Plotly
        
        Args:
            df: DataFrame with reach measurements
            geojson: GeoJSON data for the reach
            title: Optional title for the plot
            use_planet: Whether to use Planet basemap
            
        Returns:
            Plotly figure object
        """
        # Calculate center coordinates from geojson
        coords = geojson['features'][0]['geometry']['coordinates']
        center_lat = float(np.mean([coord[1] for coord in coords]))
        center_lon = float(np.mean([coord[0] for coord in coords]))
        
        # Create figure
        fig = go.Figure()
        
        # Add reach line
        fig.add_trace(
            go.Scattermapbox(
                lat=[coord[1] for coord in coords],
                lon=[coord[0] for coord in coords],
                mode='lines',
                line=dict(width=2, color='blue'),
                name='River Reach'
            )
        )
        
        # Add measurement points
        if 'latitude' in df.columns and 'longitude' in df.columns:
            fig.add_trace(
                go.Scattermapbox(
                    lat=df['latitude'],
                    lon=df['longitude'],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    text=df['time_str'].dt.strftime('%Y-%m-%d'),
                    name='Measurements'
                )
            )
        
        # Set layout
        quarter = self._get_planet_quarter(df['time_str'].iloc[-1]) if use_planet else None
        layout = self._get_base_mapbox_layout(center_lat, center_lon, zoom=10, quarter=quarter)
        
        fig.update_layout(
            **layout,
            title=title or "River Reach Map",
            height=600,
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        return fig

    def plot_timeseries_comparison(
        self,
        reach_ids: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> go.Figure:
        """
        Create time series comparison plot for multiple reaches
        
        Args:
            reach_ids: List of reach IDs to compare
            start_time: Start time for time series
            end_time: End time for time series
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Water Surface Elevation", "River Width"),
            specs=[[{"type": "scatter"}], [{"type": "scatter"}]],
            vertical_spacing=0.15
        )
        
        for reach_id in reach_ids:
            try:
                df = self.get_reach_timeseries(
                    reach_id=reach_id,
                    start_time=start_time,
                    end_time=end_time,
                    include_geometry=False
                )
                
                # Add WSE plot
                fig.add_trace(
                    go.Scatter(
                        x=df.time_str,
                        y=df.wse,
                        mode='lines+markers',
                        name=f'WSE - Reach {reach_id}'
                    ),
                    row=1, col=1
                )
                
                # Add width plot
                fig.add_trace(
                    go.Scatter(
                        x=df.time_str,
                        y=df.width,
                        mode='lines+markers',
                        name=f'Width - Reach {reach_id}'
                    ),
                    row=2, col=1
                )
            except Exception as e:
                logger.warning(f"Failed to plot time series for reach {reach_id}: {str(e)}")
        
        fig.update_layout(
            height=800,
            title="Time Series Comparison - Connected Reaches",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Water Surface Elevation (m)", row=1, col=1)
        fig.update_yaxes(title_text="River Width (m)", row=2, col=1)
        
        return fig

    def get_node_timeseries(
        self,
        node_id: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        fields: Optional[List[str]] = None,
        include_geometry: bool = False
    ) -> pd.DataFrame:
        """Get time series data for a specific river node"""
        # Format node ID if needed (CBBBBBRRRRNNNT)
        node_id = str(node_id)
        
        # Convert times to ISO format with proper URL encoding
        start_iso = quote_plus(self._to_iso_time(start_time))
        end_iso = quote_plus(self._to_iso_time(end_time))
        
        # Build request parameters
        params = {
            "feature": FeatureType.NODE.value,
            "feature_id": node_id,
            "start_time": start_iso,
            "end_time": end_iso,
            "output": "geojson" if include_geometry else "csv",
            "fields": ",".join(fields or ['node_id', 'reach_id', 'time_str', 'wse', 'width'])
        }
        
        # Set proper Accept header based on desired output
        headers = {
            "Accept": "application/geo+json" if include_geometry else "text/csv",
            **self._build_headers()
        }
        
        try:
            response = requests.get(
                f"{self.config.base_url}/timeseries",
                headers=headers,
                params=params,
                timeout=self.config.timeout
            )
            
            # Handle specific error cases
            if response.status_code == 400:
                logger.warning(f"No data found for node {node_id}: {response.text}")
                return pd.DataFrame()
            
            response.raise_for_status()
            
            if not response.text:
                logger.warning(f"Empty response for node {node_id}")
                return pd.DataFrame()
            
            # Parse response based on format
            if include_geometry:
                data = response.json()
                if not data.get('features', []):
                    return pd.DataFrame()
                
                # Extract properties from features
                records = []
                for feature in data['features']:
                    records.append(feature['properties'])
                return pd.DataFrame(records)
            else:
                try:
                    df = pd.read_csv(StringIO(response.text))
                    return df
                except pd.errors.EmptyDataError:
                    return pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for node {node_id}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error for node {node_id}: {str(e)}")
            return pd.DataFrame()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _fetch_node_timeseries(
        self,
        node_id: int,
        start_time: datetime,
        end_time: datetime,
        fields: List[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch timeseries data for a single node"""
        try:
            # Format dates with Z suffix for UTC time
            start_iso = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_iso = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Build query parameters
            params = {
                'feature': 'Node',
                'feature_id': str(node_id),
                'start_time': start_iso,
                'end_time': end_iso,
                'fields': ','.join(fields),
                'output': 'csv'  # Request CSV format
            }
            
            # Set headers according to API docs
            headers = {
                'Accept': 'application/json',  # Request JSON wrapper
                **self._build_headers()
            }
            
            response = requests.get(
                f"{self.config.base_url}/timeseries",
                params=params,
                headers=headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 400:
                logger.warning(f"No data found for node {node_id}: {response.text}")
                return None
                
            response.raise_for_status()
            
            # Parse JSON wrapper to get CSV content
            json_data = response.json()
            if not json_data.get('results', {}).get('csv'):
                logger.warning(f"No data found for node {node_id}")
                return None
                
            # Parse CSV content
            df = pd.read_csv(StringIO(json_data['results']['csv']))
            
            if not df.empty:
                logger.info(f"Successfully parsed data: {len(df)} rows")
                logger.debug(f"Sample data:\n{df.head()}")
            
            return df
                
        except Exception as e:
            logger.error(f"Error processing node {node_id}: {str(e)}")
            return None

    def fetch_parallel_timeseries(
        self,
        node_ids: List[int],
        start_time: datetime,
        end_time: datetime,
        fields: List[str],
        max_workers: int = 5,
        batch_size: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        cache_dir: Optional[Path] = Path("cache")
    ) -> Dict[str, pd.DataFrame]:
        """Fetch timeseries data for multiple nodes in parallel with caching"""
        
        # Create cache directory if it doesn't exist
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cache key based on parameters
        cache_key = f"nodes_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.parquet"
        cache_path = cache_dir / cache_key if cache_dir else None
        
        print(f"DEBUG: Cache path = {cache_path}")
        
        # Try to load from cache first
        if cache_path and cache_path.exists():
            try:
                print("DEBUG: Attempting to read from cache...")
                cached_df = pd.read_parquet(cache_path)
                print(f"DEBUG: Successfully read {len(cached_df)} records from cache")
                
                # Convert cached data back to dictionary format
                results = {}
                for node_id in node_ids:
                    node_data = cached_df[cached_df['node_id'] == str(node_id)]
                    if not node_data.empty:
                        results[str(node_id)] = node_data.copy()
                        print(f"DEBUG: Loaded {len(node_data)} records for node {node_id} from cache")
                
                if results:
                    print(f"DEBUG: Successfully loaded {len(results)} nodes from cache")
                    return results
                else:
                    print("DEBUG: Cache exists but no valid data found")
                    cache_path.unlink()  # Delete invalid cache
            except Exception as e:
                print(f"DEBUG: Error reading cache: {str(e)}")
                if cache_path.exists():
                    cache_path.unlink()
        
        print("DEBUG: Fetching fresh data...")
        
        # Fetch data
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(0, len(node_ids), batch_size):
                batch = node_ids[i:i + batch_size]
                for node_id in batch:
                    future = executor.submit(
                        self._fetch_node_timeseries,
                        node_id,
                        start_time,
                        end_time,
                        fields
                    )
                    futures[future] = node_id
            
            # Process completed futures
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        # Ensure node_id is set in the DataFrame
                        df['node_id'] = str(node_id)  # Make sure node_id is set
                        results[str(node_id)] = df
                        print(f"DEBUG: Got data for node {node_id}: {df.shape}")
                except Exception as e:
                    print(f"DEBUG: Error processing node {node_id}: {str(e)}")
        
        print(f"DEBUG: Fetched data for {len(results)} nodes")
        
        # Save to cache if we got any results
        if results and cache_path:
            try:
                # Combine all DataFrames
                combined_df = pd.concat(results.values(), ignore_index=True)
                print(f"DEBUG: Saving {len(combined_df)} records to cache")
                # Verify node_ids are present
                print(f"DEBUG: Sample of data to cache:")
                print(combined_df[['node_id', 'reach_id', 'time_str']].head())
                combined_df.to_parquet(cache_path)
                print("DEBUG: Successfully saved to cache")
            except Exception as e:
                print(f"DEBUG: Failed to save cache: {str(e)}")
                if cache_path.exists():
                    cache_path.unlink()
        
        return results

    def _fetch_node_with_retry(
        self,
        node_id: int,
        start_time: datetime,
        end_time: datetime,
        fields: List[str],
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single node with retry logic"""
        for attempt in range(retry_attempts):
            try:
                return self._fetch_node_timeseries(node_id, start_time, end_time, fields)
            except Exception as e:
                if attempt == retry_attempts - 1:
                    logger.error(f"Failed to fetch node {node_id} after {retry_attempts} attempts: {str(e)}")
                    return None
                time.sleep(retry_delay * (attempt + 1))
        return None

    def animate_profiles(
        self,
        fitted_curves: List[Dict[str, Any]],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        fade_steps: int = 15,
        fps: int = 30
    ) -> animation.FuncAnimation:
        """Create an animated visualization of water surface profiles"""
        # Create figure and axis with improved styling
        fig = plt.figure(figsize=(12, 6), facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#f8f9fa')
        
        # Get min/max values for consistent axes
        all_x = np.concatenate([curve['x'] for curve in fitted_curves])
        all_y = np.concatenate([curve['y'] for curve in fitted_curves])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        y_range = y_max - y_min
        
        # Sort curves by date for smooth animation
        fitted_curves = sorted(fitted_curves, key=lambda x: x['date'])
        
        # Create colormap for fading effect using blues
        colors = plt.cm.Blues(np.linspace(0.3, 1, fade_steps))
        
        # Initialize empty line objects for previous curves and current curve
        fade_lines = [ax.plot([], [], alpha=0, color='blue')[0] for _ in range(fade_steps)]
        current_line, = ax.plot([], [], color=colors[-1], linewidth=2.5)
        
        # Add progress bar
        progress_bar = ax.axhline(y=y_min - 0.05*y_range, xmin=0, xmax=0, 
                                 color='blue', linewidth=3, alpha=0.7)
        
        # Add date text with improved styling
        date_text = ax.text(
            0.02, 0.95, '', transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5)
        )
        
        def init():
            """Initialize animation with improved styling"""
            ax.set_xlim(x_max, x_min)  # Reversed x-axis
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
            ax.set_xlabel('Distance from Outlet (m)', fontsize=10)
            ax.set_ylabel('Water Surface Elevation (m)', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            if title:
                ax.set_title(title, fontsize=12, pad=10)
            return fade_lines + [current_line, progress_bar, date_text]
        
        def update(frame):
            """Update animation frame with smooth fading"""
            i = frame % len(fitted_curves)
            curve = fitted_curves[i]
            
            # Update current line
            current_line.set_data(curve['x'], curve['y'])
            
            # Update faded previous curves with smooth color transition
            for j in range(fade_steps):
                prev_idx = i - j - 1
                if prev_idx >= 0:
                    prev_curve = fitted_curves[prev_idx]
                    fade_lines[j].set_data(prev_curve['x'], prev_curve['y'])
                    # Use blues colormap with decreasing alpha
                    fade_lines[j].set_color(colors[fade_steps-j-1])
                    fade_lines[j].set_alpha(0.8 * (1 - (j+1)/fade_steps))
                else:
                    fade_lines[j].set_data([], [])
                    fade_lines[j].set_alpha(0)
            
            # Update progress bar
            progress = (i + 1) / len(fitted_curves)
            progress_bar.set_xdata([x_min, x_min + progress * (x_max - x_min)])
            
            # Update date text with improved formatting
            date_text.set_text(f'Date: {curve["date"].strftime("%Y-%m-%d")}')
            
            return fade_lines + [current_line, progress_bar, date_text]
        
        # Create animation with smoother frame rate
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=len(fitted_curves), 
            interval=1000/fps,
            blit=True
        )
        
        # Save animation if path provided
        if save_path:
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
        
        plt.close(fig)
        return anim

    def _clean_numeric_value(self, value: Any) -> Optional[float]:
        """Clean and validate numeric values from API response"""
        if pd.isna(value):
            return None
        
        try:
            # If it's already a number, return it
            if isinstance(value, (int, float)):
                return float(value)
            
            # If it's a string, try direct conversion
            if isinstance(value, str):
                # Remove any whitespace and handle scientific notation
                cleaned = value.strip()
                try:
                    return float(cleaned)
                except ValueError:
                    # If direct conversion fails, try to extract the first number
                    import re
                    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', cleaned)
                    return float(numbers[0]) if numbers else None
            
            return None
        except Exception as e:
            logger.debug(f"Could not convert value '{value}' to numeric: {str(e)}")
            return None

    def plot_node_comparison(
        self,
        node_dfs: Dict[str, pd.DataFrame],
        nodes_gdf: gpd.GeoDataFrame,
        reaches_gdf: gpd.GeoDataFrame,
        title: Optional[str] = None,
        ransac_zscore_threshold: float = 3.0,
        ransac_min_samples: int = 100,
        cache_file: Optional[str] = None,
        create_animation: bool = False,
        animation_path: Optional[str] = None,
        show_outliers: bool = False,
        color_by: str = 'wse_u',  # Variable to color points by
        point_size: int = 8,      # Size of points
        colorscale: str = 'Viridis'  # Colorscale to use
    ) -> Union[go.Figure, Tuple[go.Figure, animation.FuncAnimation]]:
        """Plot node comparison with uncertainty coloring"""
        logger.info(f"Starting visualization with {len(node_dfs)} nodes")
        
        try:
            # Create figure with two subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Water Surface Profiles with Outlier Detection", "Fitted Profiles Only"),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )
            
            # Convert node IDs to strings in nodes_gdf
            nodes_gdf = nodes_gdf.copy()
            nodes_gdf['node_id'] = nodes_gdf['node_id'].astype(str)
            
            # Debug print
            logger.info(f"Processing {len(node_dfs)} node dataframes")
            
            # Pre-process all dataframes to ensure numeric columns
            processed_dfs = {}
            for node_id, df in node_dfs.items():
                try:
                    if df is None or df.empty:
                        continue
                        
                    # Convert time_str to datetime
                    df['datetime'] = pd.to_datetime(df['time_str'])
                    df['date'] = df['datetime'].dt.date
                    
                    # Convert numeric columns
                    numeric_columns = ['wse', 'width', 'wse_u', 'width_u']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = df[col].apply(self._clean_numeric_value)
                    
                    # Only keep rows where we have valid WSE values
                    df = df.dropna(subset=['wse'])
                    
                    if not df.empty:
                        processed_dfs[node_id] = df
                        logger.info(f"Node {node_id}: {len(df)} valid measurements")
                        
                except Exception as e:
                    logger.error(f"Error processing node {node_id}: {str(e)}")
                    continue
            
            if not processed_dfs:
                logger.error("No valid data after processing")
                return fig
            
            # Get all unique dates
            all_dates = set()
            for df in processed_dfs.values():
                all_dates.update(df['date'].unique())
            all_dates = sorted(list(all_dates))
            logger.info(f"Found {len(all_dates)} unique dates")
            
            # Process each date
            fitted_curves = []
            for date in all_dates:
                profile_data = []
                
                # Collect data for this date
                for node_id, df in processed_dfs.items():
                    day_data = df[df['date'] == date]
                    if not day_data.empty:
                        node_info = nodes_gdf[nodes_gdf['node_id'] == node_id]
                        if not node_info.empty:
                            dist_out = node_info['dist_out'].iloc[0]
                            wse = day_data['wse'].mean()
                            
                            if pd.notna(dist_out) and pd.notna(wse):
                                profile_data.append({
                                    'dist_out': dist_out,
                                    'wse': wse,
                                    'uncertainty': day_data[color_by].mean() if color_by in day_data else 1.0
                                })
                
                # If we have enough points, fit RANSAC
                if len(profile_data) >= ransac_min_samples:
                    df = pd.DataFrame(profile_data)
                    
                    # Add to plot with configurable parameters
                    fig.add_trace(
                        go.Scatter(
                            x=df['dist_out'],
                            y=df['wse'],
                            mode='markers',
                            marker=dict(
                                size=point_size,
                                color=df['uncertainty'],
                                colorscale=colorscale,
                                showscale=True,
                                colorbar=dict(
                                    title=color_by.replace('_', ' ').title()
                                )
                            ),
                            name=f'Profile {date}'
                        ),
                        row=1, col=1
                    )
                    
                    # Store for animation
                    fitted_curves.append({
                        'date': pd.to_datetime(date),
                        'x': df['dist_out'].values,
                        'y': df['wse'].values
                    })
            
            # Create animation if requested and we have data
            anim = None
            if create_animation and fitted_curves:
                try:
                    anim = self.animate_profiles(
                        fitted_curves=fitted_curves,
                        title=title,
                        save_path=animation_path
                    )
                except Exception as e:
                    logger.error(f"Animation creation failed: {str(e)}")
                    create_animation = False
            
            # Update layout
            fig.update_layout(
                height=800,
                title=title or "River Profiles",
                showlegend=True
            )
            
            # Return appropriate result
            if create_animation and anim is not None:
                return fig, anim
            return fig
            
        except Exception as e:
            logger.error(f"Error in plot_node_comparison: {str(e)}")
            raise

    def fetch_node_and_reach_data(
        self,
        nodes_gdf: gpd.GeoDataFrame,
        start_time: datetime,
        end_time: datetime,
        cache_file: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch both node and reach data for backwater analysis
        
        Args:
            nodes_gdf: GeoDataFrame containing nodes with their reach_ids
            start_time: Start time for query
            end_time: End time for query
            cache_file: Optional path to cache results
            
        Returns:
            Dictionary containing:
                - 'nodes': DataFrame with node measurements
                - 'reaches': DataFrame with reach measurements
        """
        # Get unique reach IDs from nodes
        reach_ids = nodes_gdf['reach_id'].unique()
        logger.info(f"Found {len(reach_ids)} unique reaches for {len(nodes_gdf)} nodes")
        
        # Fields we want from each data type
        node_fields = [
            'node_id', 'reach_id', 'time_str', 
            'wse', 'wse_u', 'wse_r_u',
            'width', 'width_u',
            'dark_frac', 'ice_clim_f', 'partial_f', 'n_good_pix'
        ]
        
        reach_fields = [
            'reach_id', 'time_str',
            'slope', 'slope_u', 'slope_r_u',
            'dschg_c', 'dschg_c_u',  # composite discharge
            'width', 'width_u'
        ]
        
        # Fetch data in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Fetch node data
            node_futures = {
                executor.submit(
                    self._fetch_node_timeseries,
                    node_id,
                    start_time,
                    end_time,
                    node_fields
                ): node_id 
                for node_id in nodes_gdf['node_id']
            }
            
            # Fetch reach data
            reach_futures = {
                executor.submit(
                    self.get_reach_timeseries,
                    reach_id,
                    start_time,
                    end_time,
                    reach_fields
                ): reach_id 
                for reach_id in reach_ids
            }
            
            # Collect results
            node_data = {}
            reach_data = {}
            
            # Get node results
            for future in as_completed(node_futures):
                node_id = node_futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        node_data[node_id] = df
                except Exception as e:
                    logger.error(f"Error fetching node {node_id}: {str(e)}")
                
            # Get reach results
            for future in as_completed(reach_futures):
                reach_id = reach_futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        reach_data[reach_id] = df
                except Exception as e:
                    logger.error(f"Error fetching reach {reach_id}: {str(e)}")
        
        # Combine and merge data
        if node_data and reach_data:
            nodes_df = pd.concat(node_data.values(), ignore_index=True)
            reaches_df = pd.concat(reach_data.values(), ignore_index=True)
            
            # Cache if requested
            if cache_file:
                combined = {
                    'nodes': nodes_df,
                    'reaches': reaches_df
                }
                pd.to_pickle(combined, cache_file)
                
            return {
                'nodes': nodes_df,
                'reaches': reaches_df
            }
        else:
            raise HydrocronError("No data retrieved for nodes or reaches")

    def plot_chronohydrograph(
        self,
        node_dfs: Dict[str, pd.DataFrame],
        nodes_gdf: gpd.GeoDataFrame,
        title: Optional[str] = None,
        uncertainty_fill: bool = True,
        time_resolution: str = 'D',
        colormap: str = 'viridis'
    ) -> go.Figure:
        """
        Create a chronostratigraphic-style visualization of water surface profiles over time.
        """
        # Initial data validation
        logger.info(f"Starting chronohydrograph with {len(node_dfs)} nodes")
        logger.info(f"Nodes GDF shape: {nodes_gdf.shape}")
        
        # Convert node IDs to strings in nodes_gdf
        nodes_gdf = nodes_gdf.copy()
        nodes_gdf['node_id'] = nodes_gdf['node_id'].astype(str)
        
        # Process and combine data
        combined_data = []
        for node_id, df in node_dfs.items():
            if df is None or df.empty:
                logger.debug(f"Skipping empty DataFrame for node {node_id}")
                continue
            
            try:
                # Get distance from outlet for this node
                node_info = nodes_gdf[nodes_gdf['node_id'].str.contains(str(node_id))]
                
                if node_info.empty:
                    # Try alternative matching
                    node_id_str = str(node_id)
                    if len(node_id_str) > 11:
                        base_node_id = node_id_str[:11]
                        node_info = nodes_gdf[nodes_gdf['node_id'].str.contains(base_node_id)]
                    
                    if node_info.empty:
                        logger.warning(f"No node info found for node {node_id}")
                        continue
                
                dist_out = node_info['dist_out'].iloc[0]
                
                # Convert time and add distance
                df['datetime'] = pd.to_datetime(df['time_str'])
                df['dist_out'] = dist_out
                
                # Define resampling columns based on what's available
                resample_dict = {
                    'wse': 'mean',
                    'dist_out': 'first'
                }
                
                # Add uncertainty if available
                if 'wse_u' in df.columns:
                    resample_dict['wse_u'] = 'mean'
                
                # Resample to specified temporal resolution
                resampled = df.set_index('datetime').resample(time_resolution).agg(resample_dict).reset_index()
                
                if not resampled.empty and resampled['wse'].notna().any():
                    combined_data.append(resampled)
                    logger.debug(f"Added {len(resampled)} records for node {node_id}")
                    
            except Exception as e:
                logger.error(f"Error processing node {node_id}: {str(e)}")
                continue
        
        if not combined_data:
            logger.error("No valid data after processing")
            raise ValueError("No valid data to plot")
        
        # Combine all data
        all_data = pd.concat(combined_data, ignore_index=True)
        logger.info(f"Final combined data shape: {all_data.shape}")
        
        # Create figure
        fig = go.Figure()
        
        # Get unique timestamps
        timestamps = sorted(all_data['datetime'].unique())
        logger.info(f"Found {len(timestamps)} unique timestamps")
        
        colors = px.colors.sample_colorscale(colormap, len(timestamps))
        
        # Plot each time slice
        for i, time in enumerate(timestamps):
            time_data = all_data[all_data['datetime'] == time].sort_values('dist_out')
            
            if len(time_data) < 2:
                logger.debug(f"Skipping timestamp {time}: insufficient data points")
                continue
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=time_data['dist_out'],
                y=time_data['wse'],
                mode='lines',
                line=dict(color=colors[i], width=2),
                name=time.strftime('%Y-%m-%d'),
                showlegend=True
            ))
            
            # Add uncertainty fill if available and requested
            if uncertainty_fill and 'wse_u' in time_data.columns:
                fig.add_trace(go.Scatter(
                    x=time_data['dist_out'],
                    y=time_data['wse'] + time_data['wse_u'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=time_data['dist_out'],
                    y=time_data['wse'] - time_data['wse_u'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[i])) + [0.2])}',
                    fill='tonexty',
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=title or "Spatial Hydrograph",
            xaxis_title="Distance from Outlet (m)",
            yaxis_title="Water Surface Elevation (m)",
            hovermode='x unified',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig

def main():
    """Main function to test the API"""
    load_dotenv()
    api = HydrocronAPI()
    
    # Example reach ID and time range - using one from the API docs
    reach_id = "63470800171"  # Known valid reach ID from API docs
    start_time = datetime(2022, 2, 1)
    end_time = datetime(2024, 3, 1)
    
    print(f"Testing API with:")
    print(f"Reach ID: {reach_id}")
    print(f"Time range: {start_time} to {end_time}")
    
    try:
        # Get time series data with debug info
        print("\nFetching time series data...")
        
        # Build the request URL for debugging
        start_iso = quote_plus(api._to_iso_time(start_time))
        end_iso = quote_plus(api._to_iso_time(end_time))
        fields = ['reach_id', 'time_str', 'wse', 'width']
        
        params = {
            "feature": "Reach",
            "feature_id": reach_id,
            "start_time": start_iso,
            "end_time": end_iso,
            "fields": ",".join(fields)
        }
        
        # Print request details
        print("\nRequest details:")
        print(f"URL: {api.config.base_url}/timeseries")
        print(f"Parameters: {params}")
        print(f"Headers: {api._build_headers()}")
        
        # Make the request
        df = api.get_reach_timeseries(
            reach_id=reach_id,
            start_time=start_time,
            end_time=end_time,
            fields=fields
        )
        
        if not df.empty:
            print("\nData retrieved successfully!")
            print("\nFirst few rows:")
            print(df.head())
            
            # Create and show plot
            print("\nCreating plot...")
            fig, (ax1, ax2) = api.plot_reach_timeseries(df, title=f"Reach {reach_id}")
            plt.show()
        else:
            print("\nNo data found. Empty DataFrame returned.")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 
