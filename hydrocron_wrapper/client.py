from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Dict, Any, Tuple, List, Literal
import logging
from io import StringIO
from urllib.parse import quote_plus
import time
import threading
import json
import hashlib
from pathlib import Path
import os

import requests
import pandas as pd
import plotly.graph_objects as go
import geopandas as gpd
import numpy as np
from pydantic import ValidationError
import plotly.express as px
import plotly.subplots as make_subplots
from shapely.geometry import shape

from .types import (
    HydrocronConfig, TimeseriesRequest, 
    FeatureType, OutputFormat, HydrocronField,
    ResponseFormat, TimeseriesResponse, 
    GeoJSONResponse, CSVResponse
)
from .utils import get_planet_quarter, get_mapbox_layout, clean_reach_id

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# At the top of the file, update the BASE_URL constant
BASE_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1"

class HydrocronError(Exception):
    """Base exception for Hydrocron API errors"""
    pass

class HydrocronValidationError(HydrocronError):
    """Raised when request validation fails"""
    pass

class HydrocronAPIError(HydrocronError):
    """Raised when API request fails"""
    pass

class RateLimitExceededError(HydrocronError):
    """Raised when rate limit is exceeded"""
    pass

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: float = 10.0, burst: int = 20):
        """
        Initialize rate limiter
        
        Args:
            rate: Requests per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = threading.Lock()
        
    def _add_tokens(self) -> None:
        """Add tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.rate
        self.tokens = min(self.burst, self.tokens + new_tokens)
        self.last_update = now
        
    def acquire(self, block: bool = True) -> bool:
        """
        Acquire a token
        
        Args:
            block: Whether to block until a token is available
            
        Returns:
            True if token acquired, False otherwise
            
        Raises:
            RateLimitExceededError: If blocking and timeout occurs
        """
        with self._lock:
            self._add_tokens()
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
                
            if not block:
                return False
                
            wait_time = (1 - self.tokens) / self.rate
            if wait_time > 60:  # Don't wait more than 60 seconds
                raise RateLimitExceededError(
                    f"Rate limit exceeded. Would need to wait {wait_time:.1f}s"
                )
                
            time.sleep(wait_time)
            self.tokens = 0  # Reset after waiting
            return True

class ResponseCache:
    """Cache for API responses with TTL"""
    
    def __init__(self, ttl: timedelta = timedelta(hours=1), cache_dir: Optional[str] = None):
        """
        Initialize cache
        
        Args:
            ttl: Time-to-live for cached responses
            cache_dir: Directory to store cache files (default: ~/.hydrocron/cache)
        """
        self.ttl = ttl
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.hydrocron/cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
    def _get_cache_key(self, url: str, params: Dict[str, Any]) -> str:
        """Generate cache key from request parameters"""
        # Sort params to ensure consistent keys
        sorted_params = {k: params[k] for k in sorted(params.keys())}
        cache_str = f"{url}:{json.dumps(sorted_params)}"
        return hashlib.sha256(cache_str.encode()).hexdigest()
        
    def _get_cache_path(self, key: str) -> Path:
        """Get path to cache file"""
        return self.cache_dir / f"{key}.json"
        
    def get(self, url: str, params: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """
        Get cached response if available and not expired
        
        Returns:
            Tuple of (content, content_type) if cached, None if not found or expired
        """
        key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        with self._lock:
            try:
                with open(cache_path) as f:
                    cache_data = json.load(f)
                    
                # Check if expired
                cached_time = datetime.fromisoformat(cache_data["timestamp"])
                if datetime.now() - cached_time > self.ttl:
                    os.unlink(cache_path)
                    return None
                    
                return cache_data["content"], cache_data["content_type"]
                
            except (json.JSONDecodeError, KeyError, OSError):
                # Invalid cache file
                if cache_path.exists():
                    os.unlink(cache_path)
                return None
        
    def set(self, url: str, params: Dict[str, Any], content: str, content_type: str) -> None:
        """Store response in cache"""
        key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(key)
        
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "content_type": content_type
        }
        
        with self._lock:
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)

class HydrocronClient:
    """Client for interacting with the Hydrocron API"""
    
    def __init__(self, config: Optional[HydrocronConfig] = None):
        """Initialize the Hydrocron API client"""
        self.config = config or HydrocronConfig()
        self._session = requests.Session()
        
        # Set default headers
        self._session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Initialize rate limiter and cache
        self._rate_limiter = RateLimiter()
        self._cache = ResponseCache()
            
    def _make_request(
        self,
        url: str,
        params: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ) -> requests.Response:
        """Make HTTP request with retry logic, rate limiting, and caching"""
        # Combine base URL with path
        full_url = f"{self.config.base_url.rstrip('/')}/{url.lstrip('/')}"
        
        # Check cache first
        cached = self._cache.get(full_url, params)
        if cached is not None:
            content, content_type = cached
            logger.debug("Using cached response")
            
            # Create a Response-like object
            response = requests.Response()
            response.status_code = 200
            response._content = content.encode()
            response.headers["Content-Type"] = content_type
            return response
            
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Wait for rate limit
                if not self._rate_limiter.acquire():
                    raise RateLimitExceededError("Rate limit exceeded")
                    
                response = self._session.get(
                    full_url,  # Use the full URL
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                # Cache successful response
                content_type = response.headers.get("Content-Type", "")
                self._cache.set(full_url, params, response.text, content_type)
                
                return response
                
            except RateLimitExceededError:
                raise  # Don't retry rate limit errors
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                continue
                
        raise HydrocronAPIError(f"Request failed after {max_retries} attempts: {str(last_exception)}")

    def get_timeseries(
        self,
        feature: str,
        feature_id: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get time series data"""
        # Format parameters
        params = self._format_params(
            feature, feature_id, start_time, end_time, fields
        )
        
        # Make request using default headers
        response = self._make_request(
            url='/timeseries',
            params=params
        )
        
        # Parse response
        return self._parse_csv_response(response)

    def _parse_csv_response(self, response: Union[requests.Response, Dict[str, Any]]) -> pd.DataFrame:
        """Parse CSV response into DataFrame"""
        try:
            # Get data from either Response or dict
            data = response.json() if isinstance(response, requests.Response) else response
            csv_data = data.get('results', {}).get('csv', '')
            if not csv_data:
                return pd.DataFrame()
                
            # Read CSV with string dtype for reach_id
            df = pd.read_csv(StringIO(csv_data), dtype={'reach_id': str})
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse CSV response: {str(e)}")
            raise

    def _parse_geojson_response(self, response: Union[requests.Response, Dict[str, Any]]) -> gpd.GeoDataFrame:
        """Parse GeoJSON response into GeoDataFrame"""
        try:
            # Get data from either Response or dict
            data = response.json() if isinstance(response, requests.Response) else response
            logger.debug(f"Initial response data: {data}")
            
            # Get GeoJSON data - it's directly in results for our test case
            geojson_data = data.get('results', {}).get('geojson', {})
            logger.debug(f"Extracted GeoJSON data: {geojson_data}")
            
            if not geojson_data:
                logger.debug("No GeoJSON data found in response")
                return gpd.GeoDataFrame()
            
            # Convert to GeoDataFrame
            features = []
            for feature in geojson_data.get('features', []):
                logger.debug(f"Processing feature: {feature}")
                properties = feature.get('properties', {})
                # Ensure reach_id is string
                if 'reach_id' in properties:
                    properties['reach_id'] = str(properties['reach_id'])
                    
                # Get geometry data
                geometry_data = feature.get('geometry', {})
                try:
                    geometry = shape(geometry_data)
                    features.append({
                        **properties,
                        'geometry': geometry
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse geometry: {str(e)}")
                    continue
            
            logger.debug(f"Extracted features: {features}")
            if not features:
                logger.debug("No features found in GeoJSON data")
                return gpd.GeoDataFrame()
            
            gdf = gpd.GeoDataFrame(features)
            logger.debug(f"Created GeoDataFrame: {gdf}")
            
            if not gdf.empty:
                gdf.set_geometry('geometry', inplace=True)
                logger.debug(f"Final GeoDataFrame: {gdf}")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to parse GeoJSON response: {str(e)}")
            logger.error(f"Response data: {data}")
            raise

    def plot_river_network(
        self,
        reaches_gdf: gpd.GeoDataFrame,
        nodes_gdf: gpd.GeoDataFrame,
        title: Optional[str] = None,
        use_planet: bool = False,
        planet_key: Optional[str] = None,
        planet_quarter: Optional[str] = None
    ) -> go.Figure:
        """Plot a network of connected river reaches"""
        fig = go.Figure()
        
        # Handle empty GeoDataFrames
        if reaches_gdf.empty and nodes_gdf.empty:
            return fig
        
        # Add reaches if available
        if not reaches_gdf.empty and 'geometry' in reaches_gdf.columns:
            for _, reach in reaches_gdf.iterrows():
                try:
                    if reach.geometry is not None:
                        coords = list(reach.geometry.coords)
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=[coord[1] for coord in coords],
                                lon=[coord[0] for coord in coords],
                                mode='lines',
                                line=dict(width=2),
                                name=f'Reach {reach.get("reach_id", "unknown")}'
                            )
                        )
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Failed to plot reach: {str(e)}")
        
        # Add nodes if available
        if not nodes_gdf.empty and 'geometry' in nodes_gdf.columns:
            try:
                fig.add_trace(
                    go.Scattermapbox(
                        lat=nodes_gdf.geometry.y,
                        lon=nodes_gdf.geometry.x,
                        mode='markers',
                        marker=dict(size=5, color='red'),
                        name='Nodes',
                        text=[f"Node {node.get('node_id', 'unknown')}<br>Dist: {node.get('dist_out', 0):.1f}m" 
                              for _, node in nodes_gdf.iterrows()],
                        hoverinfo='text'
                    )
                )
            except (AttributeError, TypeError) as e:
                logger.warning(f"Failed to plot nodes: {str(e)}")
        
        # Calculate center point and bounds
        bounds = nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Calculate zoom level based on bounds
        lat_range = bounds[3] - bounds[1]
        lon_range = bounds[2] - bounds[0]
        zoom = min(10, max(5, int(np.log2(360 / max(lat_range, lon_range)))))
        
        # Set layout with optional Planet basemap
        if use_planet:
            if planet_quarter is None:
                planet_quarter = get_planet_quarter(datetime.now())
            layout = get_mapbox_layout(
                center_lat=center_lat,
                center_lon=center_lon,
                zoom=zoom,
                planet_key=planet_key,
                planet_quarter=planet_quarter
            )
        else:
            layout = get_mapbox_layout(
                center_lat=center_lat,
                center_lon=center_lon,
                zoom=zoom
            )
            
        # Update layout
        layout.update({
            "title": title or "River Network with Nodes",
            "height": 800,
            "margin": {"r":0,"t":30,"l":0,"b":0},
            "showlegend": True
        })
        fig.update_layout(**layout)
        
        return fig

    def traverse_river_network(
        self,
        start_reach_id: str,
        direction: Literal['up', 'down'] = 'down',
        max_reaches: int = 100,
        min_facc: Optional[float] = None
    ) -> List[str]:
        """
        Traverse the river network upstream or downstream from a starting reach
        
        Args:
            start_reach_id: ID of the starting reach
            direction: Direction to traverse ('up' for upstream, 'down' for downstream)
            max_reaches: Maximum number of reaches to traverse
            min_facc: Minimum flow accumulation threshold
            
        Returns:
            List of reach IDs in traversal order
        """
        # Clean reach ID
        start_reach_id = clean_reach_id(start_reach_id)
        logger.info(f"Starting {direction}stream traversal from reach {start_reach_id}")
        
        # Get initial reach data
        params = {
            "feature": FeatureType.REACH.value,
            "feature_id": start_reach_id,
            "fields": "reach_id,rch_id_up,rch_id_dn,facc",
            "output": OutputFormat.CSV.value
        }
        
        try:
            response = self._make_request(f"{self.config.base_url}/reach", params)
            start_df = pd.read_csv(StringIO(response.text))
            
            if start_df.empty:
                logger.error(f"Start reach {start_reach_id} not found")
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
                
                # Add reach if it meets flow accumulation threshold
                if min_facc is None or current_facc >= min_facc:
                    reach_ids.append(current_reach_id)
                    logger.debug(f"Added reach {current_reach_id} (FACC: {current_facc})")
                    
                    # Get next reaches
                    next_field = 'rch_id_up' if direction == 'up' else 'rch_id_dn'
                    params['feature_id'] = current_reach_id
                    
                    try:
                        response = self._make_request(f"{self.config.base_url}/reach", params)
                        df = pd.read_csv(StringIO(response.text))
                        
                        if not df.empty:
                            next_reaches = str(df.iloc[0][next_field]).split()
                            next_reaches = [r for r in next_reaches if r and r.strip()]
                            
                            if next_reaches:
                                logger.debug(f"Found {len(next_reaches)} {direction}stream reaches")
                                
                                # Get flow accumulation for next reaches
                                for next_id in next_reaches:
                                    if next_id not in visited:
                                        params['feature_id'] = next_id
                                        try:
                                            response = self._make_request(f"{self.config.base_url}/reach", params)
                                            next_df = pd.read_csv(StringIO(response.text))
                                            
                                            if not next_df.empty:
                                                next_facc = float(next_df.iloc[0]['facc'])
                                                
                                                # Add to queue based on direction
                                                if direction == 'down':
                                                    if next_facc <= current_facc * 1.2:  # Allow 20% increase
                                                        queue.append((next_id, next_facc))
                                                else:
                                                    queue.append((next_id, next_facc))
                                                    
                                        except Exception as e:
                                            logger.error(f"Error getting next reach {next_id}: {str(e)}")
                                            continue
                                            
                    except Exception as e:
                        logger.error(f"Error processing reach {current_reach_id}: {str(e)}")
                        continue
            
            logger.info(f"Traversal complete. Found {len(reach_ids)} reaches")
            return reach_ids
            
        except Exception as e:
            logger.error(f"Error starting traversal: {str(e)}")
            return [] 

    def analyze_timeseries(
        self,
        df: pd.DataFrame,
        fields: Optional[List[str]] = None,
        quality_threshold: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for time series measurements
        
        Args:
            df: DataFrame with measurements
            fields: Optional list of fields to analyze (default: ['wse', 'width'])
            quality_threshold: Optional quality threshold for filtering
            
        Returns:
            Dictionary of statistics by field
        """
        if fields is None:
            fields = ['wse', 'width']
            
        stats = {}
        
        # Filter by quality if available
        if quality_threshold is not None and 'wse_qual' in df.columns:
            df = df[df.wse_qual < quality_threshold].copy()
            
        for field in fields:
            if field not in df.columns:
                continue
                
            series = df[field].dropna()
            if len(series) == 0:
                continue
                
            # Calculate basic statistics
            stats[field] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                "n_observations": len(series)
            }
            
            # Add uncertainty stats if available
            uncertainty_field = f"{field}_u"
            if uncertainty_field in df.columns:
                uncertainty = df[uncertainty_field].dropna()
                if len(uncertainty) > 0:
                    stats[field].update({
                        "mean_uncertainty": float(uncertainty.mean()),
                        "max_uncertainty": float(uncertainty.max())
                    })
                    
        return stats

    def plot_timeseries(
        self,
        df: pd.DataFrame,
        quality_threshold: Optional[int] = None,
        show_uncertainty: bool = True,
        feature_type: Optional[str] = None,
        feature_id: Optional[str] = None
    ) -> go.Figure:
        """Plot time series data"""
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Handle empty DataFrame
        if df.empty:
            return fig
        
        try:
            # Convert time strings to datetime if column exists
            if 'time_str' in df.columns:
                df['datetime'] = pd.to_datetime(df['time_str'])
            else:
                # Create dummy time index if no time column
                df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df))
            
            # Add WSE trace if available
            if 'wse' in df.columns:
                mask = df['wse'].notna()
                if quality_threshold is not None and 'quality' in df.columns:
                    mask &= df['quality'] >= quality_threshold
                    
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=df.loc[mask, 'datetime'],
                            y=df.loc[mask, 'wse'],
                            name='Water Surface Elevation',
                            mode='lines+markers'
                        )
                    )
                    
            # Add width trace if available
            if 'width' in df.columns:
                mask = df['width'].notna()
                if quality_threshold is not None and 'quality' in df.columns:
                    mask &= df['quality'] >= quality_threshold
                    
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=df.loc[mask, 'datetime'],
                            y=df.loc[mask, 'width'],
                            name='Width',
                            mode='lines+markers',
                            yaxis='y2'
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to plot timeseries: {str(e)}")
        
        # Update layout
        title_parts = []
        if feature_type and feature_id:
            title_parts.append(f"{feature_type} {feature_id}")
        if quality_threshold is not None:
            title_parts.append(f"(Quality < {quality_threshold})")
            
        fig.update_layout(
            title=" ".join(title_parts) if title_parts else None,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Water Surface Elevation (m)"),
            yaxis2=dict(
                title="River Width (m)",
                overlaying='y',
                side='right'
            ),
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig

    def analyze_and_plot(
        self,
        feature_type: Union[FeatureType, str],
        feature_id: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        quality_threshold: Optional[int] = None,
        show_uncertainty: bool = True
    ) -> Tuple[Dict[str, Dict[str, float]], go.Figure]:
        """
        Analyze and plot time series data in one step
        
        Args:
            feature_type: Feature type (Reach or Node)
            feature_id: Feature ID
            start_time: Start time
            end_time: End time
            quality_threshold: Optional quality threshold
            show_uncertainty: Whether to show uncertainty bands
            
        Returns:
            Tuple of (statistics, plot)
        """
        # Convert feature type if string
        if isinstance(feature_type, str):
            feature_type = FeatureType(feature_type)
            
        # Get data
        df = self.get_timeseries(
            feature=feature_type,
            feature_id=feature_id,
            start_time=start_time,
            end_time=end_time,
            fields=HydrocronField.default_fields()
        )
        
        # Calculate stats
        stats = self.analyze_timeseries(df, quality_threshold=quality_threshold)
        
        # Create plot
        fig = self.plot_timeseries(
            df,
            quality_threshold=quality_threshold,
            show_uncertainty=show_uncertainty,
            feature_type=feature_type.value,
            feature_id=feature_id
        )
        
        return stats, fig 

    def _format_params(
        self,
        feature: Union[str, FeatureType],
        feature_id: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        fields: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Format request parameters for API call"""
        # Convert feature type if needed
        if isinstance(feature, FeatureType):
            feature = feature.value
        elif isinstance(feature, str):
            try:
                feature = FeatureType(feature).value
            except ValueError:
                raise HydrocronValidationError(f"Invalid feature type: {feature}")

        # Format times with Z instead of +00:00
        if isinstance(start_time, datetime):
            start_time = start_time.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        if isinstance(end_time, datetime):
            end_time = end_time.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

        # Use simpler fields list without units
        if fields is None:
            fields = ['reach_id', 'time_str', 'wse', 'width']

        # Let requests handle URL encoding
        return {
            "feature": feature,
            "feature_id": str(feature_id),
            "start_time": start_time,
            "end_time": end_time,
            "fields": ",".join(fields)
        }

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers"""
        headers = {
            'Accept': 'application/json',  # Changed from application/geo+json
            'Content-Type': 'application/json'
        }
        
        if self.config.api_key:
            headers['x-hydrocron-key'] = self.config.api_key
        
        return headers

class HydrocronConfig:
    """Configuration for Hydrocron client"""
    def __init__(
        self,
        base_url: str = BASE_URL,  # Use the BASE_URL constant as default
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout 