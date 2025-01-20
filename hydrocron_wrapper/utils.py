"""Utility functions for the Hydrocron API client"""
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import os
import logging
import time
from collections import OrderedDict

# Set up logging
logger = logging.getLogger(__name__)

def get_planet_quarter(date: datetime) -> str:
    """Convert date to Planet quarterly basemap format (e.g., '2024Q1')"""
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f"{year}Q{quarter}"

def get_planet_basemap_url(api_key: Optional[str] = None, quarter: str = "2024Q1") -> str:
    """
    Construct Planet basemap URL with API key
    
    Args:
        api_key: Planet API key (defaults to PLANET_API_KEY env var)
        quarter: Quarter in format YYYYQN (e.g., "2024Q1")
        
    Returns:
        URL template for Planet basemap tiles
        
    Raises:
        ValueError: If no API key is provided or found in environment
    """
    api_key = api_key or os.getenv('PLANET_API_KEY')
    if not api_key:
        raise ValueError("Planet API key not found. Set PLANET_API_KEY environment variable or pass key directly.")
        
    return f"https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_{quarter}_mosaic/{{z}}/{{x}}/{{y}}.png?api_key={api_key}"

def get_mapbox_layout(
    center_lat: float,
    center_lon: float,
    zoom: int = 12,
    style: str = "carto-positron",
    planet_key: Optional[str] = None,
    planet_quarter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get Mapbox layout settings with optional Planet basemap
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        zoom: Zoom level (1-20)
        style: Base map style (ignored if using Planet)
        planet_key: Optional Planet API key
        planet_quarter: Optional Planet quarter (e.g., "2024Q1")
        
    Returns:
        Mapbox layout configuration
    """
    layout = {
        "mapbox": {
            "style": style,
            "zoom": zoom,
            "center": {"lat": center_lat, "lon": center_lon}
        }
    }
    
    # Add Planet basemap if credentials provided
    if planet_key and planet_quarter:
        try:
            layout["mapbox"].update({
                "style": "white-bg",  # Blank background for satellite
                "layers": [{
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [get_planet_basemap_url(planet_key, planet_quarter)],
                    "type": "raster",
                    "opacity": 1
                }]
            })
        except ValueError as e:
            logger.warning(f"Failed to add Planet basemap: {str(e)}")
            
    return layout

def clean_reach_id(reach_id: str) -> str:
    """Convert reach ID to proper integer string format"""
    return str(int(float(str(reach_id)))) 

class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded"""
    pass

class RateLimiter:
    """Simple rate limiter using sliding window"""
    
    def __init__(self, max_requests: int = 10, time_window: float = 1.0):
        """Initialize rate limiter
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
        """
        if time_window <= 0:
            raise ValueError("time_window must be positive")
            
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def acquire(self) -> bool:
        """Try to acquire a request slot
        
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        
        # Remove expired requests
        self.requests = [t for t in self.requests if now - t < self.time_window]
        
        # Check if we can make another request
        if len(self.requests) >= self.max_requests:
            return False
            
        # Add current request
        self.requests.append(now)
        return True

class ResponseCache:
    """Simple cache for API responses"""
    
    def __init__(self, max_size: int = 100, ttl: float = 300.0):
        """Initialize cache
        
        Args:
            max_size: Maximum number of cached responses
            ttl: Time-to-live in seconds
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl < 0:
            raise ValueError("ttl must be non-negative")
            
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Tuple[float, Any, str]] = OrderedDict()
    
    def _make_key(self, url: str, params: Optional[Dict[str, Any]]) -> str:
        """Make cache key from URL and params"""
        if params is None:
            params = {}
        return f"{url}:{str(sorted(params.items()))}"
    
    def get(self, url: str, params: Optional[Dict[str, Any]]) -> Optional[Tuple[Any, str]]:
        """Get cached response if available and not expired
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            Tuple of (content, content_type) if found, None otherwise
        """
        key = self._make_key(url, params)
        if key not in self.cache:
            return None
            
        timestamp, content, content_type = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
            
        return content, content_type
    
    def set(self, url: str, params: Optional[Dict[str, Any]], content: Any, content_type: str):
        """Cache a response
        
        Args:
            url: Request URL
            params: Request parameters
            content: Response content
            content_type: Response content type
        """
        key = self._make_key(url, params)
        
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)
            
        self.cache[key] = (time.time(), content, content_type) 