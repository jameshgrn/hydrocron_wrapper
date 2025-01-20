"""Configuration constants for the Hydrocron API client"""

# Production base URL
PRODUCTION_BASE_URL = "https://soto.podaac.sit.earthdatacloud.nasa.gov/hydrocron/v1"

# Default timeout in seconds
DEFAULT_TIMEOUT = 30

# Rate limiting defaults
DEFAULT_RATE = 10.0  # requests per second
DEFAULT_BURST = 20   # maximum burst size

# Cache configuration
DEFAULT_CACHE_TTL_HOURS = 1
DEFAULT_CACHE_DIR = "~/.hydrocron/cache" 