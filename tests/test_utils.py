import pytest
from datetime import datetime, timedelta
import time

from hydrocron_wrapper.utils import RateLimiter, ResponseCache, RateLimitExceededError

def test_rate_limiter():
    """Test rate limiter functionality"""
    limiter = RateLimiter(max_requests=2, time_window=1)
    
    # First two requests should succeed
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    
    # Third request within window should fail
    assert limiter.acquire() is False
    
    # Wait for window to pass
    time.sleep(1.1)
    
    # Should succeed again
    assert limiter.acquire() is True

def test_response_cache():
    """Test response cache functionality"""
    cache = ResponseCache(max_size=2)
    
    # Test setting and getting
    cache.set("url1", {"param": "1"}, "content1", "text/plain")
    result = cache.get("url1", {"param": "1"})
    assert result is not None
    assert result[0] == "content1"
    assert result[1] == "text/plain"
    
    # Test cache miss with different params
    assert cache.get("url1", {"param": "2"}) is None
    
    # Test cache eviction
    cache.set("url2", {}, "content2", "text/plain")
    cache.set("url3", {}, "content3", "text/plain")  # Should evict url1
    assert cache.get("url1", {"param": "1"}) is None
    
    # Test cache expiry
    cache = ResponseCache(max_size=1, ttl=0.1)
    cache.set("url1", {}, "content1", "text/plain")
    time.sleep(0.2)
    assert cache.get("url1", {}) is None

def test_rate_limiter_edge_cases():
    """Test rate limiter edge cases"""
    # Test zero requests
    limiter = RateLimiter(max_requests=0)
    assert limiter.acquire() is False
    
    # Test negative window
    with pytest.raises(ValueError):
        RateLimiter(time_window=-1)
    
    # Test very large window
    limiter = RateLimiter(time_window=3600)
    assert limiter.acquire() is True

def test_response_cache_edge_cases():
    """Test response cache edge cases"""
    cache = ResponseCache()
    
    # Test None values
    cache.set("url1", None, None, None)
    assert cache.get("url1", None) == (None, None)
    
    # Test empty values
    cache.set("url2", {}, "", "")
    assert cache.get("url2", {}) == ("", "")
    
    # Test invalid max_size
    with pytest.raises(ValueError):
        ResponseCache(max_size=0)
    
    # Test invalid TTL
    with pytest.raises(ValueError):
        ResponseCache(ttl=-1) 