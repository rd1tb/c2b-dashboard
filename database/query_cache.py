import hashlib
import pandas as pd
from typing import Optional, Tuple, Any
from pathlib import Path
import logging
import pickle
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class QueryCache:
    """Cache for database query results."""
    
    def __init__(self, cache_dir: str = "cache", max_age_hours: int = 24):
        """
        Initialize the query cache.
        
        Args:
            cache_dir: Directory to store cached results
            max_age_hours: Maximum age of cached results in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
    
    def _get_cache_key(self, key_str: str, params: Tuple = None) -> str:
        """
        Generate a unique cache key.
        
        Args:
            key_str: String to use as cache key base (could be a query or custom string)
            params: Query parameters
            
        Returns:
            Unique cache key
        """
        # Handle case where key_str is already a prepared cache key
        if params is None and "_" not in key_str:
            return hashlib.md5(key_str.encode()).hexdigest()
            
        # Otherwise, create a unique key based on the string and parameters
        complete_key = f"{key_str}_{str(params) if params else ''}"
        return hashlib.md5(complete_key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key_str: str, params: Tuple = None) -> Optional[pd.DataFrame]:
        """
        Retrieve cached query result if available and not expired.
        
        Args:
            key_str: String to use as cache key base
            params: Query parameters
            
        Returns:
            Cached DataFrame or None if not available
        """
        key = self._get_cache_key(key_str, params)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mod_time > self.max_age:
            logger.info(f"Cache expired for key: {key_str[:50]}...")
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
            return None
    
    def set(self, key_str: str, params: Tuple, result: pd.DataFrame) -> None:
        """
        Cache a query result.
        
        Args:
            key_str: String to use as cache key base
            params: Query parameters
            result: Query result DataFrame
        """
        key = self._get_cache_key(key_str, params)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Cached result for key: {key_str[:50]}...")
        except Exception as e:
            logger.warning(f"Error caching result: {str(e)}")
    
    def invalidate(self, key_str: str = None, params: Tuple = None) -> None:
        """
        Invalidate cache for a specific query or all caches.
        
        Args:
            key_str: String to use as cache key base (all if None)
            params: Query parameters
        """
        if key_str is None:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)
            logger.info("All query caches invalidated")
        else:
            key = self._get_cache_key(key_str, params)
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink(missing_ok=True)
                logger.info(f"Cache invalidated for key: {key_str[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache files."""
        logger.info("Clearing all cache files")
        self.invalidate()