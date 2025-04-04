"""
Database query execution module.
Provides functions to execute queries with caching.
"""
import pandas as pd
from typing import Optional, Tuple, Any
import logging
from .connection import DatabaseConnection
from .query_cache import QueryCache

# Set up logging
logger = logging.getLogger(__name__)


class QueryExecutor:
    """Execute database queries with caching support."""
    
    def __init__(self, db_connection: DatabaseConnection, use_cache: bool = True):
        """
        Initialize database query executor.
        
        Args:
            db_connection: Database connection
            use_cache: Whether to use query cache
        """
        self.db_connection = db_connection
        self.use_cache = use_cache
        self.cache = QueryCache() if use_cache else None
    
    def execute_query(self, query: str, params: Tuple = None, use_cache: Optional[bool] = None, cache_key: Optional[str] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            use_cache: Override instance cache setting for this query
            cache_key: Optional custom key for caching (useful for parameterized queries)
            
        Returns:
            DataFrame with query results
        """
        # Determine if cache should be used for this query
        should_use_cache = self.use_cache if use_cache is None else use_cache
        
        # Check cache first if enabled
        if should_use_cache:
            # If a cache_key is provided, use it to generate a deterministic cache key
            if cache_key:
                cache_key_str = f"{cache_key}_{str(params) if params else ''}"
                cached_result = self.cache.get(cache_key_str, None)
            else:
                cached_result = self.cache.get(query, params)
                
            if cached_result is not None:
                logger.info(f"Using cached result for query: {cache_key or query[:50]}...")
                return cached_result
        
        # Execute query if no cache hit
        try:
            cursor = self.db_connection.get_cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            # Convert results to DataFrame
            results = cursor.fetchall()
            df = pd.DataFrame(results)
            
            # Cache results if enabled
            if should_use_cache and not df.empty:
                if cache_key:
                    cache_key_str = f"{cache_key}_{str(params) if params else ''}"
                    self.cache.set(cache_key_str, None, df)
                else:
                    self.cache.set(query, params, df)
                
            cursor.close()
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def execute_non_query(self, query: str, params: Tuple = None) -> int:
        """
        Execute a non-query SQL statement (INSERT, UPDATE, DELETE).
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        try:
            cursor = self.db_connection.get_cursor()
            cursor.execute(query, params or ())
            self.db_connection.connection.commit()
            
            # Invalidate cache because data has changed
            if self.use_cache:
                self.cache.invalidate()
            
            affected_rows = cursor.rowcount
            cursor.close()
            return affected_rows
            
        except Exception as e:
            self.db_connection.connection.rollback()
            logger.error(f"Error executing non-query: {str(e)}")
            raise