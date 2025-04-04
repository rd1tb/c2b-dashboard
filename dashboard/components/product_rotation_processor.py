"""
Optimized product rotation metrics calculation with improved caching.
Calculates both variance types in a single pass to reduce cache storage and computation.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict
import hashlib
from pathlib import Path
import pickle

# Set up logging
logger = logging.getLogger(__name__)


class ProductRotationCache:
    """Cache for processed product rotation data."""
    
    def __init__(self, cache_dir: str = "cache/product_rotation", max_age_hours: int = 24):
        """
        Initialize the product rotation cache.
        
        Args:
            cache_dir: Directory to store cached results
            max_age_hours: Maximum age of cached results in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_hours = max_age_hours
    
    def _get_cache_key(self, years: List[int], category_ids: List[int], 
                      top_n: int, sort_by: str, include_discounts: bool) -> str:
        """
        Generate a unique cache key based on filter parameters.
        Variance type is no longer part of the key since we store both metrics.
        
        Args:
            years: List of selected years
            category_ids: List of selected category IDs
            top_n: Number of products to include
            sort_by: Sort order for product selection
            include_discounts: Whether discounts are included
            
        Returns:
            Unique cache key
        """
        # Create a reproducible string representation of parameters
        discount_str = "with_disc" if include_discounts else "no_disc"
        key_str = (f"product_rotation_"
                  f"{'_'.join(map(str, sorted(years)))}_"
                  f"{'_'.join(map(str, sorted(category_ids)))}_"
                  f"{top_n}_{sort_by}_{discount_str}")
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, years: List[int], category_ids: List[int], 
           top_n: int, sort_by: str, include_discounts: bool) -> Optional[pd.DataFrame]:
        """
        Retrieve cached product rotation data if available.
        
        Args:
            years: List of selected years
            category_ids: List of selected category IDs
            top_n: Number of products to include
            sort_by: Sort order for product selection
            include_discounts: Whether discounts are included
            
        Returns:
            Cached DataFrame or None if not available
        """
        key = self._get_cache_key(years, category_ids, top_n, sort_by, include_discounts)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        # Load cached result
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
            return None
    
    def set(self, years: List[int], category_ids: List[int], 
           top_n: int, sort_by: str, include_discounts: bool, result: pd.DataFrame) -> None:
        """
        Cache product rotation data.
        
        Args:
            years: List of selected years
            category_ids: List of selected category IDs
            top_n: Number of products to include
            sort_by: Sort order for product selection
            include_discounts: Whether discounts are included
            result: Processed DataFrame to cache
        """
        key = self._get_cache_key(years, category_ids, top_n, sort_by, include_discounts)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Cached product rotation data for key: {key}")
        except Exception as e:
            logger.warning(f"Error caching product rotation data: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error removing cache file {cache_file}: {str(e)}")


def optimize_product_sales_data(product_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process and optimize the product sales dataframe for further analysis.
    
    Args:
        product_sales: DataFrame containing product sales data
        
    Returns:
        Optimized DataFrame
    """
    if product_sales.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original DataFrame
    processed_df = product_sales.copy()
    
    # Ensure order_date is datetime
    if not pd.api.types.is_datetime64_dtype(processed_df['order_date']):
        processed_df['order_date'] = pd.to_datetime(processed_df['order_date'])
    
    # Ensure numeric columns are properly converted to float to avoid Decimal issues
    numeric_columns = ['base_row_total_incl_tax', 'base_discount_amount', 'qty_ordered', 'base_price']
    for col in numeric_columns:
        if col in processed_df.columns:
            # Handle potential decimal.Decimal objects
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0.0)
    
    # Calculate sales_amount if not present
    if 'sales_amount' not in processed_df.columns:
        processed_df['sales_amount'] = processed_df['base_row_total_incl_tax'] - processed_df['base_discount_amount']
    
    return processed_df


def calculate_product_rotation_metrics(
    product_sales: pd.DataFrame, 
    top_n: int = 50, 
    sort_by: str = "random", 
    variance_type: str = "days",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Calculate product rotation metrics with efficient algorithm and caching.
    Now calculates both variance types at once and selects the one requested.
    
    Args:
        product_sales: DataFrame containing the filtered sales data
        top_n: Number of products to include
        sort_by: How to select products ("ascending" for top revenue, "descending" for lowest revenue, "random" for random selection)
        variance_type: Type of variance to display ("days" for order timing variance, "quantity" for quantity variance)
        use_cache: Whether to use cached results
        
    Returns:
        DataFrame with rotation metrics for each product
    """
    if product_sales.empty:
        return pd.DataFrame()
    
    # Extract filter parameters for cache key
    if 'order_date' in product_sales.columns:
        years = sorted(product_sales['order_date'].dt.year.unique().tolist())
    else:
        years = []
        
    if 'category_num' in product_sales.columns:
        category_ids = sorted(product_sales['category_num'].unique().tolist())
    else:
        category_ids = []
    
    # Determine if discounts are included
    include_discounts = True
    if 'base_discount_amount' in product_sales.columns:
        # Check if all discount amounts are 0
        if (product_sales['base_discount_amount'] == 0).all():
            include_discounts = False
    
    # Check cache first if enabled
    cache = ProductRotationCache()
    if use_cache:
        cached_result = cache.get(years, category_ids, top_n, sort_by, include_discounts)
        if cached_result is not None:
            logger.info(f"Using cached product rotation data for years={years}, categories={category_ids}")
            
            # Select the appropriate coefficient of variation based on requested variance type
            if variance_type == "days":
                cached_result['coefficient_of_variation'] = cached_result['days_coefficient_of_variation']
                cached_result['std_deviation'] = cached_result['days_std']
            else:  # quantity
                cached_result['coefficient_of_variation'] = cached_result['quantity_coefficient_of_variation']
                cached_result['std_deviation'] = cached_result['quantity_std']
                
            return cached_result
    
    # Optimize the dataframe
    product_sales = optimize_product_sales_data(product_sales)
    
    # Process 1: Calculate total revenue per product
    product_revenue_df = product_sales.groupby('sku').agg({
        'sales_amount': 'sum',
        'base_price': 'mean'
    }).reset_index()
    
    product_revenue_df.rename(columns={'sales_amount': 'total_revenue'}, inplace=True)
    
    # Select products based on sorting option
    if sort_by == "ascending":
        # Top revenue products 
        selected_products = product_revenue_df.sort_values(by='total_revenue', ascending=False).head(top_n)
    elif sort_by == "descending":
        # Worst revenue products
        selected_products = product_revenue_df.sort_values(by='total_revenue', ascending=True).head(top_n)
    else:  # random
        # Random selection of products
        if len(product_revenue_df) <= top_n:
            selected_products = product_revenue_df
        else:
            selected_products = product_revenue_df.sample(n=top_n, random_state=42)
    
    selected_product_skus = selected_products['sku'].tolist()
    
    # Filter sales data for selected products
    selected_product_sales = product_sales[product_sales['sku'].isin(selected_product_skus)]
    
    # Process 2: Calculate metrics efficiently with a dictionary-based approach
    # Initialize dictionaries to store aggregated data per SKU
    results = {}
    
    # Group data by SKU for efficient iteration
    grouped_sales = selected_product_sales.groupby('sku')
    
    for sku, product_data in grouped_sales:
        # Sort by order date for day calculations
        product_data = product_data.sort_values(by='order_date')
        
        # Get total revenue 
        total_revenue = product_data['sales_amount'].sum()
        
        # Get average base price
        avg_base_price = product_data['base_price'].mean()
        
        # Order count
        order_count = len(product_data)
        
        # Calculate average quantity per order
        avg_qty = product_data['qty_ordered'].mean()
        
        # Only calculate days between orders if we have multiple orders
        if len(product_data) > 1:
            # Calculate days difference between consecutive orders
            date_diffs = product_data['order_date'].diff().dt.days.dropna()
            date_diffs = date_diffs[date_diffs > 0]  # Remove any zero or negative values
            
            if len(date_diffs) > 0:
                avg_days = date_diffs.mean()
                days_std = date_diffs.std() if len(date_diffs) > 1 else 0
                days_coefficient_of_variation = (days_std / avg_days) if avg_days > 0 and len(date_diffs) > 1 else 0
                
                # Store days differences for distribution analysis
                days_differences = date_diffs.tolist()
            else:
                avg_days = np.nan
                days_std = np.nan
                days_coefficient_of_variation = np.nan
                days_differences = []
            
            # Calculate quantity standard deviation
            qty_ordered = product_data['qty_ordered']
            if len(qty_ordered) > 1:
                qty_std = qty_ordered.std()
                qty_coefficient_of_variation = (qty_std / avg_qty) if avg_qty > 0 else 0
            else:
                qty_std = 0
                qty_coefficient_of_variation = 0
                
            quantities = qty_ordered.tolist()
        else:
            # Only one order, can't calculate days between orders or quantity variance
            avg_days = np.nan
            days_std = np.nan
            days_coefficient_of_variation = np.nan
            qty_std = np.nan
            qty_coefficient_of_variation = np.nan
            days_differences = []
            quantities = []
        
        # Choose coefficient of variation based on variance type
        # We'll store both but select the correct one after cache retrieval
        if variance_type == "days":
            coefficient_of_variation = days_coefficient_of_variation
            std_deviation = days_std
        else:  # quantity
            coefficient_of_variation = qty_coefficient_of_variation
            std_deviation = qty_std
        
        # Only add products with valid metrics
        if not np.isnan(avg_days) and not np.isnan(avg_qty):
            results[sku] = {
                'sku': sku,
                'avg_days_between_orders': avg_days,
                'avg_quantity_per_order': avg_qty,
                'days_std': days_std,
                'quantity_std': qty_std,
                'days_coefficient_of_variation': days_coefficient_of_variation,
                'quantity_coefficient_of_variation': qty_coefficient_of_variation,
                'coefficient_of_variation': coefficient_of_variation,  # Current selected variance
                'std_deviation': std_deviation,  # Current selected std deviation
                'total_revenue': total_revenue,
                'base_price': avg_base_price,
                'order_count': order_count,
                'days_differences': days_differences,
                'quantities': quantities
            }
    
    # Convert to DataFrame
    result_df = pd.DataFrame(list(results.values()))
    
    # Filter to remove NaN values if any slipped through
    result_df = result_df.dropna(subset=['avg_days_between_orders', 'avg_quantity_per_order'])
    
    # Cache the result if enabled - store both variance types
    if use_cache and not result_df.empty:
        cache.set(years, category_ids, top_n, sort_by, include_discounts, result_df)
    
    return result_df