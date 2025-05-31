"""
Optimized product rotation metrics calculation with improved caching.
Updated to include customer filtering in the cache key.
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
                      top_n: int, sort_by: str, include_discounts: bool,
                      customer_types: Optional[List[str]] = None,
                      customer_status: Optional[List[str]] = None) -> str:
        """
        Generate a unique cache key based on filter parameters.
        
        Args:
            years: List of selected years
            category_ids: List of selected category IDs
            top_n: Number of products to include
            sort_by: Sort order for product selection
            include_discounts: Whether discounts are included
            customer_types: List of customer types (Guest, Registered)
            customer_status: List of customer status (New, Returning)
            
        Returns:
            Unique cache key
        """
        discount_str = "with_disc" if include_discounts else "no_disc"
        customer_type_str = "_".join(sorted(customer_types)) if customer_types else "all_types"
        customer_status_str = "_".join(sorted(customer_status)) if customer_status else "all_status"
        key_str = (f"product_rotation_"
                  f"{'_'.join(map(str, sorted(years)))}_"
                  f"{'_'.join(map(str, sorted(category_ids)))}_"
                  f"{top_n}_{sort_by}_{discount_str}_"
                  f"{customer_type_str}_{customer_status_str}")
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, years: List[int], category_ids: List[int], 
           top_n: int, sort_by: str, include_discounts: bool,
           customer_types: Optional[List[str]] = None,
           customer_status: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve cached product rotation data if available.
        
        Args:
            years: List of selected years
            category_ids: List of selected category IDs
            top_n: Number of products to include
            sort_by: Sort order for product selection
            include_discounts: Whether discounts are included
            customer_types: List of customer types (Guest, Registered)
            customer_status: List of customer status (New, Returning)
            
        Returns:
            Cached DataFrame or None if not available
        """
        key = self._get_cache_key(years, category_ids, top_n, sort_by, include_discounts, 
                                  customer_types, customer_status)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
            return None
    
    def set(self, years: List[int], category_ids: List[int], 
           top_n: int, sort_by: str, include_discounts: bool, 
           result: pd.DataFrame,
           customer_types: Optional[List[str]] = None,
           customer_status: Optional[List[str]] = None) -> None:
        """
        Cache product rotation data.
        
        Args:
            years: List of selected years
            category_ids: List of selected category IDs
            top_n: Number of products to include
            sort_by: Sort order for product selection
            include_discounts: Whether discounts are included
            result: Processed DataFrame to cache
            customer_types: List of customer types (Guest, Registered)
            customer_status: List of customer status (New, Returning)
        """
        key = self._get_cache_key(years, category_ids, top_n, sort_by, include_discounts,
                                 customer_types, customer_status)
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
    processed_df = product_sales.copy()
    if not pd.api.types.is_datetime64_dtype(processed_df['order_date']):
        processed_df['order_date'] = pd.to_datetime(processed_df['order_date'])
    numeric_columns = ['base_row_total_incl_tax', 'base_discount_amount', 'qty_ordered', 'base_price']
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0.0)
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
    Updated to handle customer filtering in cache keys and retain zero-day differences.
    
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
    if 'order_date' in product_sales.columns:
        years = sorted(product_sales['order_date'].dt.year.unique().tolist())
    else:
        years = []
    if 'category_num' in product_sales.columns:
        category_ids = sorted(product_sales['category_num'].unique().tolist())
    else:
        category_ids = []
    include_discounts = True
    if 'base_discount_amount' in product_sales.columns:
        if (product_sales['base_discount_amount'] == 0).all():
            include_discounts = False
    customer_types = None
    if 'customer_is_guest' in product_sales.columns or 'customer_group_id' in product_sales.columns:
        has_guest = False
        if 'customer_is_guest' in product_sales.columns:
            has_guest = (product_sales['customer_is_guest'] == 1).any()
        has_registered = False
        if 'customer_group_id' in product_sales.columns:
            has_registered = product_sales['customer_group_id'].isin([1, 2, 7, 8, 9]).any()
        customer_types = []
        if has_guest:
            customer_types.append('Guest')
        if has_registered:
            customer_types.append('Registered')
    customer_status = None
    if 'hashed_customer_email' in product_sales.columns:
        customer_order_counts = product_sales.groupby('hashed_customer_email')['order_id'].nunique().reset_index()
        customer_order_counts.columns = ['hashed_customer_email', 'unique_order_count']
        has_new = (customer_order_counts['unique_order_count'] == 1).any()
        has_returning = (customer_order_counts['unique_order_count'] > 1).any()
        customer_status = []
        if has_new:
            customer_status.append('New')
        if has_returning:
            customer_status.append('Returning')
    cache = ProductRotationCache()
    if use_cache:
        cached_result = cache.get(years, category_ids, top_n, sort_by, include_discounts,
                                 customer_types, customer_status)
        if cached_result is not None:
            logger.info(f"Using cached product rotation data for years={years}, categories={category_ids}, customer_types={customer_types}")
            if variance_type == "days":
                cached_result['coefficient_of_variation'] = cached_result['days_coefficient_of_variation']
                cached_result['std_deviation'] = cached_result['days_std']
            else:
                cached_result['coefficient_of_variation'] = cached_result['quantity_coefficient_of_variation']
                cached_result['std_deviation'] = cached_result['quantity_std']
            return cached_result
    product_sales = optimize_product_sales_data(product_sales)
    product_revenue_df = product_sales.groupby('sku').agg({
        'sales_amount': 'sum',
        'base_price': 'mean'
    }).reset_index()
    product_revenue_df.rename(columns={'sales_amount': 'total_revenue'}, inplace=True)
    if sort_by == "ascending":
        selected_products = product_revenue_df.sort_values(by='total_revenue', ascending=False).head(top_n)
    elif sort_by == "descending":
        selected_products = product_revenue_df.sort_values(by='total_revenue', ascending=True).head(top_n)
    else:
        if len(product_revenue_df) <= top_n:
            selected_products = product_revenue_df
        else:
            selected_products = product_revenue_df.sample(n=top_n, random_state=42)
    selected_product_skus = selected_products['sku'].tolist()
    selected_product_sales = product_sales[product_sales['sku'].isin(selected_product_skus)]
    results = {}
    grouped_sales = selected_product_sales.groupby('sku')
    for sku, product_data in grouped_sales:
        product_data = product_data.sort_values(by='order_date')
        total_revenue = product_data['sales_amount'].sum()
        avg_base_price = product_data['base_price'].mean()
        order_count = len(product_data)
        avg_qty = product_data['qty_ordered'].mean()
        if len(product_data) > 1:
            date_diffs = product_data['order_date'].diff().dt.days.dropna()
            date_diffs = date_diffs[date_diffs >= 0]
            if len(date_diffs) > 0:
                avg_days = date_diffs.mean()
                days_std = date_diffs.std() if len(date_diffs) > 1 else 0
                if avg_days > 0 and len(date_diffs) > 1:
                    days_coefficient_of_variation = days_std / avg_days
                else:
                    days_coefficient_of_variation = 0
                days_differences = date_diffs.tolist()
            else:
                avg_days = np.nan
                days_std = np.nan
                days_coefficient_of_variation = np.nan
                days_differences = []
            qty_ordered = product_data['qty_ordered']
            if len(qty_ordered) > 1:
                qty_std = qty_ordered.std()
                if avg_qty > 0:
                    qty_coefficient_of_variation = qty_std / avg_qty
                else:
                    qty_coefficient_of_variation = 0
            else:
                qty_std = 0
                qty_coefficient_of_variation = 0
            quantities = qty_ordered.tolist()
        else:
            avg_days = np.nan
            days_std = np.nan
            days_coefficient_of_variation = np.nan
            qty_std = np.nan
            qty_coefficient_of_variation = np.nan
            days_differences = []
            quantities = []
        if variance_type == "days":
            coefficient_of_variation = days_coefficient_of_variation
            std_deviation = days_std
        else:
            coefficient_of_variation = qty_coefficient_of_variation
            std_deviation = qty_std
        if (avg_days is not None and not np.isnan(avg_days)) and (avg_qty is not None and not np.isnan(avg_qty)):
            results[sku] = {
                'sku': sku,
                'avg_days_between_orders': avg_days,
                'avg_quantity_per_order': avg_qty,
                'days_std': days_std,
                'quantity_std': qty_std,
                'days_coefficient_of_variation': days_coefficient_of_variation,
                'quantity_coefficient_of_variation': qty_coefficient_of_variation,
                'coefficient_of_variation': coefficient_of_variation,
                'std_deviation': std_deviation,
                'total_revenue': total_revenue,
                'base_price': avg_base_price,
                'order_count': order_count,
                'days_differences': days_differences,
                'quantities': quantities
            }
    result_df = pd.DataFrame(list(results.values()))
    result_df = result_df.dropna(subset=['avg_days_between_orders', 'avg_quantity_per_order'])
    if use_cache and not result_df.empty:
        cache.set(years, category_ids, top_n, sort_by, include_discounts, result_df, 
                 customer_types, customer_status)
    return result_df
