"""
Final Fixed Combined Promotion Analysis Component
Completely redesigned to properly handle all filters, slider values, and metric calculations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from pathlib import Path
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
import time

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO


class PromotionAnalysisCache:
    """Cache for processed promotion analysis data."""
    
    def __init__(self, cache_dir: str = "cache/promotion_analysis", max_age_hours: int = 24):
        """
        Initialize the promotion analysis cache.
        
        Args:
            cache_dir: Directory to store cached results
            max_age_hours: Maximum age of cached results in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, filter_params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key based on filter parameters.
        
        Args:
            filter_params: Dictionary of filter parameters
            
        Returns:
            Unique cache key
        """
        # Sort the keys to ensure consistent cache keys
        sorted_params = sorted(filter_params.items())
        
        # Convert filter params to string representation
        param_strings = []
        for key, value in sorted_params:
            if isinstance(value, list):
                value_str = "_".join(map(str, sorted(value) if value else []))
            else:
                value_str = str(value) if value is not None else ""
            param_strings.append(f"{key}:{value_str}")
        
        key_str = f"promotion_analysis_{'__'.join(param_strings)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, filter_params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Retrieve cached promotion analysis data if available.
        
        Args:
            filter_params: Dictionary of filter parameters
            
        Returns:
            Cached DataFrame or None if not available
        """
        key = self._get_cache_key(filter_params)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        # Load cached result
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
                logger.info(f"Loaded cached result for key: {key}")
                return result
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
            return None
    
    def set(self, filter_params: Dict[str, Any], result: pd.DataFrame) -> None:
        """
        Cache promotion analysis data.
        
        Args:
            filter_params: Dictionary of filter parameters
            result: Processed DataFrame to cache
        """
        key = self._get_cache_key(filter_params)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"Cached promotion analysis data for key: {key}")
        except Exception as e:
            logger.warning(f"Error caching promotion analysis data: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                logger.info(f"Removed cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Error removing cache file {cache_file}: {str(e)}")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str], df_name: str) -> bool:
    """
    Validate that a dataframe has the required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name of the dataframe for logging
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning(f"{df_name} is empty or None")
        return False
        
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"{df_name} missing required columns: {missing_columns}")
        return False
        
    return True


def process_applied_rule_ids(rule_ids_str: str) -> List[int]:
    """
    Process applied rule IDs string into a list of integers.
    
    Args:
        rule_ids_str: Comma-separated string of rule IDs
        
    Returns:
        List of rule IDs as integers
    """
    if not rule_ids_str or pd.isna(rule_ids_str) or rule_ids_str == '':
        return []
    
    # Split by comma
    rule_id_parts = str(rule_ids_str).split(',')
    
    # Process each part - remove 'c-' prefix and convert to int
    rule_ids = []
    for part in rule_id_parts:
        try:
            # Remove 'c-' prefix if exists
            clean_part = part.strip()
            if clean_part.startswith('c-'):
                clean_part = clean_part[2:]
            
            # Convert to integer
            rule_id = int(clean_part)
            rule_ids.append(rule_id)
        except (ValueError, TypeError):
            # Skip invalid values
            continue
    
    return rule_ids


def apply_filters_to_sales(
    original_sales_df: pd.DataFrame,
    filter_years: List[int] = None,
    filter_category_ids: List[int] = None
) -> pd.DataFrame:
    """
    Apply filters to sales data.
    
    Args:
        original_sales_df: Original unfiltered sales DataFrame
        filter_years: Years to filter by
        filter_category_ids: Category IDs to filter by
        
    Returns:
        Filtered sales DataFrame
    """
    # Make a copy to avoid modifying the original
    filtered_df = original_sales_df.copy()
    
    # Apply year filter if provided
    if filter_years and len(filter_years) > 0 and 'order_date' in filtered_df.columns:
        # Convert order_date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['order_date']):
            filtered_df['order_date'] = pd.to_datetime(filtered_df['order_date'])
        
        # Filter by year
        filtered_df = filtered_df[filtered_df['order_date'].dt.year.isin(filter_years)]
        logger.info(f"Filtered sales data to years: {filter_years}, remaining rows: {len(filtered_df)}")
    
    # Apply category filter if provided
    if filter_category_ids and len(filter_category_ids) > 0 and 'category_num' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category_num'].isin(filter_category_ids)]
        logger.info(f"Filtered sales data to categories: {filter_category_ids}, remaining rows: {len(filtered_df)}")
    
    return filtered_df


def calculate_promotion_metrics(
    original_sales_df: pd.DataFrame, 
    promotion_df: pd.DataFrame,
    filter_years: List[int] = None,
    filter_category_ids: List[int] = None,
    max_promotions: int = 50,
    sort_option: str = "top_revenue",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Calculate comprehensive metrics by promotion rule ID.
    
    Args:
        original_sales_df: Original unfiltered sales DataFrame
        promotion_df: DataFrame with promotion details
        filter_years: Years to filter by
        filter_category_ids: Category IDs to filter by
        max_promotions: Maximum number of promotions to include
        sort_option: How to sort promotions (top_revenue, lowest_revenue, random)
        use_cache: Whether to use cached results
        
    Returns:
        DataFrame with metrics by promotion rule ID
    """
    # Create filter parameters dictionary for cache key
    filter_params = {
        'years': filter_years,
        'category_ids': filter_category_ids,
        'max_promotions': max_promotions,
        'sort_option': sort_option
    }
    
    start_time = time.time()
    
    # Check cache first if enabled
    cache = PromotionAnalysisCache()
    if use_cache:
        cached_result = cache.get(filter_params)
        if cached_result is not None:
            logger.info(f"Using cached promotion metrics data for filters: {filter_params}")
            logger.info(f"Retrieved from cache in {time.time() - start_time:.2f} seconds")
            return cached_result
    
    # Validate input dataframes
    sales_required_cols = ['applied_rule_ids', 'base_row_total_incl_tax', 'base_discount_amount', 'qty_ordered']
    if not validate_dataframe(original_sales_df, sales_required_cols, 'Sales DataFrame'):
        return pd.DataFrame()
        
    promotion_required_cols = ['rule_id', 'name', 'discount_amount', 'ctb_discount_used', 'ctb_units_used', 'from_date', 'to_date']
    if not validate_dataframe(promotion_df, promotion_required_cols, 'Promotion DataFrame'):
        return pd.DataFrame()
    
    # Apply filters to sales data
    sales_df = apply_filters_to_sales(original_sales_df, filter_years, filter_category_ids)
    
    if sales_df.empty:
        logger.warning("After filtering, sales DataFrame is empty")
        return pd.DataFrame()
    
    # Process promotion data
    # Filter to used promotions (those with actual discount usage)
    used_promotions = promotion_df[promotion_df['ctb_discount_used'] > 0].copy()
    
    # Handle empty used_promotions case
    if used_promotions.empty:
        logger.warning("No promotions with discount usage found")
        return pd.DataFrame()
    
    used_rule_ids = set(used_promotions['rule_id'].tolist())
    logger.info(f"Found {len(used_rule_ids)} promotions with discount usage")
    
    # Create dictionaries to store metrics by rule ID
    rule_revenue = {rule_id: 0.0 for rule_id in used_rule_ids}
    rule_orders = {rule_id: 0 for rule_id in used_rule_ids}
    rule_items = {rule_id: 0 for rule_id in used_rule_ids}
    rule_unique_skus = {rule_id: set() for rule_id in used_rule_ids}  # Using sets to track unique SKUs
    
    # Process sales data
    # Create a new column with processed rule IDs
    sales_df['rule_ids_list'] = sales_df['applied_rule_ids'].apply(process_applied_rule_ids)
    
    # Check if we have any rule IDs after processing
    if all(len(rules) == 0 for rules in sales_df['rule_ids_list']):
        logger.warning("No valid rule IDs found in sales data after processing")
    else:
        # Debug: Print a sample of processed rule IDs
        sample_rules = sales_df['rule_ids_list'].head(5).tolist()
        logger.info(f"Sample of processed rule IDs: {sample_rules}")
    
    # Count sales rows with valid rule IDs that match our promotions
    matching_sales_rows = sum(
        1 for rules in sales_df['rule_ids_list'] 
        if any(rule_id in used_rule_ids for rule_id in rules)
    )
    logger.info(f"Found {matching_sales_rows} sales rows with matching rule IDs")
    
    # Calculate metrics for each sale and distribute to applicable rules
    row_count = 0
    for _, row in sales_df.iterrows():
        row_count += 1
        
        # Extract the rule IDs for this sale
        sale_rule_ids = row['rule_ids_list']
        
        # Skip if no rules applied
        if len(sale_rule_ids) == 0:
            continue
        
        # Calculate revenue for this item - convert to float to handle Decimal type
        try:
            item_revenue = float(row['base_row_total_incl_tax']) - float(row['base_discount_amount'])
        except (ValueError, TypeError) as e:
            logger.warning(f"Error calculating revenue: {e} - Row: {row['base_row_total_incl_tax']}, {row['base_discount_amount']}")
            continue
            
        if item_revenue <= 0:
            continue
        
        # Get the used rules for this item that match our filter
        matching_rules = [rule_id for rule_id in sale_rule_ids if rule_id in used_rule_ids]
        
        # Skip if no matching rules
        if len(matching_rules) == 0:
            continue
        
        # Distribute revenue equally among rules (simplified approach)
        revenue_per_rule = item_revenue / len(matching_rules)
        qty_ordered = float(row['qty_ordered']) if pd.notna(row['qty_ordered']) else 0
        
        # Add to rule metrics
        for rule_id in matching_rules:
            rule_revenue[rule_id] += revenue_per_rule
            rule_orders[rule_id] += 1
            rule_items[rule_id] += qty_ordered
            
            # Add SKU to the set if it exists
            if 'sku' in row and pd.notna(row['sku']):
                rule_unique_skus[rule_id].add(str(row['sku']))
    
    logger.info(f"Processed {row_count} sales rows")
    
    # Convert sets of unique SKUs to counts
    rule_unique_sku_counts = {rule_id: len(skus) for rule_id, skus in rule_unique_skus.items()}
    
    # Create DataFrame from dictionaries
    metrics_df = pd.DataFrame({
        'rule_id': list(rule_revenue.keys()),
        'revenue': list(rule_revenue.values()),
        'order_count': list(rule_orders.values()),
        'item_count': list(rule_items.values()),
        'unique_sku_count': list(rule_unique_sku_counts.values())
    })
    
    # Debug: Check key metrics sizes
    logger.info(f"Generated metrics for {len(metrics_df)} promotions")
    if not metrics_df.empty:
        logger.info(f"Revenue range: {metrics_df['revenue'].min()} - {metrics_df['revenue'].max()}")
        logger.info(f"Unique SKU count range: {metrics_df['unique_sku_count'].min()} - {metrics_df['unique_sku_count'].max()}")
    
    # Merge with promotion details
    if not metrics_df.empty and not promotion_df.empty:
        # Convert rule_id to int in both DataFrames to ensure proper merging
        metrics_df['rule_id'] = metrics_df['rule_id'].astype(int)
        promotion_df_copy = promotion_df.copy()
        promotion_df_copy['rule_id'] = promotion_df_copy['rule_id'].astype(int)
        
        # Get columns to merge
        promotion_cols = ['rule_id', 'name', 'discount_amount', 'ctb_discount_used', 'ctb_units_used', 'from_date', 'to_date']
        promotion_cols = [col for col in promotion_cols if col in promotion_df_copy.columns]
        
        # Perform the merge
        result_df = pd.merge(
            metrics_df, 
            promotion_df_copy[promotion_cols],
            on='rule_id',
            how='left'
        )
        
        # Check for NaN values after merge
        nan_count = result_df['name'].isna().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} promotions with NaN names after merging")
            
            # Debug: Show rule IDs that didn't match
            missing_rule_ids = result_df[result_df['name'].isna()]['rule_id'].tolist()
            logger.warning(f"Rule IDs not found in promotion data: {missing_rule_ids[:5]}...")
        
        # Calculate duration
        if 'from_date' in result_df.columns and 'to_date' in result_df.columns:
            result_df['from_date'] = pd.to_datetime(result_df['from_date'])
            result_df['to_date'] = pd.to_datetime(result_df['to_date'])
            result_df['duration_days'] = (result_df['to_date'] - result_df['from_date']).dt.days + 1
    else:
        # Empty result if no matches
        result_df = pd.DataFrame(columns=[
            'rule_id', 'revenue', 'order_count', 'item_count', 'unique_sku_count', 'name',
            'discount_amount', 'ctb_discount_used', 'ctb_units_used',
            'from_date', 'to_date', 'duration_days'
        ])
    
    # Sort by revenue (descending)
    result_df = result_df.sort_values('revenue', ascending=False)
    
    # Calculate additional metrics
    # 1. Units sold per day
    result_df['units_sold_per_day'] = result_df.apply(
        lambda row: row['ctb_units_used'] / row['duration_days'] if row['duration_days'] > 0 else 0, 
        axis=1
    )
    
    # 2. Average price per item (will be used for bubble size)
    # Calculate average price per item from sales data
    if 'base_original_price' in original_sales_df.columns and 'rule_ids_list' in sales_df.columns:
        # Pre-compute average price by rule_id
        rule_avg_prices = {}
        
        for _, row in sales_df.iterrows():
            if pd.isna(row.get('base_original_price', None)) or not hasattr(row, 'rule_ids_list'):
                continue
                
            rule_ids = row['rule_ids_list'] if isinstance(row['rule_ids_list'], list) else []
            
            for rule_id in rule_ids:
                if rule_id not in rule_avg_prices:
                    rule_avg_prices[rule_id] = {'total_price': 0, 'count': 0}
                
                rule_avg_prices[rule_id]['total_price'] += float(row['base_original_price'])
                rule_avg_prices[rule_id]['count'] += 1
        
        # Calculate averages
        avg_prices = {
            rule_id: data['total_price'] / data['count'] if data['count'] > 0 else 0
            for rule_id, data in rule_avg_prices.items()
        }
        
        # Add to metrics_data
        result_df['avg_price_per_item'] = result_df['rule_id'].map(
            lambda x: avg_prices.get(x, 0)
        )
    else:
        # Fallback if base_original_price is not available
        result_df['avg_price_per_item'] = result_df['revenue'] / result_df['item_count'].replace(0, 1)
    
    # 3. Calculate profit margin (Revenue - Discount Used)
    # Convert both values to float to avoid decimal.Decimal incompatibility
    result_df['profit_margin'] = result_df.apply(
        lambda row: float(row['revenue']) - float(row['ctb_discount_used']), 
        axis=1
    )
    
    # Cache the full result if enabled
    if use_cache and not result_df.empty:
        cache.set(filter_params, result_df)
        logger.info(f"Cached result with {len(result_df)} promotions in {time.time() - start_time:.2f} seconds")
    
    return result_df


def sort_promotions(metrics_df: pd.DataFrame, max_promotions: int, sort_option: str) -> pd.DataFrame:
    """
    Sort and filter promotion data based on sorting option.
    
    Args:
        metrics_df: DataFrame with promotion metrics
        max_promotions: Maximum number of promotions to include
        sort_option: Sorting option (top_revenue, lowest_revenue, random)
        
    Returns:
        DataFrame with sorted and filtered promotions
    """
    if metrics_df.empty:
        return metrics_df
        
    # Apply sorting logic
    if sort_option == "top_revenue":
        return metrics_df.sort_values('revenue', ascending=False).head(max_promotions)
    elif sort_option == "lowest_revenue":
        # Filter to promotions with revenue > 0
        revenue_gt_zero = metrics_df[metrics_df['revenue'] > 0]
        if revenue_gt_zero.empty:
            return metrics_df.head(max_promotions)
        else:
            return revenue_gt_zero.sort_values('revenue', ascending=True).head(max_promotions)
    elif sort_option == "random":
        if len(metrics_df) <= max_promotions:
            return metrics_df
        else:
            return metrics_df.sample(n=max_promotions)
    else:  # fallback
        return metrics_df.head(max_promotions)


def create_promotion_chart(
    promotion_data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    size_metric: str,
    metric_options: Dict[str, Dict[str, Any]]
) -> go.Figure:
    """
    Create a customizable bubble chart for promotion analysis with profit margin coloring.
    
    Args:
        promotion_data: DataFrame with promotion metrics
        x_axis: Column name to use for x-axis
        y_axis: Column name to use for y-axis
        size_metric: Column name to use for bubble size
        metric_options: Display options for metrics
        
    Returns:
        Plotly figure object
    """
    if promotion_data.empty:
        logger.warning("Cannot create chart with empty promotion data")
        return None
    
    # Check required columns exist
    required_cols = [x_axis, y_axis, size_metric, 'name', 'profit_margin']
    if not all(col in promotion_data.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in promotion_data.columns]
        logger.warning(f"Cannot create chart, missing columns: {missing_cols}")
        return None
    
    # Check for NaN values
    for col in [x_axis, y_axis, size_metric, 'profit_margin']:
        nan_count = promotion_data[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column {col} has {nan_count} NaN values")
    
    # Handle NaN values by filling with 0
    chart_data = promotion_data.copy()
    chart_data[x_axis] = chart_data[x_axis].fillna(0)
    chart_data[y_axis] = chart_data[y_axis].fillna(0)
    chart_data[size_metric] = chart_data[size_metric].fillna(0)
    chart_data['profit_margin'] = chart_data['profit_margin'].fillna(0)
    
    # Create the bubble chart
    fig = go.Figure()
    
    # Get display names from options
    x_name = metric_options[x_axis]['display_name']
    y_name = metric_options[y_axis]['display_name']
    size_name = metric_options[size_metric]['display_name']
    
    # Get formatting options
    x_format = metric_options[x_axis].get('format', ':.2f')
    y_format = metric_options[y_axis].get('format', '')
    size_format = metric_options[size_metric].get('format', '')
    
    # Make sure we have positive values for size (use absolute value)
    size_values = np.sqrt(np.abs(chart_data[size_metric])) + 5
    
    # Use profit_margin for coloring (Revenue - Discount Used)
    color_values = chart_data['profit_margin']
    colorbar_title = 'Profit Margin (€)'
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=chart_data[x_axis],
        y=chart_data[y_axis],
        mode='markers',
        marker=dict(
            size=size_values,
            sizemode='area',
            sizeref=2.*max(size_values)/(40.**2) if len(size_values) > 0 else 1,
            color=color_values,
            colorscale='Viridis',
            colorbar=dict(
                title=colorbar_title,
                thickness=15,
                len=0.7,
                tickfont=dict(size=12),
                tickprefix='€'
            ),
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=chart_data['name'],
        customdata=np.stack((
            chart_data[size_metric],  # Original size value (not sqrt-scaled)
            chart_data['profit_margin'],  # Profit margin
            chart_data['duration_days'] if 'duration_days' in chart_data.columns else np.ones(len(chart_data))
        ), axis=-1),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            f'{x_name}: %{{x}}<br>' +
            f'{y_name}: %{{y}}<br>' +
            f'{size_name}: %{{customdata[0]:.2f}}<br>' +
            f'Profit Margin: €%{{customdata[1]:.2f}}<br>' +
            'Duration: %{customdata[2]:.0f} days<br>' +
            '<extra></extra>'
        )
    ))
    
    # Set up layout
    fig.update_layout(
        title="Promotion Analysis Dashboard",
        xaxis=dict(
            title=x_name,
            gridcolor='rgba(230, 230, 230, 0.3)',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=y_name,
            gridcolor='rgba(230, 230, 230, 0.3)',
            tickfont=dict(size=12)
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.05)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='rgba(70, 70, 70, 1)', size=12),
        height=800,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode='closest'
    )
    
    # Add currency prefix if dealing with revenue or price
    if x_axis in ['revenue', 'avg_price_per_item', 'profit_margin']:
        fig.update_layout(xaxis=dict(tickprefix='€', tickformat=',.2f'))
    if y_axis in ['revenue', 'avg_price_per_item', 'profit_margin']:
        fig.update_layout(yaxis=dict(tickprefix='€', tickformat=',.2f'))
        
    return fig


def display_combined_promotion_chart(
    original_sales_df: pd.DataFrame, 
    promotion_df: pd.DataFrame,
    filter_years: List[int] = None,
    filter_category_ids: List[int] = None
) -> go.Figure:
    """
    Display a combined promotion analysis dashboard with configurable metrics.
    
    Args:
        original_sales_df: Original unfiltered sales DataFrame
        promotion_df: DataFrame with promotion details
        filter_years: Years to filter by (optional)
        filter_category_ids: Category IDs to filter by (optional)
        
    Returns:
        Plotly figure object or None
    """
    # Validate input data
    if original_sales_df.empty or promotion_df.empty:
        st.error("No sales or promotion data available.")
        return None
        
    # Define available metrics and their display options
    metric_options = {
        'discount_amount': {
            'display_name': 'Discount Amount',
            'format': 'x:.2f'
        },
        'ctb_discount_used': {
            'display_name': 'Times Used',
            'format': 'y:,d'
        },
        'ctb_units_used': {
            'display_name': 'Units Sold',
            'format': ':.0f'
        },
        'units_sold_per_day': {
            'display_name': 'Units Sold per Day',
            'format': ':.2f'
        },
        'revenue': {
            'display_name': 'Revenue Generated (€)',
            'format': 'y:,.2f'
        },
        'profit_margin': {
            'display_name': 'Profit Margin (€)',
            'format': 'y:,.2f'
        },
        'avg_price_per_item': {
            'display_name': 'Avg Price per Item (€)',
            'format': ':.2f'
        },
        'order_count': {
            'display_name': 'Order Count',
            'format': ':.0f'
        },
        'item_count': {
            'display_name': 'Item Count',
            'format': ':.0f'
        },
        'unique_sku_count': {
            'display_name': 'Unique SKUs',
            'format': ':.0f'
        },
        'duration_days': {
            'display_name': 'Duration (days)',
            'format': ':.0f'
        }
    }
    
    # Create filter container with columns (6:2:1:1 ratio)
    col1, col2, col3, col4 = st.columns([6, 2, 1, 1])
    
    with col1:
        max_promotions = st.slider(
            "Number of promotions to analyze", 
            min_value=10, 
            max_value=200, 
            value=50,
            step=10,
            key="promotion_max_slider"
        )
    
    with col2:
        sort_option = st.selectbox(
            "Promotion selection",
            options=["top_revenue", "lowest_revenue", "random"],
            format_func=lambda x: {
                "top_revenue": "Top Revenue",
                "lowest_revenue": "Lowest Revenue",
                "random": "Random Selection"
            }[x],
            key="promotion_combined_sort_option"
        )
    
    with col3:
        y_axis_option = st.selectbox(
            "Y-Axis",
            options=["ctb_units_used", "units_sold_per_day"],
            format_func=lambda x: {
                "ctb_units_used": "Units Sold",
                "units_sold_per_day": "Units Sold per Day"
            }[x],
            key="promotion_y_axis_option"
        )
    
    with col4:
        x_axis_option = st.selectbox(
            "X-Axis",
            options=["discount_amount", "unique_sku_count"],
            format_func=lambda x: {
                "discount_amount": "Discount Amount",
                "unique_sku_count": "Unique SKUs"
            }[x],
            key="promotion_x_axis_option"
        )
    
    # Set y-axis based on user selection
    y_axis = y_axis_option
    
    # Set x-axis based on user selection
    x_axis = x_axis_option
    
    # Use avg_price_per_item for bubble size
    size_metric = 'avg_price_per_item'
    
    # Process and calculate promotion metrics
    with st.spinner("Processing promotion data..."):
        # Calculate comprehensive metrics - passing ALL parameters now
        metrics_data = calculate_promotion_metrics(
            original_sales_df, 
            promotion_df,
            filter_years=filter_years,
            filter_category_ids=filter_category_ids,
            max_promotions=max_promotions,
            sort_option=sort_option,
            use_cache=True
        )
        
        if metrics_data.empty:
            st.warning("No promotion metrics data available for the selected filters.")
            return None
        
        # Sort and display the selected number of promotions
        selected_promotions = sort_promotions(metrics_data, max_promotions, sort_option)
        
    # Create and display the visualization
    with st.spinner("Generating promotion analysis chart..."):
        # Create the chart
        fig = create_promotion_chart(
            selected_promotions,
            x_axis=x_axis,
            y_axis=y_axis,
            size_metric=size_metric,
            metric_options=metric_options
        )
        
        # Display the chart
        if fig is None:
            st.warning("Could not generate chart with the current data.")
            return None
            
        st.plotly_chart(fig, use_container_width=True)
        
        # Create explanation based on selected metrics
        x_name = metric_options[x_axis]['display_name']
        y_name = metric_options[y_axis]['display_name']
        size_name = metric_options[size_metric]['display_name']
        
        # Add explanatory text
        st.markdown(f"""
        **How to read this chart:**
        - **X-axis**: {x_name}
        - **Y-axis**: {y_name} 
        - **Bubble size**: {size_name}
        - **Color**: Profit Margin (Revenue - Discount Used)
        
        Larger bubbles indicate promotions with higher average price per item. Brighter colors indicate higher profit margins.
        """)
    
    return fig