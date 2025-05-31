"""
Improved product basket analysis component with physical caching.
Updated to include customer filtering in the cache key.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from itertools import combinations
from collections import Counter, defaultdict
import logging
from pathlib import Path
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple, Set

# Set up logging
logger = logging.getLogger(__name__)


class BasketAnalysisCache:
    """Cache for processed basket analysis data."""
    
    def __init__(self, cache_dir: str = "cache/basket_analysis", max_age_hours: int = 24):
        """
        Initialize the basket analysis cache.
        
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
        Updated to include customer type and status filters.
        
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
        # Create a reproducible string representation of parameters
        discount_str = "with_disc" if include_discounts else "no_disc"
        
        # Add customer type and status to the key if provided
        customer_type_str = "_".join(sorted(customer_types)) if customer_types else "all_types"
        customer_status_str = "_".join(sorted(customer_status)) if customer_status else "all_status"
        
        key_str = (f"basket_analysis_"
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
        Retrieve cached basket analysis data if available.
        Updated to include customer filtering parameters.
        
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
           top_n: int, sort_by: str, include_discounts: bool, result: pd.DataFrame,
           customer_types: Optional[List[str]] = None,
           customer_status: Optional[List[str]] = None) -> None:
        """
        Cache basket analysis data.
        Updated to include customer filtering parameters.
        
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
            logger.debug(f"Cached basket analysis data for key: {key}")
        except Exception as e:
            logger.warning(f"Error caching basket analysis data: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error removing cache file {cache_file}: {str(e)}")


def optimize_sales_data(product_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process and optimize the product sales dataframe for basket analysis.
    
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


def calculate_association_metrics(
    product_sales: pd.DataFrame, 
    top_n: int = 20, 
    sort_by: str = "ascending",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Calculate association metrics (support, confidence, lift) for product pairs
    with improved caching and efficiency.
    
    Args:
        product_sales: DataFrame containing the filtered sales data
        top_n: Number of products to include in the analysis
        sort_by: How to select products ("ascending" for top revenue, "descending" for lowest revenue, "random" for random selection)
        use_cache: Whether to use cached results
        
    Returns:
        DataFrame with association metrics for product pairs
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
    
    cache = BasketAnalysisCache()
    if use_cache:
        cached_result = cache.get(years, category_ids, top_n, sort_by, include_discounts,
                                 customer_types, customer_status)
        if cached_result is not None:
            logger.info(f"Using cached basket analysis data for years={years}, categories={category_ids}, customer_types={customer_types}")
            return cached_result
    
    product_sales = optimize_sales_data(product_sales)
    
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
    
    all_orders = product_sales['order_id'].unique()
    total_orders = len(all_orders)
    
    order_products = {}
    for order_id in all_orders:
        order_data = product_sales[product_sales['order_id'] == order_id]
        order_products[order_id] = order_data['sku'].unique().tolist()
    
    product_counts = defaultdict(int)
    for products in order_products.values():
        for product in products:
            product_counts[product] += 1
    
    product_support = {sku: count / total_orders for sku, count in product_counts.items()}
    
    product_co_occurrences = {sku: defaultdict(int) for sku in selected_product_skus}
    
    all_products_in_orders = set()
    for products in order_products.values():
        all_products_in_orders.update(products)
    
    for products in order_products.values():
        for product_a in products:
            if product_a in selected_product_skus:
                for product_b in products:
                    if product_a != product_b:
                        product_co_occurrences[product_a][product_b] += 1
    
    association_data = []
    
    for product_a in selected_product_skus:
        if not product_co_occurrences[product_a]:
            continue
            
        co_occur_items = list(product_co_occurrences[product_a].items())
        co_occur_items.sort(key=lambda x: x[1], reverse=True)
        
        top_co_occurrences = co_occur_items[:5]
        
        for product_b, both_count in top_co_occurrences:
            support = both_count / total_orders
            
            confidence_a_b = both_count / product_counts[product_a] if product_counts[product_a] > 0 else 0
            confidence_b_a = both_count / product_counts[product_b] if product_counts[product_b] > 0 else 0
            
            lift = 0
            if product_support.get(product_a, 0) > 0 and product_support.get(product_b, 0) > 0:
                lift = support / (product_support.get(product_a, 0) * product_support.get(product_b, 0))
            
            if confidence_a_b >= confidence_b_a:
                association_data.append({
                    'product_a': product_a,
                    'product_b': product_b,
                    'support': support,
                    'confidence': confidence_a_b,
                    'lift': lift,
                    'count': both_count,
                    'product_a_count': product_counts[product_a],
                    'product_b_count': product_counts[product_b]
                })
            else:
                association_data.append({
                    'product_a': product_b,
                    'product_b': product_a,
                    'support': support,
                    'confidence': confidence_b_a,
                    'lift': lift,
                    'count': both_count,
                    'product_a_count': product_counts[product_b],
                    'product_b_count': product_counts[product_a]
                })
    
    association_df = pd.DataFrame(association_data)
    
    if not association_df.empty:
        association_df = association_df.sort_values(by='lift', ascending=False)
    
    if use_cache and not association_df.empty:
        cache.set(years, category_ids, top_n, sort_by, include_discounts, association_df)
    
    return association_df


def create_basket_scatter_plot(association_metrics):
    """
    Create a scatter plot visualization for basket analysis metrics.
    With improved color scaling to handle outliers in lift values.
    
    Args:
        association_metrics: DataFrame with association metrics for product pairs
        
    Returns:
        Plotly figure object
    """
    if association_metrics.empty:
        return None
    
    metrics_df = association_metrics.copy()
    
    lift_cap = min(10, metrics_df['lift'].quantile(0.95))
    
    metrics_df['lift_display'] = metrics_df['lift'].clip(upper=lift_cap)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=metrics_df['support'],
        y=metrics_df['confidence'],
        mode='markers',
        name='',
        marker=dict(
            size=40,
            color=metrics_df['lift_display'],
            colorscale=[[0, '#4060A8'], [0.5, '#7A5B8F'], [1, '#B22222']],
            colorbar=dict(
                thickness=15,
                len=0.7,
                tickfont=dict(size=26, color='rgba(0, 0, 0, 0.9)'),
                title=dict(text='Lift', font=dict(size=26, color='rgba(0, 0, 0, 0.9)', weight='bold'))
            ),
            symbol='circle', 
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=[f"{row['product_a']} → {row['product_b']}" for _, row in metrics_df.iterrows()],
        customdata=np.stack((
            metrics_df['count'].fillna(0).astype(float),
            metrics_df['lift'].fillna(0).astype(float)
        ), axis=-1),
        hovertemplate=(
            '<b>Product Pair: %{text}</b><br>' +
            'Support: %{x:.4f}<br>' +
            'Confidence: %{y:.4f}<br>' +
            'Lift: %{customdata[1]:.2f}<br>' +
            'Co-occurrence Count: %{customdata[0]:.0f}<extra></extra>'
        ),
        hoverlabel=dict(
            font_size=32,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            font=dict(color='rgba(0, 0, 0, 0.9)', weight='normal')
        )
    ))
    
    fig.update_layout(
        xaxis=dict(
            title='Support (Frequency of Co-occurrence)',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            gridcolor='rgba(0, 0, 0, 0.3)',
            showgrid=True,
            showticklabels=True,
            nticks=15
        ),
        yaxis=dict(
            title='Confidence (Probability of B given A)',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            gridcolor='rgba(0, 0, 0, 0.3)',
            showgrid=True,
            showticklabels=True,
            nticks=15
        ),
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',
        font=dict(size=25, color='rgba(0, 0, 0, 1)', family='Monospace'),
        hovermode='closest',
        hoverlabel=dict(
            font_size=28,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            font=dict(color='rgba(0, 0, 0, 0.9)', weight='bold')
        ),
        height=1200
    )
    
    return fig


def display_basket_analysis(product_sales: pd.DataFrame, max_products: int = 20):
    """
    Display product basket analysis showing which products are frequently purchased together.
    Uses optimized caching and focuses on top product pairs.
    
    Args:
        product_sales: DataFrame containing the filtered sales data
        max_products: Default number of products to include in the analysis
    """
    if product_sales.empty:
        st.info("No sales data available for the selected criteria.")
        return
    
    filter_col1, filter_col2 = st.columns([4, 2])
    
    with filter_col1:
        num_products = st.slider(
            "Number of products for basket analysis", 
            min_value=5, 
            max_value=30, 
            value=15,
            step=5
        )
    
    with filter_col2:
        sort_option = st.selectbox(
            "Product selection",
            options=["ascending", "descending", "random"],
            format_func=lambda x: {
                "ascending": "Top revenue",
                "descending": "Lowest revenue",
                "random": "Random"
            }[x],
            key="basket_sort_option"
        )
    
    with st.spinner("Calculating product associations..."):
        association_metrics = calculate_association_metrics(
            product_sales, 
            top_n=num_products,
            sort_by=sort_option,
            use_cache=True
        )
    
    if association_metrics.empty or len(association_metrics) < 3:
        st.info("Not enough product associations found with the current settings. Try increasing the number of products.")
        return
    
    fig = create_basket_scatter_plot(association_metrics)
    
    if fig is None:
        st.info("Could not generate chart with the current data.")
        return
    
    st.plotly_chart(fig, use_container_width=True)
    