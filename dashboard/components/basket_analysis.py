"""
Improved product basket analysis component with physical caching.
Analyzes which products are frequently purchased together and visualizes
their support, confidence, and lift metrics in a scatter plot.
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
                      top_n: int, sort_by: str, include_discounts: bool) -> str:
        """
        Generate a unique cache key based on filter parameters.
        
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
        key_str = (f"basket_analysis_"
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
        Retrieve cached basket analysis data if available.
        
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
        Cache basket analysis data.
        
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
    cache = BasketAnalysisCache()
    if use_cache:
        cached_result = cache.get(years, category_ids, top_n, sort_by, include_discounts)
        if cached_result is not None:
            logger.info(f"Using cached basket analysis data for years={years}, categories={category_ids}")
            return cached_result
    
    # Optimize the dataframe
    product_sales = optimize_sales_data(product_sales)
    
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
    
    # Get list of all orders
    all_orders = product_sales['order_id'].unique()
    total_orders = len(all_orders)
    
    # Create a dictionary mapping each order to its products
    order_products = {}
    for order_id in all_orders:
        order_data = product_sales[product_sales['order_id'] == order_id]
        order_products[order_id] = order_data['sku'].unique().tolist()
    
    # Count individual product frequencies for ALL products in orders
    product_counts = defaultdict(int)
    for products in order_products.values():
        for product in products:
            product_counts[product] += 1
    
    # Calculate individual product support for ALL products
    product_support = {sku: count / total_orders for sku, count in product_counts.items()}
    
    # Create a dictionary to store co-occurrence counts
    # For each selected product, we'll track its co-occurrences with ALL other products
    product_co_occurrences = {sku: defaultdict(int) for sku in selected_product_skus}
    
    # Get all unique products in the dataset (not just selected ones)
    all_products_in_orders = set()
    for products in order_products.values():
        all_products_in_orders.update(products)
    
    # Calculate co-occurrence counts for each order
    for products in order_products.values():
        # For each selected product in this order, find co-occurrences with ALL products
        for product_a in products:
            if product_a in selected_product_skus:  # If it's one of our selected products
                for product_b in products:
                    if product_a != product_b:  # Don't count co-occurrence with itself
                        product_co_occurrences[product_a][product_b] += 1
    
    # For each selected product, find the top 5 products that co-occur with it most frequently
    association_data = []
    
    for product_a in selected_product_skus:
        # Skip if product has no co-occurrences
        if not product_co_occurrences[product_a]:
            continue
            
        # Get co-occurrence counts and sort by frequency (most frequent first)
        co_occur_items = list(product_co_occurrences[product_a].items())
        co_occur_items.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top 5 co-occurring products (or fewer if there aren't 5)
        top_co_occurrences = co_occur_items[:5]
        
        # Now calculate metrics only for these top co-occurring pairs
        for product_b, both_count in top_co_occurrences:
            # Calculate metrics
            support = both_count / total_orders
            
            confidence_a_b = both_count / product_counts[product_a] if product_counts[product_a] > 0 else 0
            confidence_b_a = both_count / product_counts[product_b] if product_counts[product_b] > 0 else 0
            
            lift = 0
            if product_support.get(product_a, 0) > 0 and product_support.get(product_b, 0) > 0:
                lift = support / (product_support.get(product_a, 0) * product_support.get(product_b, 0))
            
            # Choose the direction with higher confidence
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
    
    # Convert to DataFrame
    association_df = pd.DataFrame(association_data)
    
    # Sort by lift descending
    if not association_df.empty:
        association_df = association_df.sort_values(by='lift', ascending=False)
    
    # Cache the result if enabled
    if use_cache and not association_df.empty:
        cache.set(years, category_ids, top_n, sort_by, include_discounts, association_df)
    
    return association_df


def create_basket_scatter_plot(association_metrics):
    """
    Create a scatter plot visualization for basket analysis metrics.
    
    Args:
        association_metrics: DataFrame with association metrics for product pairs
        
    Returns:
        Plotly figure object
    """
    if association_metrics.empty:
        return None
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=association_metrics['support'],
        y=association_metrics['confidence'],
        mode='markers',
        name='',
        marker=dict(
            size=16,  # Larger markers
            color=association_metrics['lift'],
            colorscale='Viridis',
            colorbar=dict(
                title='Lift',
                thickness=15,
                len=0.7,
                tickfont=dict(size=20, color='rgba(255, 255, 255, 0.8)')
            ),
            symbol='circle', 
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=[f"{row['product_a']} → {row['product_b']}" for _, row in association_metrics.iterrows()],
        hovertemplate=(
            'Product Pair: %{text}<br>' +
            'Support: %{x:.4f}<br>' +
            'Confidence: %{y:.4f}<br>' +
            'Lift: %{marker.color:.2f}<br>' +
            'Co-occurrence Count: %{customdata[0]}'
        ),
        customdata=np.stack((
            association_metrics['count'],
        ), axis=-1)
    ))
    
    # Set up layout with dark theme
    fig.update_layout(
        xaxis=dict(
            title='Support (Frequency of Co-occurrence)',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            title='Confidence (Probability of B given A)',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        plot_bgcolor='rgba(11, 14, 31, 1)',
        paper_bgcolor='rgba(11, 14, 31, 1)',
        font=dict(color='rgba(255, 255, 255, 0.8)', size=16),
        hovermode='closest',
        hoverlabel=dict(
        font_size=16),
         height=800
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
    
    # Create filters container
    filter_col1, filter_col2 = st.columns([4, 2])
    
    with filter_col1:
        num_products = st.slider(
            "Number of products for basket analysis", 
            min_value=5, 
            max_value=50, 
            value=max_products,
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
    
    # Calculate association metrics
    with st.spinner("Calculating product associations..."):
        # Use our optimized function with caching
        association_metrics = calculate_association_metrics(
            product_sales, 
            top_n=num_products,
            sort_by=sort_option,
            use_cache=True
        )
    
    if association_metrics.empty or len(association_metrics) < 3:
        st.info("Not enough product associations found with the current settings. Try increasing the number of products.")
        return
    
    # Create the scatter plot (this will be cached)
    fig = create_basket_scatter_plot(association_metrics)
    
    if fig is None:
        st.info("Could not generate chart with the current data.")
        return
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanatory text
    st.markdown("""
    <div style="background-color: rgba(11, 14, 31, 0.7); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">
            <strong>How to read this chart:</strong><br>
            • <strong>Support</strong>: The percentage of orders containing both products (X-axis)<br>
            • <strong>Confidence</strong>: The probability that a customer buys product B when buying product A (Y-axis)<br>
            • <strong>Lift</strong>: Indicates how much more likely products are bought together than expected if they were independent (Color)<br>
            • Higher lift (brighter color) suggests stronger association between products
        </p>
    </div>
    """, unsafe_allow_html=True)