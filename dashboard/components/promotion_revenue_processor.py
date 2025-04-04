"""
Promotion revenue processing module.
Calculates revenue by promotion rule ID based on sales data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from decimal import Decimal

# Set up logging
logger = logging.getLogger(__name__)


class PromotionRevenueCache:
    """Cache for processed promotion revenue data."""
    
    def __init__(self, cache_dir: str = "cache/promotion_revenue", max_age_hours: int = 24):
        """
        Initialize the promotion revenue cache.
        
        Args:
            cache_dir: Directory to store cached results
            max_age_hours: Maximum age of cached results in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, years: List[int], category_ids: List[int]) -> str:
        """
        Generate a unique cache key based on filter parameters.
        
        Args:
            years: List of selected years
            category_ids: List of selected category IDs
            
        Returns:
            Unique cache key
        """
        key_str = f"promotion_revenue_{'_'.join(map(str, sorted(years)))}_{'_'.join(map(str, sorted(category_ids)))}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, years: List[int], category_ids: List[int]) -> Optional[pd.DataFrame]:
        """
        Retrieve cached promotion revenue data if available.
        
        Args:
            years: List of selected years
            category_ids: List of selected category IDs
            
        Returns:
            Cached DataFrame or None if not available
        """
        key = self._get_cache_key(years, category_ids)
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
    
    def set(self, years: List[int], category_ids: List[int], result: pd.DataFrame) -> None:
        """
        Cache promotion revenue data.
        
        Args:
            years: List of selected years
            category_ids: List of selected category IDs
            result: Processed DataFrame to cache
        """
        key = self._get_cache_key(years, category_ids)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Cached promotion revenue data for key: {key}")
        except Exception as e:
            logger.warning(f"Error caching promotion revenue data: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error removing cache file {cache_file}: {str(e)}")


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


def calculate_promotion_revenue(
    sales_df: pd.DataFrame, 
    promotion_df: pd.DataFrame,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Calculate revenue by promotion rule ID.
    
    Args:
        sales_df: DataFrame with sales data
        promotion_df: DataFrame with promotion details
        use_cache: Whether to use cached results
        
    Returns:
        DataFrame with revenue by promotion rule ID
    """
    # Extract filter parameters for cache key
    if 'order_date' in sales_df.columns:
        years = sorted(sales_df['order_date'].dt.year.unique().tolist())
    else:
        years = []
        
    if 'category_num' in sales_df.columns:
        category_ids = sorted(sales_df['category_num'].unique().tolist())
    else:
        category_ids = []
    
    # Check cache first if enabled
    cache = PromotionRevenueCache()
    if use_cache:
        cached_result = cache.get(years, category_ids)
        if cached_result is not None:
            logger.info(f"Using cached promotion revenue data for years={years}, categories={category_ids}")
            return cached_result
    
    # Process promotion data
    # Filter to used promotions (those with actual discount usage)
    used_promotions = promotion_df[promotion_df['ctb_discount_used'] > 0]
    used_rule_ids = used_promotions['rule_id'].tolist()
    
    # Create a dictionary to store revenue by rule ID
    rule_revenue = {rule_id: 0.0 for rule_id in used_rule_ids}
    rule_orders = {rule_id: 0 for rule_id in used_rule_ids}
    rule_items = {rule_id: 0 for rule_id in used_rule_ids}
    
    # Process sales data
    # Create a new column with processed rule IDs
    sales_df['rule_ids_list'] = sales_df['applied_rule_ids'].apply(process_applied_rule_ids)
    
    # Calculate revenue for each sale and distribute to applicable rules
    for _, row in sales_df.iterrows():
        # Skip if no rules applied
        if len(row['rule_ids_list']) == 0:
            continue
        
        # Calculate revenue for this item - convert to float to handle Decimal type
        item_revenue = float(row['base_row_total_incl_tax']) - float(row['base_discount_amount'])
        if item_revenue <= 0:
            continue
        
        # Get the used rules for this item that match our filter
        matching_rules = [rule_id for rule_id in row['rule_ids_list'] if rule_id in used_rule_ids]
        
        # Skip if no matching rules
        if len(matching_rules) == 0:
            continue
        
        # Distribute revenue equally among rules (simplified approach)
        revenue_per_rule = item_revenue / len(matching_rules)
        
        # Add to rule revenue
        for rule_id in matching_rules:
            rule_revenue[rule_id] += revenue_per_rule
            rule_orders[rule_id] += 1
            rule_items[rule_id] += float(row['qty_ordered'])
    
    # Create DataFrame from dictionaries
    revenue_df = pd.DataFrame({
        'rule_id': list(rule_revenue.keys()),
        'revenue': list(rule_revenue.values()),
        'order_count': list(rule_orders.values()),
        'item_count': list(rule_items.values())
    })
    
    # Merge with promotion details
    if not revenue_df.empty and not promotion_df.empty:
        result_df = pd.merge(
            revenue_df, 
            promotion_df[['rule_id', 'name', 'discount_amount', 'ctb_discount_used', 'ctb_units_used', 'from_date', 'to_date']],
            on='rule_id',
            how='left'
        )
        
        # Calculate duration
        result_df['from_date'] = pd.to_datetime(result_df['from_date'])
        result_df['to_date'] = pd.to_datetime(result_df['to_date'])
        result_df['duration_days'] = (result_df['to_date'] - result_df['from_date']).dt.days + 1
    else:
        # Empty result if no matches
        result_df = pd.DataFrame(columns=[
            'rule_id', 'revenue', 'order_count', 'item_count', 'name',
            'discount_amount', 'ctb_discount_used', 'ctb_units_used',
            'from_date', 'to_date', 'duration_days'
        ])
    
    # Sort by revenue (descending)
    result_df = result_df.sort_values('revenue', ascending=False)
    
    # Cache the result if enabled
    if use_cache and not result_df.empty:
        cache.set(years, category_ids, result_df)
    
    return result_df


def create_promotion_revenue_chart(revenue_data: pd.DataFrame) -> go.Figure:
    """
    Create a bubble chart visualization for promotion revenue.
    
    Args:
        revenue_data: DataFrame with promotion revenue data
        
    Returns:
        Plotly figure object
    """
    # Create the bubble chart
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=revenue_data['discount_amount'],
        y=revenue_data['revenue'],  # Using revenue as y-axis instead of ctb_discount_used
        mode='markers',
        marker=dict(
            size=np.sqrt(revenue_data['item_count']) + 5,  # Square root scaling for better visual
            sizemode='area',
            sizeref=2.*max(np.sqrt(revenue_data['item_count'])+5)/(40.**2),
            color=revenue_data['duration_days'],
            colorscale='Viridis',
            colorbar=dict(
                title='Duration (days)',
                thickness=15,
                len=0.7,
                tickfont=dict(size=12)
            ),
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=revenue_data['name'],
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Discount Amount: %{x:.2f}<br>' +
            'Generated Revenue: %{y:,.2f}<br>' +
            'Items Sold: %{marker.size:.0f}<br>' +
            'Duration: %{marker.color:.0f} days<br>' +
            '<extra></extra>'
        )
    ))
    
    # Set up layout
    fig.update_layout(
        title="Promotion Revenue Analysis",
        xaxis=dict(
            title='Discount Amount',
            gridcolor='rgba(230, 230, 230, 0.3)',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='Revenue Generated (€)',
            gridcolor='rgba(230, 230, 230, 0.3)',
            tickfont=dict(size=12),
            tickprefix='€',
            tickformat=',.2f'
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.05)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='rgba(70, 70, 70, 1)', size=12),
        height=800,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode='closest'
    )
    
    return fig


def display_promotion_revenue_chart(sales_df: pd.DataFrame, promotion_df: pd.DataFrame) -> go.Figure:
    """
    Display a bubble chart visualization for promotion revenue with UI filters.
    
    Args:
        sales_df: DataFrame with sales data
        promotion_df: DataFrame with promotion details
        
    Returns:
        Plotly figure object
    """
    if sales_df.empty or promotion_df.empty:
        st.info("No sales or promotion data available.")
        return None
    
    # Create filters container
    filter_col1, filter_col2 = st.columns([2, 1])
    
    with filter_col1:
        max_promotions = st.slider(
            "Number of promotions to analyze", 
            min_value=10, 
            max_value=100, 
            value=50,
            step=10
        )
    
    with filter_col2:
        sort_option = st.selectbox(
            "Promotion selection",
            options=["most_revenue", "least_revenue", "random"],
            format_func=lambda x: {
                "most_revenue": "Highest revenue",
                "least_revenue": "Lowest revenue",
                "random": "Random"
            }[x],
            key="promotion_revenue_sort_option"
        )
    
    # Process and calculate promotion revenue
    with st.spinner("Processing promotion revenue data..."):
        # Calculate all promotion revenue
        revenue_data = calculate_promotion_revenue(sales_df, promotion_df)
        
        # Filter based on user selection
        if revenue_data.empty:
            st.info("No promotion revenue data available.")
            return None
        
        # Apply sorting and filtering
        if sort_option == "most_revenue":
            selected_promotions = revenue_data.sort_values('revenue', ascending=False).head(max_promotions)
        elif sort_option == "least_revenue":
            # Filter to promotions with revenue > 0
            revenue_gt_zero = revenue_data[revenue_data['revenue'] > 0]
            if revenue_gt_zero.empty:
                selected_promotions = revenue_data.head(max_promotions)
            else:
                selected_promotions = revenue_gt_zero.sort_values('revenue', ascending=True).head(max_promotions)
        elif sort_option == "random":
            if len(revenue_data) <= max_promotions:
                selected_promotions = revenue_data
            else:
                selected_promotions = revenue_data.sample(n=max_promotions)
        else:
            selected_promotions = revenue_data.head(max_promotions)
    
    # Create the visualization
    with st.spinner("Generating promotion revenue chart..."):
        # Create the chart
        fig = create_promotion_revenue_chart(selected_promotions)
        
        # Display the chart
        if fig is None:
            st.info("Could not generate chart with the current data.")
            return None
            
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanatory text
        st.markdown("""
        **How to read this chart:**
        - **X-axis**: Discount amount applied in the promotion
        - **Y-axis**: Revenue generated by the promotion
        - **Bubble size**: Number of items sold with this promotion
        - **Color**: Duration of the promotion in days
        
        Larger bubbles with brighter colors indicate promotions that sold more items and ran for longer periods.
        """)
    
    return fig