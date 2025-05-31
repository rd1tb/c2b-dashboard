import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple, Any
import logging
import time
import pickle
from pathlib import Path
import hashlib
from decimal import Decimal


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SalesSensitivityCache:
    """Cache for processed sales sensitivity data with customer type filtering."""
    
    def __init__(self, cache_dir: str = "cache/sales_sensitivity", max_age_hours: int = 24):
        """Initialize the sales sensitivity cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, filter_params: Dict[str, Any]) -> str:
        """Generate a unique cache key based on filter parameters."""
        sorted_params = sorted(filter_params.items())
        
        param_strings = []
        for key, value in sorted_params:
            if isinstance(value, list):
                value_str = "_".join(map(str, sorted(value) if value else []))
            else:
                value_str = str(value) if value is not None else ""
            param_strings.append(f"{key}:{value_str}")
        
        key_str = f"sales_sensitivity_{'__'.join(param_strings)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"
        
    def clear_all_cache(self):
        """Clear all cached files."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            return True
        except Exception as e:
            logger.warning(f"Error clearing cache: {str(e)}")
            return False
    
    def get(self, filter_params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Retrieve cached sales sensitivity data if available."""
        key = self._get_cache_key(filter_params)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
                logger.info(f"Loaded cached result for key: {key}")
                return result
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
            return None
    
    def set(self, filter_params: Dict[str, Any], result: pd.DataFrame) -> None:
        """Cache sales sensitivity data."""
        key = self._get_cache_key(filter_params)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"Cached sales sensitivity data for key: {key}")
        except Exception as e:
            logger.warning(f"Error caching sales sensitivity data: {str(e)}")


def process_applied_rule_ids(rule_ids_str: str) -> List[int]:
    """Process applied rule IDs string into a list of integers."""
    if not rule_ids_str or pd.isna(rule_ids_str) or rule_ids_str == '':
        return []
    
    # Split by comma
    rule_id_parts = str(rule_ids_str).split(',')
    
    rule_ids = []
    for part in rule_id_parts:
        try:
            clean_part = part.strip()
            if clean_part.startswith('c-'):
                clean_part = clean_part[2:]
            
            rule_id = int(clean_part)
            rule_ids.append(rule_id)
        except (ValueError, TypeError):
            continue
    
    return rule_ids


def get_selected_products(
    sales_df: pd.DataFrame, 
    max_products: int = 50,
    selection_method: str = "top_revenue"
) -> List[str]:
    """
    Get products based on selection method.
    
    Args:
        sales_df: DataFrame with sales data
        max_products: Maximum number of products to include
        selection_method: Method to select products (top_revenue, least_revenue, random)
        
    Returns:
        List of selected product SKUs
    """
    if sales_df.empty or 'sku' not in sales_df.columns or 'base_row_total_incl_tax' not in sales_df.columns:
        return []
    
    product_revenue = sales_df.groupby('sku')['base_row_total_incl_tax'].sum().reset_index()
    
    product_revenue = product_revenue[product_revenue['base_row_total_incl_tax'] > 0]
    
    if product_revenue.empty:
        return []
    
    if selection_method == "top_revenue":
        selected_products = product_revenue.sort_values('base_row_total_incl_tax', ascending=False)
    elif selection_method == "least_revenue":
        selected_products = product_revenue.sort_values('base_row_total_incl_tax', ascending=True)
    elif selection_method == "random":
        selected_products = product_revenue.sample(min(len(product_revenue), max_products))
    else:
        selected_products = product_revenue.sort_values('base_row_total_incl_tax', ascending=False)
    
    return selected_products.head(max_products)['sku'].tolist()


def get_discount_bucket(discount: float) -> str:
    """
    Assign a discount amount to a fixed bucket.
    
    Args:
        discount: Discount percentage
        
    Returns:
        Bucket label
    """
    if discount <= 15:
        return "≤15%"
    elif discount <= 20:
        return "20%"
    elif discount <= 25:
        return "25%"
    elif discount <= 30:
        return "30%"
    else:
        return "≥35%"


def calculate_sales_sensitivity(
    sales_df: pd.DataFrame,
    promotion_df: pd.DataFrame,
    filter_years: List[int] = None,
    filter_category_ids: List[int] = None,
    filter_customer_types: List[str] = None,
    filter_customer_status: List[str] = None,
    max_products: int = 50,
    selection_method: str = "top_revenue",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Calculate product sensitivity to different discount levels.
    
    Args:
        sales_df: DataFrame with sales data
        promotion_df: DataFrame with promotion details
        filter_years: Years to filter by
        filter_category_ids: Category IDs to filter by
        filter_customer_types: Customer types to filter by (Guest/Registered)
        filter_customer_status: Customer status to filter by (New/Returning)
        max_products: Maximum products to analyze
        selection_method: Method to select products
        use_cache: Whether to use cached results
        
    Returns:
        DataFrame with sales sensitivity data
    """
    def get_max_discount(rule_ids, discount_map):
        """Get the highest discount amount from a list of rule IDs."""
        if not rule_ids:
            return 0.0
        discounts = []
        for rule_id in rule_ids:
            discount_value = discount_map.get(rule_id, 0)
            try:
                if hasattr(discount_value, 'to_eng_string'):
                    discount_value = float(discount_value.to_eng_string())
                else:
                    discount_value = float(discount_value)
                discounts.append(discount_value)
            except (ValueError, TypeError):
                discounts.append(0.0)
        return max(discounts) if discounts else 0.0
    
    filter_params = {
        'years': filter_years,
        'category_ids': filter_category_ids,
        'customer_types': filter_customer_types,
        'customer_status': filter_customer_status,
        'max_products': max_products,
        'selection_method': selection_method
    }
    
    start_time = time.time()
    
    cache = SalesSensitivityCache()
    if use_cache:
        cached_result = cache.get(filter_params)
        if cached_result is not None:
            logger.info(f"Using cached sales sensitivity data for filters: {filter_params}")
            logger.info(f"Retrieved from cache in {time.time() - start_time:.2f} seconds")
            return cached_result
    
    sales_df_copy = sales_df.copy()
    
    for col in sales_df_copy.columns:
        if col in ['qty_ordered', 'base_row_total_incl_tax', 'base_price', 'base_discount_amount']:
            try:
                sales_df_copy[col] = sales_df_copy[col].apply(
                    lambda x: float(x.to_eng_string()) if hasattr(x, 'to_eng_string') else float(x) 
                    if x is not None and not pd.isna(x) else 0.0
                )
            except Exception as e:
                logger.warning(f"Error converting column {col} to float: {str(e)}")
                try:
                    sales_df_copy[col] = sales_df_copy[col].astype(float)
                except:
                    pass
        elif sales_df_copy[col].dtype.name == 'object':
            try:
                sales_df_copy[col] = sales_df_copy[col].astype(float)
            except (ValueError, TypeError):
                pass
        elif hasattr(sales_df_copy[col], 'astype'):
            try:
                sales_df_copy[col] = sales_df_copy[col].astype(float)
            except (ValueError, TypeError):
                pass
    
    promotion_df = promotion_df.copy()
    for col in promotion_df.columns:
        if col == 'discount_amount':
            try:
                promotion_df[col] = promotion_df[col].apply(
                    lambda x: float(x.to_eng_string()) if hasattr(x, 'to_eng_string') else float(x) 
                    if x is not None and not pd.isna(x) else 0.0
                )
            except Exception as e:
                logger.warning(f"Error converting discount_amount to float: {str(e)}")
                try:
                    promotion_df[col] = promotion_df[col].astype(float)
                except:
                    pass
        elif promotion_df[col].dtype.name == 'object':
            try:
                promotion_df[col] = promotion_df[col].astype(float)
            except (ValueError, TypeError):
                pass
        elif hasattr(promotion_df[col], 'astype'):
            try:
                promotion_df[col] = promotion_df[col].astype(float)
            except (ValueError, TypeError):
                pass


    if filter_years and 'order_date' in sales_df_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(sales_df_copy['order_date']):
            sales_df_copy['order_date'] = pd.to_datetime(sales_df_copy['order_date'])
            
        sales_df_copy = sales_df_copy[sales_df_copy['order_date'].dt.year.isin(filter_years)]
    
    if filter_category_ids and 'category_num' in sales_df_copy.columns:
        sales_df_copy = sales_df_copy[sales_df_copy['category_num'].isin(filter_category_ids)]
    
    if filter_customer_types and 'customer_is_guest' in sales_df_copy.columns and 'customer_group_id' in sales_df_copy.columns:
        if 'Guest' in filter_customer_types and 'Registered' in filter_customer_types:
            pass
        elif 'Guest' in filter_customer_types:
            sales_df_copy = sales_df_copy[sales_df_copy['customer_is_guest'] == 1]
        elif 'Registered' in filter_customer_types:
            sales_df_copy = sales_df_copy[sales_df_copy['customer_group_id'].isin([1, 2, 7, 8, 9])]
        else:
            logger.warning("No customer types selected, returning empty result")
            return pd.DataFrame()
    
    if filter_customer_status and 'hashed_customer_email' in sales_df_copy.columns:
        if 'New' in filter_customer_status and 'Returning' in filter_customer_status:
            pass
        elif len(filter_customer_status) > 0:
            customer_order_counts = sales_df_copy.groupby('hashed_customer_email')['order_id'].nunique().reset_index()
            customer_order_counts.columns = ['hashed_customer_email', 'unique_order_count']
            
            new_customers = set(customer_order_counts[customer_order_counts['unique_order_count'] == 1]['hashed_customer_email'].tolist())
            returning_customers = set(customer_order_counts[customer_order_counts['unique_order_count'] > 1]['hashed_customer_email'].tolist())
            
            if 'New' in filter_customer_status and not 'Returning' in filter_customer_status:
                sales_df_copy = sales_df_copy[sales_df_copy['hashed_customer_email'].isin(new_customers)]
            elif 'Returning' in filter_customer_status and not 'New' in filter_customer_status:
                sales_df_copy = sales_df_copy[sales_df_copy['hashed_customer_email'].isin(returning_customers)]
            elif not filter_customer_status:
                logger.warning("No customer status selected, returning empty result")
                return pd.DataFrame()
    
    if sales_df_copy.empty:
        logger.warning("No sales data after filtering")
        return pd.DataFrame()
    
    selected_products = get_selected_products(sales_df_copy, max_products, selection_method)
    if not selected_products:
        logger.warning("No products found with the selected criteria")
        return pd.DataFrame()
    
    sales_df_copy = sales_df_copy[sales_df_copy['sku'].isin(selected_products)]
    
    sales_df_copy['rule_ids_list'] = sales_df_copy['applied_rule_ids'].apply(process_applied_rule_ids)
    
    rule_discount_map = {}
    if 'rule_id' in promotion_df.columns and 'discount_amount' in promotion_df.columns:
        try:
            rule_ids = promotion_df['rule_id'].apply(
                lambda x: int(x) if x is not None and not pd.isna(x) else 0
            ).tolist()
            
            discount_amounts = promotion_df['discount_amount'].apply(
                lambda x: float(x.to_eng_string()) if hasattr(x, 'to_eng_string') else float(x) 
                if x is not None and not pd.isna(x) else 0.0
            ).tolist()
            
            rule_discount_map = dict(zip(rule_ids, discount_amounts))
        except Exception as e:
            logger.warning(f"Error creating rule_discount_map: {str(e)}")
            rule_discount_map = dict(zip(promotion_df['rule_id'], promotion_df['discount_amount']))
    
    sales_df_copy['max_discount'] = sales_df_copy['rule_ids_list'].apply(
        lambda rule_ids: get_max_discount(rule_ids, rule_discount_map)
    )
    
    sales_df_copy['discount_bucket'] = sales_df_copy['max_discount'].apply(get_discount_bucket)
    
    if not pd.api.types.is_datetime64_any_dtype(sales_df_copy['order_date']):
        sales_df_copy['order_date'] = pd.to_datetime(sales_df_copy['order_date'])
    
    sales_df_copy['sale_date'] = sales_df_copy['order_date'].dt.date
    
    sensitivity_data = []
    products_only_with_discounts = []
    
    for sku in selected_products:
        product_sales = sales_df_copy[sales_df_copy['sku'] == sku]
        if product_sales.empty:
            continue
        
        baseline_sales = product_sales[product_sales['rule_ids_list'].apply(lambda x: len(x) == 0)]
        
        baseline_days = len(baseline_sales['sale_date'].unique()) if not baseline_sales.empty else 0
        
        try:
            if not baseline_sales.empty:
                qty_sum = baseline_sales['qty_ordered'].sum()
                if hasattr(qty_sum, 'to_eng_string'):
                    baseline_qty = float(qty_sum.to_eng_string())
                else:
                    baseline_qty = float(qty_sum)
            else:
                baseline_qty = 0.0
        except Exception as e:
            logger.warning(f"Error converting qty_ordered sum to float: {str(e)}")
            baseline_qty = 0.0
        
        daily_baseline_qty = baseline_qty / baseline_days if baseline_days > 0 else 0
        
        if daily_baseline_qty == 0 and not product_sales.empty:
            products_only_with_discounts.append(sku)
            continue
            
        if daily_baseline_qty == 0:
            daily_baseline_qty = 0.01
        
        all_promo_sales = product_sales[product_sales['rule_ids_list'].apply(lambda x: len(x) > 0)]
        
        if not all_promo_sales.empty:
            overall_promo_days = len(all_promo_sales['sale_date'].unique())
            if overall_promo_days == 0:
                overall_promo_days = 1
                
            try:
                overall_qty_sum = all_promo_sales['qty_ordered'].sum()
                if hasattr(overall_qty_sum, 'to_eng_string'):
                    overall_promo_qty = float(overall_qty_sum.to_eng_string())
                else:
                    overall_promo_qty = float(overall_qty_sum)
            except Exception as e:
                logger.warning(f"Error converting overall qty_ordered sum to float: {str(e)}")
                overall_promo_qty = 0.0
                
            daily_overall_promo_qty = overall_promo_qty / overall_promo_days
            
            overall_sensitivity_ratio = daily_overall_promo_qty / daily_baseline_qty
            
            sensitivity_data.append({
                'sku': sku,
                'discount_bucket': "Overall",
                'baseline_qty': daily_baseline_qty,
                'promo_qty': daily_overall_promo_qty,
                'sensitivity_ratio': overall_sensitivity_ratio,
                'baseline_days': baseline_days,
                'promo_days': overall_promo_days
            })
        
        promo_sales = product_sales[product_sales['rule_ids_list'].apply(lambda x: len(x) > 0)]
        
        if not promo_sales.empty:
            for bucket in promo_sales['discount_bucket'].unique():
                bucket_sales = promo_sales[promo_sales['discount_bucket'] == bucket]
                
                bucket_days = len(bucket_sales['sale_date'].unique())
                if bucket_days == 0:
                    bucket_days = 1
                
                try:
                    bucket_qty_sum = bucket_sales['qty_ordered'].sum()
                    if hasattr(bucket_qty_sum, 'to_eng_string'):
                        bucket_promo_qty = float(bucket_qty_sum.to_eng_string())
                    else:
                        bucket_promo_qty = float(bucket_qty_sum)
                except Exception as e:
                    logger.warning(f"Error converting bucket qty_ordered to float: {str(e)}")
                    bucket_promo_qty = 0.0
                
                daily_bucket_promo_qty = bucket_promo_qty / bucket_days
                
                bucket_sensitivity_ratio = daily_bucket_promo_qty / daily_baseline_qty
                
                sensitivity_data.append({
                    'sku': sku,
                    'discount_bucket': bucket,
                    'baseline_qty': daily_baseline_qty,
                    'promo_qty': daily_bucket_promo_qty,
                    'sensitivity_ratio': bucket_sensitivity_ratio,
                    'baseline_days': baseline_days,
                    'promo_days': bucket_days
                })
        
        all_buckets = ["≤15%", "20%", "25%", "30%", "≥35%", "Overall"]
        existing_buckets = [row['discount_bucket'] for row in sensitivity_data if row['sku'] == sku]
        
        for bucket in all_buckets:
            if bucket not in existing_buckets:
                sensitivity_data.append({
                    'sku': sku,
                    'discount_bucket': bucket,
                    'baseline_qty': float(daily_baseline_qty),
                    'promo_qty': 0.0,
                    'sensitivity_ratio': 1.0,
                    'baseline_days': baseline_days,
                    'promo_days': 0
                })
    
    result_df = pd.DataFrame(sensitivity_data)
    
    if not result_df.empty:
        result_df.attrs['products_only_with_discounts'] = products_only_with_discounts
        result_df.attrs['count_products_only_with_discounts'] = len(products_only_with_discounts)
    
    if use_cache and not result_df.empty:
        cache.set(filter_params, result_df)
        logger.info(f"Cached result with {len(result_df)} records in {time.time() - start_time:.2f} seconds")
    
    return result_df


def create_sensitivity_distribution_chart(sensitivity_data: pd.DataFrame) -> go.Figure:
    """
    Create a violin plot of sales sensitivity by discount bucket.
    
    Args:
        sensitivity_data: DataFrame with sales sensitivity data
        
    Returns:
        Plotly figure object
    """
    if sensitivity_data.empty or 'discount_bucket' not in sensitivity_data.columns or 'sensitivity_ratio' not in sensitivity_data.columns:
        return None
    
    bucket_order = ["≤15%", "20%", "25%", "30%", "≥35%", "Overall"]
    
    fig = go.Figure()
    
    bucket_colors = {
        "≤15%": "rgb(34,139,34)",
        "20%": "rgb(178,34,34)",
        "25%": "rgb(106,90,205)",
        "30%": "rgb(139,69,19)",
        "≥35%": "rgb(199,21,133)",
        "Overall": "rgb(0,0,139)"
    }
    
    for bucket in bucket_order:
        bucket_data = sensitivity_data[sensitivity_data['discount_bucket'] == bucket]
        
        if not bucket_data.empty:
            color = bucket_colors.get(bucket, "rgb(8,81,156)")
            
            fig.add_trace(go.Box(
                y=bucket_data['sensitivity_ratio'].clip(upper=10),
                name=bucket,
                marker=dict(
                    color=color,
                    opacity=0.95,
                    size=10,
                    outliercolor=color,
                    symbol='circle'
                ),
                line=dict(
                    color=color,
                    width=5
                ),
                boxpoints='outliers',
                hoverlabel=dict(
                    font_size=30,
                    font=dict()
                ),
            ))
    
    # Update layout
    fig.update_layout(
        title="Product Sensitivity to Discounts (Normalized by Days)",
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',
        xaxis=dict(
            title="Discount",
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            ticktext=bucket_order,
            tickvals=list(range(len(bucket_order))),
            gridcolor='rgba(0, 0, 0, 0.1)'
        ),
        yaxis=dict(
            title="Uplift Ratio (Daily Promo Sales / Daily Baseline Sales)",
            gridcolor='rgba(0, 0, 0, 0.1)',
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            autorange=True
        ),
        font=dict(color='rgba(0, 0, 0, 1)', size=25),
        height=1200,
        margin=dict(l=40, r=40, t=50, b=40),
        hoverlabel=dict(
            font_size=32,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            font=dict(color='rgba(0, 0, 0, 0.9)')
        ),
        showlegend=False
    )
    
    return fig


def display_sales_sensitivity_analysis(
    sales_df: pd.DataFrame,
    promotion_df: pd.DataFrame,
    filter_years: List[int] = None,
    filter_category_ids: List[int] = None,
    filter_customer_types: List[str] = None,
    filter_customer_status: List[str] = None
) -> go.Figure:
    """
    Display a sales sensitivity analysis by discount bucket.
    
    Args:
        sales_df: DataFrame with sales data
        promotion_df: DataFrame with promotion details
        filter_years: Years to filter by
        filter_category_ids: Category IDs to filter by
        filter_customer_types: Customer types to filter by (Guest/Registered)
        filter_customer_status: Customer status to filter by (New/Returning)
        
    Returns:
        Plotly figure object or None
    """
    if sales_df.empty or promotion_df.empty:
        st.error("No sales or promotion data available.")
        return None
    
    cache = SalesSensitivityCache()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        max_products = st.slider(
            "Number of products to analyze", 
            min_value=100, 
            max_value=500, 
            value=300,
            step=10,
            key="sensitivity_max_products_slider"
        )
    
    with col2:
        selection_method = st.selectbox(
            "Product selection",
            options=["top_revenue", "least_revenue", "random"],
            format_func=lambda x: {
                "top_revenue": "Highest revenue",
                "least_revenue": "Lowest revenue",
                "random": "Random"
            }[x],
            key="sensitivity_product_selection"
        )
    
    with st.spinner("Analyzing product sensitivity to discounts..."):
        sensitivity_data = calculate_sales_sensitivity(
            sales_df,
            promotion_df,
            filter_years=filter_years,
            filter_category_ids=filter_category_ids,
            filter_customer_types=filter_customer_types,
            filter_customer_status=filter_customer_status,
            max_products=max_products,
            selection_method=selection_method,
            use_cache=True
        )
        
        if sensitivity_data.empty:
            st.warning("No sensitivity data available for the selected filters.")
            return None
    
    with st.spinner("Generating sensitivity distribution chart..."):
        fig = create_sensitivity_distribution_chart(sensitivity_data)
        
        if fig is None:
            st.warning("Could not generate chart with the current data.")
            return None
            
        st.plotly_chart(fig, use_container_width=True)
    
    
    return fig