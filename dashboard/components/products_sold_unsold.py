"""
Products sold and unsold metrics component with discount filtering and caching.
"""
import streamlit as st
import pandas as pd


@st.cache_data(ttl=7200, show_spinner=False)
def calculate_product_metrics(products_sold: pd.DataFrame, filtered_sales: pd.DataFrame, total_products: int):
    """
    Calculate product metrics with caching.
    
    Args:
        products_sold: DataFrame containing the products sold/unsold data
        filtered_sales: DataFrame containing the filtered sales data (with/without discounts)
        total_products: Total number of products in selected categories
        
    Returns:
        Dictionary containing the product metrics
    """
    if products_sold.empty or filtered_sales.empty:
        return {
            "total_products": 0,
            "unique_products_sold": 0,
            "single_order_products": 0,
            "products_not_sold": 0
        }
    
    # Get unique products sold (after discount filtering)
    unique_products_sold = filtered_sales['sku'].nunique()
    
    # Calculate products not sold
    products_not_sold = total_products - unique_products_sold
    
    # Get products sold only once
    product_order_counts = filtered_sales.groupby('sku')['order_id'].nunique().reset_index()
    single_order_products = len(product_order_counts[product_order_counts['order_id'] == 1])
    
    return {
        "total_products": total_products,
        "unique_products_sold": unique_products_sold,
        "single_order_products": single_order_products,
        "products_not_sold": products_not_sold
    }


def display_products_sold_unsold(products_sold: pd.DataFrame, filtered_sales: pd.DataFrame, total_products: int):
    """
    Display products sold and unsold metrics with discount filtering.
    
    Args:
        products_sold: DataFrame containing the products sold/unsold data
        filtered_sales: DataFrame containing the filtered sales data (with/without discounts)
        total_products: Total number of products in selected categories
    """
    if products_sold.empty or filtered_sales.empty:
        st.info("No product data available for the selected criteria.")
        return
    
    # Calculate metrics using cached function
    metrics = calculate_product_metrics(products_sold, filtered_sales, total_products)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total number of products", value=f"{metrics['total_products']:,.0f}")
    
    with col2:
        st.metric(label="Products sold once", value=f"{metrics['single_order_products']:,.0f}")

    with col3:
        st.metric(label="Products never sold", value=f"{metrics['products_not_sold']:,.0f}")