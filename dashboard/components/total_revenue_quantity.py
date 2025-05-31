"""
Total revenue and quantity metrics component.
"""
import streamlit as st
import pandas as pd


def display_total_revenue_quantity(category_sales: pd.DataFrame):
    """
    Display total revenue and quantity metrics.
    
    Args:
        category_sales: DataFrame containing the filtered sales data
    """
    if category_sales.empty:
        st.info("No sales data available for the selected criteria.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        total_revenue = category_sales['sales_amount'].sum()
        st.metric(label="Revenue", value=f"€{total_revenue:,.0f}")
    
    with col2:
        total_products_sold = category_sales['qty_ordered'].sum()
        st.metric(label="Products Sold", value=f"{total_products_sold:,.0f}")