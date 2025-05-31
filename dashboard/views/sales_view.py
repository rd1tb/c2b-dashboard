"""
Sales overview dashboard view with improved architecture and tabbed interface.
Updated to include customer type filtering.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from database import QueryRepository
from dashboard.components.monthly_sales_chart import display_monthly_sales_chart
from dashboard.components.total_revenue_quantity import display_total_revenue_quantity
from dashboard.components.sales_analysis import display_sales_analysis, display_time_series_decomposition, display_sales_growth_rate

def apply_custom_styles():
    """Apply custom CSS styles to make UI elements larger and bolder."""
    st.markdown("""
    <style>
        /* Increase size for page navigation */
        .st-emotion-cache-z5fcl4 {
            font-size: 22px !important;
            font-weight: bold !important;
        }
        
        /* Make tab names bigger */
        button[data-baseweb="tab"] p {
            font-size: 22px !important;
            font-weight: bold !important;
        }
        
        /* Increase filter caption text size */
        .st-emotion-cache-1c7y2kd {
            font-size: 22px !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 20px !important;
            font-weight: bold !important;
        }
        
        [data-testid="stMetricLabel"] * {
            font-size: inherit !important;
            font-weight: inherit !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 40px !important;
        }
    
        div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 20px;
        }
        div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 20px;
        }
        div[class*="stSelectbox"] .st-bf {
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)

def display_sales_view(query_repo: QueryRepository):
    """
    Display sales overview section with tabbed interface.
    Updated to include customer type filtering.
    
    Args:
        query_repo: Query repository instance
    """
    apply_custom_styles()
    st.header("Sales Overview")
    
    selected_years = st.session_state.selected_years
    selected_category_ids = st.session_state.selected_category_ids
    are_discounts_selected = st.session_state.selected_discounts
    selected_customer_types = st.session_state.selected_customer_types
    selected_customer_status = st.session_state.selected_customer_status
    
    year_str = ", ".join(map(str, selected_years))
    category_str = ", ".join(st.session_state.selected_category_names)
    customer_type_str = ", ".join(selected_customer_types)
    customer_status_str = ", ".join(selected_customer_status)
    
    st.caption(f"Filtered by Years: {year_str} | Categories: {category_str} | Customer Types: {customer_type_str} | Customer Status: {customer_status_str}")
    
    try:
        category_sales_df = query_repo.get_enhanced_category_item_sales()
        
        if selected_category_ids:
            category_sales_df = category_sales_df[category_sales_df['category_num'].isin(selected_category_ids)]
        
        if not are_discounts_selected:
            category_sales_df = category_sales_df[category_sales_df['base_discount_amount'] == 0]
        
        if not category_sales_df.empty:
            category_sales_df['order_date'] = pd.to_datetime(category_sales_df['order_date'])
            if selected_years:
                category_sales_df = category_sales_df[category_sales_df['order_date'].dt.year.isin(selected_years)]

            if not are_discounts_selected:
                category_sales_df = category_sales_df[category_sales_df['base_discount_amount'] <= 0]
            
            category_sales_df['sales_amount'] = category_sales_df['base_row_total_incl_tax'] - category_sales_df['base_discount_amount']
            
            if 'Guest' in selected_customer_types and 'Registered' in selected_customer_types:
                pass
            elif 'Guest' in selected_customer_types:
                category_sales_df = category_sales_df[category_sales_df['customer_is_guest'] == 1]
            elif 'Registered' in selected_customer_types:
                category_sales_df = category_sales_df[category_sales_df['customer_group_id'].isin([1, 2, 7, 8, 9])]
            else:
                category_sales_df = category_sales_df.head(0)
                
            if selected_customer_status and not category_sales_df.empty:
                customer_order_counts = category_sales_df.groupby('hashed_customer_email')['order_id'].nunique().reset_index()
                customer_order_counts.columns = ['hashed_customer_email', 'unique_order_count']
                
                new_customers = set(customer_order_counts[customer_order_counts['unique_order_count'] == 1]['hashed_customer_email'].tolist())
                returning_customers = set(customer_order_counts[customer_order_counts['unique_order_count'] > 1]['hashed_customer_email'].tolist())
                
                if 'New' in selected_customer_status and 'Returning' in selected_customer_status:
                    pass
                elif 'New' in selected_customer_status:
                    category_sales_df = category_sales_df[category_sales_df['hashed_customer_email'].isin(new_customers)]
                elif 'Returning' in selected_customer_status:
                    category_sales_df = category_sales_df[category_sales_df['hashed_customer_email'].isin(returning_customers)]
                else:
                    category_sales_df = category_sales_df.head(0)
            
            display_total_revenue_quantity(category_sales_df)
            st.write("")
            tabs = st.tabs([
                "Monthly Sales Trend",
                "Sales Growth Rate",
                "Sales Decomposition"
            ])
            
            with tabs[0]:
                display_monthly_sales_chart(category_sales_df)
            
            with tabs[1]:
                display_sales_growth_rate(category_sales_df)
            
            with tabs[2]:
                display_time_series_decomposition(category_sales_df)
            
        else:
            st.info("No sales data available for the selected filters.")
    except Exception as e:
        st.error(f"Error loading sales data: {str(e)}")
        st.info("This visualization requires category sales data. Please ensure you have access to this data.")
