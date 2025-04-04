"""
Sales overview dashboard view with improved architecture and tabbed interface.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import numpy as np
# Add the project root to the Python path
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
            font-size: 20px !important;
            font-weight: bold !important;
        }
        
        /* Make tab names bigger */
        button[data-baseweb="tab"] p {
            font-size: 18px !important;
            font-weight: bold !important;
        }
        
        /* Increase filter caption text size */
        .st-emotion-cache-1c7y2kd {
            font-size: 20px !important;
        }
        
        [data-testid="stMetricLabel"] > div {
        font-size: 30px !important;
        font-weight: bold !important;
        }
        
        /* Make metric values bigger */
        [data-testid="stMetricValue"] {
            font-size: 40px !important;
        }
    
        div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
        }
        div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
        }
        div[class*="stSelectbox"] .st-bf {
        font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

def display_sales_view(query_repo: QueryRepository):
    """
    Display sales overview section with tabbed interface.
    
    Args:
        query_repo: Query repository instance
    """
    apply_custom_styles()
    st.header("Sales Overview")
    
    # Get selected filters from session state
    selected_years = st.session_state.selected_years
    selected_category_ids = st.session_state.selected_category_ids
    are_discounts_selected = st.session_state.selected_discounts
    
    # Display filter information
    year_str = ", ".join(map(str, selected_years))
    category_str = ", ".join(st.session_state.selected_category_names)
    st.caption(f"Filtered by Years: {year_str} | Categories: {category_str}")
    
    # Get the sales data first
    try:
        category_sales_df = query_repo.get_category_item_sales_with_rules_ids()
        
        # Filter data by selected categories
        if selected_category_ids:
            category_sales_df = category_sales_df[category_sales_df['category_num'].isin(selected_category_ids)]
        
        if not are_discounts_selected:
            category_sales_df = category_sales_df[category_sales_df['base_discount_amount'] == 0]
        
        # Convert dates to datetime and filter by selected years
        if not category_sales_df.empty:
            category_sales_df['order_date'] = pd.to_datetime(category_sales_df['order_date'])
            if selected_years:
                category_sales_df = category_sales_df[category_sales_df['order_date'].dt.year.isin(selected_years)]

            if not are_discounts_selected:
                category_sales_df = category_sales_df[category_sales_df['base_discount_amount'] <= 0]
            
            # Calculate sales amount (total minus discount)
            category_sales_df['sales_amount'] = category_sales_df['base_row_total_incl_tax'] - category_sales_df['base_discount_amount']
            
            # Display revenue and quantity metrics (always on top)
            display_total_revenue_quantity(category_sales_df)
            
            # Add some spacing
            st.write("")
            
            # Create tabs for visualizations
            tabs = st.tabs([
                "Monthly Sales Trend",
                "Sales Growth Rate",
                "Sales Decomposition"
            ])
            
            # Tab 1: Monthly Sales Trend
            with tabs[0]:
                st.subheader("Monthly Sales Trend")
                display_monthly_sales_chart(category_sales_df)
            
            # Tab 2: Sales Growth Rate
            with tabs[1]:
                display_sales_growth_rate(category_sales_df)
            
            # Tab 3: Sales Decomposition
            with tabs[2]:
                display_time_series_decomposition(category_sales_df)
            
        else:
            st.info("No sales data available for the selected filters.")
    except Exception as e:
        st.error(f"Error loading sales data: {str(e)}")
        st.info("This visualization requires category sales data. Please ensure you have access to this data.")