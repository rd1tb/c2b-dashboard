"""
Updated promotion and campaign analysis dashboard view.
Includes new Sales Uplift Analysis component.
"""
import streamlit as st
import pandas as pd
import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from database import QueryRepository
from dashboard.components.promotion_metrics import display_promotion_duration_metrics
from dashboard.components.promotion_chart import create_promotion_usage_chart
from dashboard.components.combined_promotion_chart import display_combined_promotion_chart
from dashboard.components.sales_uplift_analysis import display_sales_sensitivity_analysis


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


def display_promotion_view(query_repo: QueryRepository):
    """
    Display promotion and campaign analysis section.
    
    Args:
        query_repo: Query repository instance
    """
    apply_custom_styles()
    
    st.header("Promotion & Campaign Analysis")
    
    try:
        # Get promotion duration statistics
        promotion_stats = query_repo.get_promotion_duration_stats()
        if not promotion_stats.empty:
            display_promotion_duration_metrics(promotion_stats)
        else:
            st.info("No promotion statistics available.")
        
        # Get promotion details for visualization
        promotion_details = query_repo.get_promotion_details()
        
        if promotion_details.empty:
            st.info("No promotion details available for visualization.")
            return
        
        selected_years = st.session_state.selected_years
        selected_category_ids = st.session_state.selected_category_ids
        
        # Get the sales data
        category_sales_df = query_repo.get_category_item_sales_with_rules_ids()
        
        # Filter data by selected categories
        if selected_category_ids:
            category_sales_df = category_sales_df[category_sales_df['category_num'].isin(selected_category_ids)]
        
        # Convert dates to datetime and filter by selected years
        if not category_sales_df.empty:
            category_sales_df['order_date'] = pd.to_datetime(category_sales_df['order_date'])
            if selected_years:
                category_sales_df = category_sales_df[category_sales_df['order_date'].dt.year.isin(selected_years)]
        
        # Create tabs for different visualizations
        promotion_tabs = st.tabs(["Promotion Performance", "Product Discount Sensitivity"])
        
        with promotion_tabs[0]:
            # Display the combined promotion chart
            if category_sales_df.empty:
                st.info("No sales data available for the selected filters.")
            else:
                # Use the combined promotion chart
                display_combined_promotion_chart(category_sales_df, promotion_details, selected_years, selected_category_ids)
        
        with promotion_tabs[1]:
            # Add the new Sales Uplift Analysis visualization
            if category_sales_df.empty:
                st.info("No sales data available for the selected filters.")
            else:
                # Use the new sales sensitivity analysis
                display_sales_sensitivity_analysis(category_sales_df, promotion_details, selected_years, selected_category_ids)
        
    except Exception as e:
        st.error(f"Error analyzing promotion data: {str(e)}")