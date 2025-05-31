"""
Updated promotion and campaign analysis dashboard view.
Includes customer type filtering.
"""
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from database import QueryRepository
from dashboard.components.promotion_metrics import display_promotion_metrics
from dashboard.components.combined_promotion_chart import display_combined_promotion_chart
from dashboard.components.promotions_uplift_analysis import display_sales_sensitivity_analysis


def apply_custom_styles():
    """Apply custom CSS styles to make UI elements larger and bolder."""
    st.markdown("""
    <style>
        .st-emotion-cache-z5fcl4 {
            font-size: 22px !important;
            font-weight: bold !important;
        }
        button[data-baseweb="tab"] p {
            font-size: 22px !important;
            font-weight: bold !important;
        }
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


def display_promotion_view(query_repo: QueryRepository):
    """
    Display promotion and campaign analysis section.
    Updated to include all five metrics in one row and pass customer type filters to all components.
    
    Args:
        query_repo: Query repository instance
    """
    apply_custom_styles()
    
    st.header("Promotion & Campaign Analysis")
    
    try:
        promotion_stats = query_repo.get_promotion_duration_stats()
        promotion_details = query_repo.get_promotion_details()
        
        if not promotion_stats.empty:
            display_promotion_metrics(promotion_stats, promotion_details)
        else:
            st.info("No promotion statistics available.")
            
        if promotion_details.empty:
            st.info("No promotion details available for visualization.")
            return
        
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
        
        category_sales_df = query_repo.get_enhanced_category_item_sales()
        
        if selected_category_ids:
            category_sales_df = category_sales_df[category_sales_df['category_num'].isin(selected_category_ids)]
        
        if not category_sales_df.empty:
            category_sales_df['order_date'] = pd.to_datetime(category_sales_df['order_date'])
            if selected_years:
                category_sales_df = category_sales_df[category_sales_df['order_date'].dt.year.isin(selected_years)]
                
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
        
        promotion_tabs = st.tabs(["Promotion Performance", "Product Discount Sensitivity"])
        
        with promotion_tabs[0]:
            if category_sales_df.empty:
                st.info("No sales data available for the selected filters.")
            else:
                display_combined_promotion_chart(
                    category_sales_df, 
                    promotion_details, 
                    selected_years, 
                    selected_category_ids,
                    selected_customer_types,
                    selected_customer_status
                )
        
        with promotion_tabs[1]:
            if category_sales_df.empty:
                st.info("No sales data available for the selected filters.")
            else:
                display_sales_sensitivity_analysis(
                    category_sales_df, 
                    promotion_details, 
                    selected_years, 
                    selected_category_ids,
                    selected_customer_types,
                    selected_customer_status
                )
        
    except Exception as e:
        st.error(f"Error analyzing promotion data: {str(e)}")
