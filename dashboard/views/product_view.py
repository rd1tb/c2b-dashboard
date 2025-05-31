"""
Product analysis dashboard view with tab-based navigation and preloaded data.
Updated to include customer type filtering and fix bubble chart display issues.
"""
import streamlit as st
import pandas as pd
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from database import QueryRepository
from dashboard.components.products_sold_unsold import display_products_sold_unsold
from dashboard.components.product_rotation_charts import (
    display_product_rotation_bubble_chart,
    display_product_distributions
)
from dashboard.components.basket_analysis import display_basket_analysis


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


def load_data(years, category_ids, discount_filter, customer_types, customer_status, query_repository):
    """
    Load and process data for product analysis.
    Updated to include customer type filtering.
    
    Args:
        years: Selected years
        category_ids: Selected category IDs
        discount_filter: Whether to include discounts
        customer_types: Selected customer types (Guest, Registered)
        customer_status: Selected customer status (New, Returning)
        query_repository: Query repository for database access
        
    Returns:
        Dictionary containing all required data
    """
    data = {}

    if years:
        if len(years) == 1:
            if 2023 in years:
                data['products_sold_df'] = query_repository.get_number_of_products_2023()
            elif 2024 in years:
                data['products_sold_df'] = query_repository.get_number_of_products_2024()
            else:
                data['products_sold_df'] = query_repository.get_cross_year_product_metrics()
        else:
            data['products_sold_df'] = query_repository.get_cross_year_product_metrics()

    if data.get('products_sold_df') is not None and category_ids:
        data['products_sold_df'] = data['products_sold_df'][data['products_sold_df']['category_num'].isin(category_ids)]

    if data.get('products_sold_df') is not None and not data['products_sold_df'].empty:
        data['total_products'] = data['products_sold_df']['total_products'].sum()
    else:
        data['total_products'] = 0

    data['category_sales_df'] = query_repository.get_enhanced_category_item_sales()

    if data.get('category_sales_df') is not None and not data['category_sales_df'].empty:
        data['category_sales_df']['order_date'] = pd.to_datetime(data['category_sales_df']['order_date'])
        if 'base_row_total_incl_tax' in data['category_sales_df'].columns and 'base_discount_amount' in data['category_sales_df'].columns:
            data['category_sales_df']['sales_amount'] = data['category_sales_df']['base_row_total_incl_tax'] - data['category_sales_df']['base_discount_amount']
        data['filtered_sales'] = data['category_sales_df'][
            (data['category_sales_df']['order_date'].dt.year.isin(years)) &
            (data['category_sales_df']['category_num'].isin(category_ids))
        ]
        if not discount_filter:
            data['filtered_sales'] = data['filtered_sales'][data['filtered_sales']['base_discount_amount'] == 0]
        if customer_types:
            if 'Guest' in customer_types and 'Registered' in customer_types:
                pass
            elif 'Guest' in customer_types:
                data['filtered_sales'] = data['filtered_sales'][data['filtered_sales']['customer_is_guest'] == 1]
            elif 'Registered' in customer_types:
                data['filtered_sales'] = data['filtered_sales'][data['filtered_sales']['customer_group_id'].isin([1, 2, 7, 8, 9])]
            else:
                data['filtered_sales'] = data['filtered_sales'].head(0)
        if customer_status and data['filtered_sales'] is not None and not data['filtered_sales'].empty:
            customer_order_counts = data['filtered_sales'].groupby('hashed_customer_email')['order_id'].nunique().reset_index()
            customer_order_counts.columns = ['hashed_customer_email', 'unique_order_count']
            new_customers = set(customer_order_counts[customer_order_counts['unique_order_count'] == 1]['hashed_customer_email'].tolist())
            returning_customers = set(customer_order_counts[customer_order_counts['unique_order_count'] > 1]['hashed_customer_email'].tolist())
            if 'New' in customer_status and 'Returning' in customer_status:
                pass
            elif 'New' in customer_status:
                data['filtered_sales'] = data['filtered_sales'][data['filtered_sales']['hashed_customer_email'].isin(new_customers)]
            elif 'Returning' in customer_status:
                data['filtered_sales'] = data['filtered_sales'][data['filtered_sales']['hashed_customer_email'].isin(returning_customers)]
            else:
                data['filtered_sales'] = data['filtered_sales'].head(0)
    else:
        data['filtered_sales'] = pd.DataFrame()

    return data


def display_product_view(query_repo: QueryRepository):
    """
    Display products section with tab-based navigation and preloaded data.
    Updated to use customer type filters and fix bubble chart display issues.
    """
    apply_custom_styles()

    st.header("Products")
    
    selected_years = st.session_state.selected_years if "selected_years" in st.session_state else [2023, 2024]
    selected_category_ids = st.session_state.selected_category_ids if "selected_category_ids" in st.session_state else [694, 685]
    are_discounts_selected = st.session_state.selected_discounts if "selected_discounts" in st.session_state else True
    selected_customer_types = st.session_state.selected_customer_types if "selected_customer_types" in st.session_state else ["Guest", "Registered"]
    selected_customer_status = st.session_state.selected_customer_status if "selected_customer_status" in st.session_state else ["New", "Returning"]
    
    year_str = ", ".join(map(str, selected_years))
    category_str = ", ".join(st.session_state.selected_category_names) if "selected_category_names" in st.session_state else "Face Cream, Sunscreen"
    discount_str = "Including discounts" if are_discounts_selected else "Excluding discounts"
    customer_type_str = ", ".join(selected_customer_types)
    customer_status_str = ", ".join(selected_customer_status)
    
    st.caption(f"Filtered by Years: {year_str} | Categories: {category_str} | {discount_str} | Customer Types: {customer_type_str} | Customer Status: {customer_status_str}")
    
    if not selected_years or not selected_category_ids:
        st.warning("Please select at least one option for each filter.")
        return
    
    try:
        with st.spinner("Loading product data..."):
            data = load_data(
                selected_years, 
                selected_category_ids, 
                are_discounts_selected,
                selected_customer_types,
                selected_customer_status,
                query_repo
            )
        
        if (data.get('products_sold_df') is not None and 
            not data.get('products_sold_df').empty and 
            data.get('filtered_sales') is not None and
            not data.get('filtered_sales').empty and
            'total_products' in data and 
            data['total_products'] > 0):
            
            display_products_sold_unsold(
                data['products_sold_df'], 
                data['filtered_sales'],
                data['total_products']
            )
        else:
            st.info("No product data available for the selected filters.")
            
        st.write("")
        
        tabs = st.tabs([
            "Product Rotation", 
            "Distribution Analysis",
            "Basket Analysis"
        ])
        
        with tabs[0]:
            if data.get('filtered_sales') is not None and len(data['filtered_sales']) >= 5:
                try:
                    display_product_rotation_bubble_chart(data['filtered_sales'])
                except Exception as e:
                    st.error(f"Error generating bubble chart: {str(e)}")
                    st.info("Please check that your data contains enough valid products with sales metrics.")
            else:
                st.info("Not enough products with multiple orders to generate the rotation chart. Please adjust your filters to include more data.")
        
        with tabs[1]:
            if data.get('filtered_sales') is not None and len(data['filtered_sales']) >= 5:
                try:
                    display_product_distributions(data['filtered_sales'])
                except Exception as e:
                    st.error(f"Error generating distribution plots: {str(e)}")
                    st.info("Please check that your data contains valid distribution metrics.")
            else:
                st.info("Not enough products with multiple orders to generate the distribution plots.")
        
        with tabs[2]:
            if data.get('filtered_sales') is not None and len(data['filtered_sales']) >= 5:
                try:
                    display_basket_analysis(data['filtered_sales'])
                except Exception as e:
                    st.error(f"Error generating basket analysis: {str(e)}")
                    st.info("Please check that your data contains valid transaction information.")
            else:
                st.info("Not enough products with multiple orders to generate the basket analysis.")
                
    except Exception as e:
        st.error(f"Error in product analysis: {str(e)}")
        st.info("Please try adjusting your filters or contact support if the problem persists.")