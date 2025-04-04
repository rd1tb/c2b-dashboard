"""
Product analysis dashboard view with tab-based navigation and preloaded data.
Removed problematic Streamlit caching to avoid unhashable type errors.
"""
import streamlit as st
import pandas as pd
import sys
import os
import time
# Add the project root to the Python path
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


def load_data(years, category_ids, discount_filter, query_repository):
    """
    Load and process data for product analysis.
    Removed Streamlit caching to avoid unhashable type errors.
    
    Args:
        years: Selected years
        category_ids: Selected category IDs
        discount_filter: Whether to include discounts
        query_repository: Query repository for database access
        
    Returns:
        Dictionary containing all required data
    """
    data = {}
    
    # Load products sold/unsold data
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
    
    # Filter by selected category IDs
    if data.get('products_sold_df') is not None and category_ids:
        data['products_sold_df'] = data['products_sold_df'][data['products_sold_df']['category_num'].isin(category_ids)]
    
    # Store the total number of products from selected categories
    if data.get('products_sold_df') is not None and not data['products_sold_df'].empty:
        data['total_products'] = data['products_sold_df']['total_products'].sum()
    else:
        data['total_products'] = 0
    
    # Load category sales data
    data['category_sales_df'] = query_repository.get_category_item_sales_with_rules_ids()
    
    if data.get('category_sales_df') is not None and not data['category_sales_df'].empty:
        # Convert order_date to datetime
        data['category_sales_df']['order_date'] = pd.to_datetime(data['category_sales_df']['order_date'])
        
        # Calculate sales_amount to prevent errors in downstream components
        if 'base_row_total_incl_tax' in data['category_sales_df'].columns and 'base_discount_amount' in data['category_sales_df'].columns:
            data['category_sales_df']['sales_amount'] = data['category_sales_df']['base_row_total_incl_tax'] - data['category_sales_df']['base_discount_amount']
        
        # Filter by selected years and categories
        data['filtered_sales'] = data['category_sales_df'][
            (data['category_sales_df']['order_date'].dt.year.isin(years)) &
            (data['category_sales_df']['category_num'].isin(category_ids))
        ]
        
        # Apply discount filter if needed
        if not discount_filter:
            data['filtered_sales'] = data['filtered_sales'][data['filtered_sales']['base_discount_amount'] == 0]
    else:
        data['filtered_sales'] = pd.DataFrame()
    
    return data


def display_product_view(query_repo: QueryRepository):
    """
    Display products section with tab-based navigation and preloaded data.
    Removed Streamlit caching to resolve unhashable type errors.
    """
    apply_custom_styles()

    st.header("Products")
    
    # Get selected filters from session state
    selected_years = st.session_state.selected_years
    selected_category_ids = st.session_state.selected_category_ids
    are_discounts_selected = st.session_state.selected_discounts
    
    # Display filter information
    year_str = ", ".join(map(str, selected_years))
    category_str = ", ".join(st.session_state.selected_category_names)
    discount_str = "Including discounts" if are_discounts_selected else "Excluding discounts"
    st.caption(f"Filtered by Years: {year_str} | Categories: {category_str} | {discount_str}")
    
    # Only proceed if we have valid selections
    if not selected_years or not selected_category_ids:
        st.warning("Please select at least one option for each filter.")
        return
    
    try:
        # Show loading message while fetching data
        with st.spinner("Loading product data..."):
            # Convert lists to tuples to avoid unhashable type errors
            # Our actual data loading function is now uncached
            data = load_data(
                selected_years, 
                selected_category_ids, 
                are_discounts_selected, 
                query_repo
            )
        
        # Products Overview section at the top of the page (not in tabs)
        st.subheader("Products Overview")
        if data.get('products_sold_df') is not None and not data.get('products_sold_df').empty and data.get('filtered_sales') is not None:
            display_products_sold_unsold(
                data['products_sold_df'], 
                data['filtered_sales'],
                data['total_products']
            )
        else:
            st.info("No product data available for the selected filters.")
            
        # Add some spacing
        st.write("")
        
        # Create tabs for other visualizations
        tabs = st.tabs([
            "Product Rotation", 
            "Distribution Analysis",
            "Basket Analysis"
        ])
        
        # Tab 1: Product Rotation Bubble Chart
        with tabs[0]:
            if data.get('filtered_sales') is not None and len(data['filtered_sales']) >= 5:
                st.subheader("Product Rotation Analysis")
                display_product_rotation_bubble_chart(data['filtered_sales'])
            else:
                st.info("Not enough products with multiple orders to generate the rotation chart.")
        
        # Tab 2: Distribution Analysis
        with tabs[1]:
            if data.get('filtered_sales') is not None and len(data['filtered_sales']) >= 5:
                st.subheader("Distribution analysis")
                display_product_distributions(data['filtered_sales'])
            else:
                st.info("Not enough products with multiple orders to generate the distribution plots.")
        
        # Tab 3: Basket Analysis
        with tabs[2]:
            if data.get('filtered_sales') is not None and len(data['filtered_sales']) >= 5:
                st.subheader("Product basket analysis")
                # Use our new optimized and properly cached basket analysis
                display_basket_analysis(data['filtered_sales'])
            else:
                st.info("Not enough products with multiple orders to generate the basket analysis.")
                
    except Exception as e:
        st.error(f"Error in product analysis: {str(e)}")
        st.info("Please try adjusting your filters or contact support if the problem persists.")