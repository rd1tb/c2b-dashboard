"""
Main Streamlit application for the sales dashboard.

To run:
    streamlit run dashboard/streamlit_app.py
"""
import streamlit as st
import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.utils import init_session_state, display_login_form, logout
from dashboard.views.sales_view import display_sales_view
from dashboard.views.product_view import display_product_view
from dashboard.views.promotion_view import display_promotion_view

# Set page config
st.set_page_config(
    page_title="Sales Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define colors
BACKGROUND_COLOR = "#0B0E1F"
CARD_BACKGROUND = "#1A283E"  # Slightly lighter blue for info cards
TEXT_COLOR = "white"
ACCENT_COLOR = "#4C8BF5"  # Blue accent color for buttons

st.markdown(
    f"""
    <style>
    body {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}
    .stButton>button {{
        background-color: {CARD_BACKGROUND} !important;
        color: {TEXT_COLOR} !important;
        border: none !important;
        width: 100%;
    }}
    .stProgress>div>div {{
        background-color: {ACCENT_COLOR};
    }}
    .stDataFrame, .stTable {{
        background-color: {CARD_BACKGROUND};
    }}
    .stInfo {{
        background-color: {CARD_BACKGROUND} !important;
        color: {TEXT_COLOR} !important;
        padding: 20px !important;
        border-radius: 5px !important;
        border: none !important;
    }}
    [data-testid="stButton"][kind="secondary"] {{
    font-size: 14px;
    padding: 4px 8px;
    }}

    [data-testid="stButton"][kind="primary"] {{
    font-size: 18px;
    padding: 4px 8px;
    }}
    /* Set wider sidebar for better navigation display */
    [data-testid="stSidebar"] {{
        min-width: 350px !important;
        width: 350px !important;
    }}
    /* Fix checkbox alignment */
    .stCheckbox label p {{
        font-size: 20px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def show_filters():
    """Show global filters in sidebar."""
    st.sidebar.header("Data Filters")
    
    # Initialize session state for filters if not present
    if "selected_years" not in st.session_state:
        st.session_state.selected_years = [2023, 2024]
    if "selected_category_ids" not in st.session_state:
        st.session_state.selected_category_ids = [694, 685]
    if "selected_category_names" not in st.session_state:
        st.session_state.selected_category_names = ["Face Cream", "Sunscreen"]
    if "selected_discounts" not in st.session_state:
        st.session_state.selected_discounts = True
    
    # Define callbacks for year checkboxes
    def on_year_2023_change():
        if 2023 in st.session_state.selected_years:
            st.session_state.selected_years.remove(2023)
        else:
            st.session_state.selected_years.append(2023)
            
    def on_year_2024_change():
        if 2024 in st.session_state.selected_years:
            st.session_state.selected_years.remove(2024)
        else:
            st.session_state.selected_years.append(2024)
    
    # Define callbacks for category checkboxes
    def on_face_cream_change():
        if 694 in st.session_state.selected_category_ids:
            st.session_state.selected_category_ids.remove(694)
            if "Face Cream" in st.session_state.selected_category_names:
                st.session_state.selected_category_names.remove("Face Cream")
        else:
            st.session_state.selected_category_ids.append(694)
            if "Face Cream" not in st.session_state.selected_category_names:
                st.session_state.selected_category_names.append("Face Cream")
    
    def on_sunscreen_change():
        if 685 in st.session_state.selected_category_ids:
            st.session_state.selected_category_ids.remove(685)
            if "Sunscreen" in st.session_state.selected_category_names:
                st.session_state.selected_category_names.remove("Sunscreen")
        else:
            st.session_state.selected_category_ids.append(685)
            if "Sunscreen" not in st.session_state.selected_category_names:
                st.session_state.selected_category_names.append("Sunscreen")
    def on_discount_toggle_change():
    # Toggle the value when the callback is triggered
        st.session_state.selected_discounts = not st.session_state.selected_discounts
    
    # Year filter - using columns to place them side by side
    st.sidebar.subheader("Select Year(s)")
    year_cols = st.sidebar.columns(2)
    
    # Year checkboxes side by side with callbacks
    with year_cols[0]:
        st.checkbox("2023", 
                   value=2023 in st.session_state.selected_years, 
                   key="year_2023_cb", 
                   on_change=on_year_2023_change)
            
    with year_cols[1]:
        st.checkbox("2024", 
                   value=2024 in st.session_state.selected_years, 
                   key="year_2024_cb", 
                   on_change=on_year_2024_change)
    
    # Category filter - also using columns
    st.sidebar.subheader("Select Category(s)")
    category_cols = st.sidebar.columns(2)
    
    # Category checkboxes side by side with callbacks
    with category_cols[0]:
        st.checkbox("Face Cream", 
                   value=694 in st.session_state.selected_category_ids, 
                   key="face_cream_cb", 
                   on_change=on_face_cream_change)
    
    with category_cols[1]:
        st.checkbox("Sunscreen", 
                   value=685 in st.session_state.selected_category_ids, 
                   key="sunscreen_cb", 
                   on_change=on_sunscreen_change)
        
    # Discount toggle - using a direct approach without callbacks
    st.sidebar.subheader("Discounts")
    st.sidebar.checkbox(
        "Include discounted orders",
        value=st.session_state.selected_discounts,
        key="discount_toggle",
        on_change=on_discount_toggle_change
    )
    return st.session_state.selected_years, st.session_state.selected_category_ids, st.session_state.selected_discounts

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Initialize filter state variables if they don't exist
    if "selected_years" not in st.session_state:
        st.session_state.selected_years = [2023, 2024]
    if "selected_category_ids" not in st.session_state:
        st.session_state.selected_category_ids = [694, 685]
    if "selected_category_names" not in st.session_state:
        st.session_state.selected_category_names = ["Face Cream", "Sunscreen"]
    if "selected_discounts" not in st.session_state:
        st.session_state.selected_discounts = True
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Sales Overview"
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        display_login_form()
    else:
        # Navigation in sidebar - use full-width buttons
        st.sidebar.title("Navigation")
        
        if st.sidebar.button("📈 Sales Overview", use_container_width=True, type="primary"):
            st.session_state.current_page = "Sales Overview"
            
        if st.sidebar.button("🛍️ Product Analysis", use_container_width=True, type="primary"):
            st.session_state.current_page = "Product Analysis"
            
        if st.sidebar.button("🏷️ Promotion Analysis", use_container_width=True, type="primary"):
            st.session_state.current_page = "Promotion Analysis"
            
        # Separator
        st.sidebar.markdown("---")
        
        # Show filters in sidebar
        selected_years, selected_category_ids, selected_discounts = show_filters()
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", use_container_width=True, type="secondary"):
            logout()
        
        # Display selected view
        # Display a message if no filters are selected
        if not selected_years or not selected_category_ids:
            st.warning("Please select at least one option for each filter.")
            return
            
        query_repo = st.session_state.query_repo
        
        if st.session_state.current_page == "Sales Overview":
            display_sales_view(query_repo)
        elif st.session_state.current_page == "Product Analysis":
            display_product_view(query_repo)
        elif st.session_state.current_page == "Promotion Analysis":
            display_promotion_view(query_repo)


if __name__ == "__main__":
    main()