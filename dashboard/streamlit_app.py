"""
Main Streamlit application for the sales dashboard.

To run:
    streamlit run dashboard/streamlit_app.py
"""
import streamlit as st
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.utils import init_session_state, display_login_form, logout
from dashboard.views.sales_view import display_sales_view
from dashboard.views.product_view import display_product_view
from dashboard.views.promotion_view import display_promotion_view

st.set_page_config(
    page_title="Sales Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKGROUND_COLOR = "#F5F0E6"
CARD_BACKGROUND = "#EDE6D9"
TEXT_COLOR = "#2C2C2C"
ACCENT_COLOR = "#1A365D"
ACCENT_HOVER = "#2C5282"
SECONDARY_COLOR = "#4A5568"
SECONDARY_HOVER = "#2D3748"

st.markdown(
    f"""
    <style>
    /* Set default text color for all elements except specific components */
    *:not([data-baseweb="select"] *):not([data-baseweb="popover"] *):not([data-baseweb="menu"] *):not([data-baseweb="option"] *):not([role="listbox"] *):not([role="option"] *):not([data-testid="stButton"] > button *):not([data-testid="stButton"] > button) {{
        color: {TEXT_COLOR} !important;
    }}
    
    body {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}
    
    /* Button styling similar to metric labels */
    button {{
        font-size: 22px !important;
        padding: 10px !important;
        font-weight: 500 !important;
    }}
    
    button * {{
        font-size: inherit !important;
        font-weight: inherit !important;
    }}
    
    .stButton > button {{
        background-color: {ACCENT_COLOR} !important;
        color: white !important;
        border: none !important;
        width: 100%;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }}
    
    .stButton > button:hover {{
        background-color: {ACCENT_HOVER} !important;
    }}
    
    .stProgress>div>div {{
        background-color: {ACCENT_COLOR};
    }}
    .stDataFrame, .stTable {{
        background-color: {CARD_BACKGROUND};
        border: 1px solid #D4C9B8;
        color: {TEXT_COLOR};
    }}
    .stInfo {{
        background-color: {CARD_BACKGROUND} !important;
        color: {TEXT_COLOR} !important;
        padding: 20px !important;
        border-radius: 5px !important;
        border: 1px solid #D4C9B8 !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {CARD_BACKGROUND} !important;
        min-width: 450px !important;
        width: 450px !important;
        border-right: 1px solid #D4C9B8;
    }}
    .stCheckbox label p,
    .stCheckbox label div,
    .stCheckbox label span {{
        font-size: 32px !important;
        color: {TEXT_COLOR} !important;
        font-weight: bold !important;
    }}
    
    .spacer {{
        margin-top: 20px;
        margin-bottom: 20px;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_COLOR} !important;
        font-size: 32px !important;
    }}
    .sidebar .sidebar-content {{
        background-color: {CARD_BACKGROUND};
    }}
    .sidebar-header {{
        color: {TEXT_COLOR} !important;
        font-size: 28px !important;
    }}
    button[data-baseweb="tab"] p {{
        font-size: 26px !important;
        font-weight: bold !important;
    }}
    
    button[data-baseweb="tab"] {{
        background-color: transparent !important;
        color: #666666 !important;
        opacity: 1 !important;
        border: none !important;
        font-size: 26px !important;
    }}
    
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: transparent !important;
        color: #C0392B !important;
        opacity: 1 !important;
        border: none !important;
        font-size: 26px !important;
    }}
    
    .st-emotion-cache-1c7y2kd {{
        font-size: 28px !important;
        color: {TEXT_COLOR} !important;
    }}
    
    [data-testid="stButton"][kind="primary"] {{
        background-color: {ACCENT_COLOR} !important;
        color: white !important;
        font-size: 24px !important;
        padding: 12px 8px !important;
        transition: background-color 0.2s ease;
    }}
    
    [data-testid="stButton"][kind="secondary"] {{
        background-color: {SECONDARY_COLOR} !important;
        color: white !important;
        font-size: 24px !important;
        padding: 12px 8px !important;
        transition: background-color 0.2s ease;
    }}
    
    .stSubheader {{
        font-size: 28px !important;
        color: {TEXT_COLOR} !important;
    }}
    
    .stMarkdown p {{
        font-size: 20px !important;
        color: {TEXT_COLOR} !important;
    }}
    
    .stInfo, .stWarning {{
        font-size: 20px !important;
        color: {TEXT_COLOR} !important;
    }}

    [data-testid="stSidebar"] [data-testid="stButton"] > button p,
    [data-testid="stSidebar"] [data-testid="stButton"] > button div,
    [data-testid="stSidebar"] [data-testid="stButton"] > button span {{
        font-size: 30px !important;
        color: white !important;
        font-weight: bold !important;
    }}

    [data-testid="stSidebar"] [data-testid="stButton"] > button {{
        padding: 10px 8px !important;
    }}

    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] div,
    button[data-baseweb="tab"] span {{
        font-size: 28px !important;
        font-weight: bold !important;
        color: {TEXT_COLOR} !important;
    }}
    
    button[data-baseweb="tab"]:not([aria-selected="true"]) {{
        background-color: transparent !important;
        opacity: 1 !important;
        border: none !important;
    }}

    button[data-baseweb="tab"]:not([aria-selected="true"]) p,
    button[data-baseweb="tab"]:not([aria-selected="true"]) div,
    button[data-baseweb="tab"]:not([aria-selected="true"]) span {{
        font-size: 28px !important;
        color: #666666 !important;
    }}
    
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: transparent !important;
        opacity: 1 !important;
        border: none !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] div,
    button[data-baseweb="tab"][aria-selected="true"] span {{
        font-size: 32px !important;
        font-weight: bold !important;
        color: #C0392B !important;
    }}

    .st-emotion-cache-1c7y2kd {{
        font-size: 28px !important;
        color: {TEXT_COLOR} !important;
    }}

    div[data-testid="stForm"] label p,
    div[data-testid="stForm"] label div,
    div[data-testid="stForm"] label span,
    div[data-testid="stForm"] .stSelectbox div div div p,
    div[data-testid="stForm"] .stSelectbox div div div div span,
    .stSelectbox label p,
    .stSelectbox div[data-testid="stSingleSelectValue"] p,
    .stSelectbox div[role="listbox"] div p,
    .stSelectbox div[role="option"] p,
    .stSelectbox ul li div p,
    div[data-baseweb="select"] div[role="listbox"] div p,
    div[data-baseweb="select"] div[role="listbox"] div div,
    div[data-baseweb="select"] div[role="listbox"] div span,
    div[data-baseweb="select"] div[role="option"] p,
    div[data-baseweb="select"] div[role="option"] div,
    div[data-baseweb="select"] div[role="option"] span,
    .stSelectbox div[data-testid="stOptionInserted"] p,
    .stSelectbox div[data-testid="stOptionInserted"] div,
    .stSelectbox div[data-testid="stOptionInserted"] span,
    .stSlider label p,
    .stSlider label div,
    .stSlider label span,
    .stSlider div[data-testid="stTickV"] div,
    .stSlider div[data-testid="stTooltip"] div,
    .stSlider div[data-testid^="stNumberInput"] input[type="number"],
    .stSlider div[data-testid="stLabelActionable"] div,
    .stSlider div[data-testid="stTickV"] div div,
    .stSlider div[data-testid="stTickV"] div p,
    .stSlider div[data-testid="stTickV"] div span,
    .stSlider div[data-testid="stTooltip"] div div,
    .stSlider div[data-testid="stTooltip"] div p,
    .stSlider div[data-testid="stTooltip"] div span,
    .stSlider div[data-testid="stTickV"] div[data-testid="stTickV"] div,
    .stSlider div[data-testid="stTickV"] div[data-testid="stTickV"] p,
    .stSlider div[data-testid="stTickV"] div[data-testid="stTickV"] span,
    .stSlider div[data-testid="stTickV"] div[data-testid="stTickV"] div div,
    .stSlider div[data-testid="stTickV"] div[data-testid="stTickV"] div p,
    .stSlider div[data-testid="stTickV"] div[data-testid="stTickV"] div span {{
        font-size: 28px !important;
        color: {TEXT_COLOR} !important;
    }}

    .stPlotlyChart, .stVegaLiteChart, .stDeckGlChart {{
        background-color: {CARD_BACKGROUND} !important;
        border: 1px solid #D4C9B8 !important;
        border-radius: 5px !important;
    }}

    .stMetric {{
        background-color: {CARD_BACKGROUND} !important;
        border: 1px solid #D4C9B8 !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }}

    .stMetric [data-testid="stMetricValue"],
    .stMetric [data-testid="stMetricLabel"] {{
        color: #000000 !important;
    }}

    .streamlit-expanderHeader {{
        background-color: {CARD_BACKGROUND} !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid #D4C9B8 !important;
    }}

    [data-testid="stSpinner"] div {{
        color: #808080 !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 28px !important;
        font-weight: 600 !important;
        color: {TEXT_COLOR} !important;
    }}
    
    [data-testid="stMetricLabel"] p {{
        font-size: 28px !important;
        font-weight: bold !important;
        color: {TEXT_COLOR} !important;
    }}
    
    [data-testid="stMetricValue"] {{
        font-size: 36px !important;
        color: {TEXT_COLOR} !important;
    }}

    .stSlider > label > div > p,
    .stSlider > label p,
    .stSelectbox > label > div > p,
    .stSelectbox > label p,
    div[data-testid="stSlider"] > label,
    div[data-testid="stSelectbox"] > label,
    div[data-testid="stSlider"] > label *,
    div[data-testid="stSelectbox"] > label * {{
        font-weight: bold !important;
        font-size: 30px !important;
        color: {TEXT_COLOR} !important;
    }}

    .stSlider label p,
    .stSlider label div,
    .stSlider label span,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {{
        font-size: 30px !important;
        color: {TEXT_COLOR} !important;
        font-weight: bold !important;
    }}
    
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider [data-testid="stThumbValue"],
    .stSlider [data-testid="stMarkdownContainer"] p,
    .stSlider > div > div > div > div > div,
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stSlider"] > div > div > div > div > div {{
        font-size: 26px !important;
        color: {TEXT_COLOR} !important;
    }}
    
    [data-testid="stSlider"] [data-testid="stMarkdownContainer"] * {{
        font-size: 26px !important;
    }}
    
    .stSlider * {{
        font-size: 26px !important;
    }}
    
    [data-baseweb="select"] > div {{
        min-height: 60px !important;
    }}
    
    [data-baseweb="select"] [role="combobox"] {{
        min-height: 58px !important;
        padding: 16px 20px !important;
    }}
    
    [data-baseweb="select"] [role="listbox"] {{
        font-size: 28px !important;
        font-weight: bold !important;
    }}
    
    [data-baseweb="select"] [role="option"] {{
        font-size: 28px !important;
        font-weight: bold !important;
        padding: 16px 20px !important;
        color: white !important;
    }}
    
    [data-baseweb="select"] [role="option"] > div {{
        font-size: 28px !important;
        font-weight: bold !important;
        color: white !important;
    }}
    
    [data-baseweb="select"] [data-testid="stSelectbox"] > div > div > div {{
        font-size: 30px !important;
        font-weight: bold !important;
    }}
    
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"] [role="option"] {{
        font-size: 28px !important;
        font-weight: bold !important;
        color: white !important;
    }}
    
    div[data-baseweb="select"] div[role="combobox"] > div {{
        font-size: 30px !important;
        font-weight: bold !important;
    }}
    
    ul[role="listbox"] li {{
        font-size: 28px !important;
        font-weight: bold !important;
        color: white !important;
    }}
    
    [role="option"] * {{
        font-size: 28px !important;
        font-weight: bold !important;
        color: white !important;
    }}
    
    [data-baseweb="select"] svg {{
        width: 24px !important;
        height: 24px !important;
    }}
    
    [data-baseweb="popover"] > div {{
        font-size: 28px !important;
        font-weight: bold !important;
    }}
    
    .stSlider [aria-label*="slider"] + div,
    .stSlider [role="slider"] + div {{
        font-size: 26px !important;
    }}
    
    [data-testid="stSlider"] div:not([class]):not([data-testid]) {{
        font-size: 26px !important;
    }}
    
    [data-testid="stSlider"] *:not(input):not(svg) {{
        font-size: 26px !important;
        line-height: 1.5 !important;
    }}
    
    [data-baseweb="select"] *:not(svg) {{
        font-size: 28px !important;
        font-weight: bold !important;
    }}
    
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] *,
    [role="listbox"] [role="option"],
    [role="listbox"] [role="option"] * {{
        color: white !important;
        font-weight: bold !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def show_filters():
    """Show global filters in sidebar."""
    st.sidebar.header("Data Filters")
    
    if "selected_years" not in st.session_state:
        st.session_state.selected_years = [2023, 2024]
    if "selected_category_ids" not in st.session_state:
        st.session_state.selected_category_ids = [694, 685]
    if "selected_category_names" not in st.session_state:
        st.session_state.selected_category_names = ["Face Cream", "Sunscreen"]
    if "selected_discounts" not in st.session_state:
        st.session_state.selected_discounts = True
    
    if "selected_customer_types" not in st.session_state:
        st.session_state.selected_customer_types = ["Guest", "Registered"]
    if "selected_customer_status" not in st.session_state:
        st.session_state.selected_customer_status = ["New", "Returning"]
    
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
        st.session_state.selected_discounts = not st.session_state.selected_discounts
    
    def on_guest_change():
        if st.session_state.guest_cb:
            if "Guest" not in st.session_state.selected_customer_types:
                st.session_state.selected_customer_types.append("Guest")
        else:
            if "Guest" in st.session_state.selected_customer_types:
                st.session_state.selected_customer_types.remove("Guest")
    
    def on_registered_change():
        if st.session_state.registered_cb:
            if "Registered" not in st.session_state.selected_customer_types:
                st.session_state.selected_customer_types.append("Registered")
        else:
            if "Registered" in st.session_state.selected_customer_types:
                st.session_state.selected_customer_types.remove("Registered")
    
    def on_new_change():
        if st.session_state.new_cb:
            if "New" not in st.session_state.selected_customer_status:
                st.session_state.selected_customer_status.append("New")
        else:
            if "New" in st.session_state.selected_customer_status:
                st.session_state.selected_customer_status.remove("New")
    
    def on_returning_change():
        if st.session_state.returning_cb:
            if "Returning" not in st.session_state.selected_customer_status:
                st.session_state.selected_customer_status.append("Returning")
        else:
            if "Returning" in st.session_state.selected_customer_status:
                st.session_state.selected_customer_status.remove("Returning")
    
    st.sidebar.subheader("Select Year(s)")
    year_cols = st.sidebar.columns(2)
    
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
    
    st.sidebar.subheader("Select Category(s)")
    category_cols = st.sidebar.columns(2)
    
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
        
    st.sidebar.subheader("Discounts")
    st.sidebar.checkbox(
        "Include discounted orders",
        value=st.session_state.selected_discounts,
        key="discount_toggle",
        on_change=on_discount_toggle_change
    )
    
    st.sidebar.subheader("Type of Customer")
    customer_type_cols = st.sidebar.columns(2)
    
    with customer_type_cols[0]:
        st.checkbox("Guest",
                   value="Guest" in st.session_state.selected_customer_types,
                   key="guest_cb",
                   on_change=on_guest_change)
    
    with customer_type_cols[1]:
        st.checkbox("Registered",
                   value="Registered" in st.session_state.selected_customer_types,
                   key="registered_cb",
                   on_change=on_registered_change)
    
    customer_status_cols = st.sidebar.columns(2)
    
    with customer_status_cols[0]:
        st.checkbox("New",
                   value="New" in st.session_state.selected_customer_status,
                   key="new_cb",
                   on_change=on_new_change)
    
    with customer_status_cols[1]:
        st.checkbox("Returning",
                   value="Returning" in st.session_state.selected_customer_status,
                   key="returning_cb",
                   on_change=on_returning_change)
    
    return (st.session_state.selected_years, 
            st.session_state.selected_category_ids, 
            st.session_state.selected_discounts, 
            st.session_state.selected_customer_types, 
            st.session_state.selected_customer_status)

def main():
    """Main application function."""
    init_session_state()
    
    if "selected_years" not in st.session_state:
        st.session_state.selected_years = [2023, 2024]
    if "selected_category_ids" not in st.session_state:
        st.session_state.selected_category_ids = [694, 685]
    if "selected_category_names" not in st.session_state:
        st.session_state.selected_category_names = ["Face Cream", "Sunscreen"]
    if "selected_discounts" not in st.session_state:
        st.session_state.selected_discounts = True
    if "selected_customer_types" not in st.session_state:
        st.session_state.selected_customer_types = ["Guest", "Registered"]
    if "selected_customer_status" not in st.session_state:
        st.session_state.selected_customer_status = ["New", "Returning"]
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Sales Overview"
    
    if not st.session_state.authenticated:
        display_login_form()
    else:
        st.sidebar.title("Navigation")
        
        if st.sidebar.button("📈 Sales Overview", use_container_width=True):
            st.session_state.current_page = "Sales Overview"
            
        if st.sidebar.button("🛍️ Product Analysis", use_container_width=True):
            st.session_state.current_page = "Product Analysis"
            
        if st.sidebar.button("🏷️ Promotion Analysis", use_container_width=True):
            st.session_state.current_page = "Promotion Analysis"
            
        st.sidebar.markdown("---")
        
        selected_years, selected_category_ids, selected_discounts, selected_customer_types, selected_customer_status = show_filters()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        st.sidebar.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        
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