import streamlit as st
import pandas as pd
from datetime import datetime

def display_promotion_metrics(promotion_stats: pd.DataFrame, promotion_details: pd.DataFrame = None):
    """
    Display metrics about promotion durations for the selected years.
    Shows five metrics in one row:
    - Total promotions
    - Used promotions (count where ctb_units_used > 0)
    - Promotions supported by C2B (count where ctb_discount_amount > 0)
    - Avg duration
    - Max duration
    
    Args:
        promotion_stats: DataFrame containing promotion duration statistics
        promotion_details: DataFrame containing detailed promotion information
    """
    if promotion_stats.empty:
        st.info("No promotion statistics available.")
        return
        
    selected_years = st.session_state.selected_years
    

    if len(selected_years) == 1:
        if 2023 in selected_years:
            year_data = promotion_stats[promotion_stats['year'] == '2023']
        elif 2024 in selected_years:
            year_data = promotion_stats[promotion_stats['year'] == '2024']
    else:
        year_data = promotion_stats[promotion_stats['year'] == '2023-2024']
    
    if year_data.empty:
        st.info(f"No promotion statistics available for the selected years: {', '.join(map(str, selected_years))}")
        return
    
    year_data = year_data.iloc[0]
    
    used_promotions_count = 0
    c2b_supported_promotions_count = 0
    
    if promotion_details is not None and not promotion_details.empty:
        if len(selected_years) == 1:
            year = selected_years[0]
            start_date = pd.to_datetime(f"{year}-01-01")
            end_date = pd.to_datetime(f"{year}-12-31")
            
            if not pd.api.types.is_datetime64_any_dtype(promotion_details['from_date']):
                promotion_details['from_date'] = pd.to_datetime(promotion_details['from_date'])
            
            if not pd.api.types.is_datetime64_any_dtype(promotion_details['to_date']):
                promotion_details['to_date'] = pd.to_datetime(promotion_details['to_date'])
            
            filtered_details = promotion_details[
                (promotion_details['to_date'] >= start_date) &
                (promotion_details['from_date'] <= end_date)
            ]
        else:
            filtered_details = promotion_details
        
        used_promotions_count = len(filtered_details[filtered_details['ctb_units_used'] > 0])
        
        c2b_supported_promotions_count = len(filtered_details[filtered_details['ctb_discount_amount'] > 0])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Promotions",
            value=f"{year_data['promotion_count']}"
        )
    
    with col2:
        st.metric(
            label="Used Promotions",
            value=f"{used_promotions_count}"
        )
    
    with col3:
        st.metric(
            label="C2B Supported",
            value=f"{c2b_supported_promotions_count}"
        )
    
    with col4:
        st.metric(
            label="Avg Duration",
            value=f"{year_data['average_duration_days']:.1f} days"
        )
    
    with col5:
        st.metric(
            label="Max Duration",
            value=f"{year_data['max_duration_days']} days"
        )