import streamlit as st
import pandas as pd

def display_promotion_duration_metrics(promotion_stats: pd.DataFrame):
    """
    Display metrics about promotion durations for the selected years.
    Shows data for each selected year (2023, 2024, or both).
    
    Args:
        promotion_stats: DataFrame containing promotion duration statistics
    """
    if promotion_stats.empty:
        st.info("No promotion statistics available.")
        return
    
    # Get the selected years from session state
    selected_years = st.session_state.selected_years
    
    # Filter data based on selected years
    if len(selected_years) == 1:
        if 2023 in selected_years:
            year_data = promotion_stats[promotion_stats['year'] == '2023']
        elif 2024 in selected_years:
            year_data = promotion_stats[promotion_stats['year'] == '2024']
    else:
        # Both years selected, use combined data
        year_data = promotion_stats[promotion_stats['year'] == '2023-2024']
    
    # Check if filtered data is empty
    if year_data.empty:
        st.info(f"No promotion statistics available for the selected years: {', '.join(map(str, selected_years))}")
        return
    
    # Convert to a single row for metric display
    year_data = year_data.iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"Avg Duration",
            value=f"{year_data['average_duration_days']:.1f} days"
        )
    with col2:
        st.metric(
            label=f"Max Duration",
            value=f"{year_data['max_duration_days']} days"
        )
    with col3:
        st.metric(
            label=f"Total Promotions",
            value=f"{year_data['promotion_count']}")