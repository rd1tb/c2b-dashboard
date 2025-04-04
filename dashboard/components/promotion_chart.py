"""
Promotion and campaign analysis component.
Analyzes promotion usage patterns with interactive visualization.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def process_promotion_data(promotion_data, max_promotions, sort_option):
    """
    Process and filter promotion data based on sorting option.
    
    Args:
        promotion_data: DataFrame with promotion details
        max_promotions: Number of promotions to include
        sort_option: How to sort the promotions
        
    Returns:
        DataFrame with processed promotion data
    """
    # Handle any missing or null values
    for col in ['discount_amount', 'ctb_discount_used', 'ctb_units_used']:
        if col in promotion_data.columns:
            promotion_data[col] = pd.to_numeric(promotion_data[col], errors='coerce').fillna(0)
    
    # Calculate effective dates and duration
    promotion_data['from_date'] = pd.to_datetime(promotion_data['from_date'])
    promotion_data['to_date'] = pd.to_datetime(promotion_data['to_date'])
    promotion_data['duration_days'] = (promotion_data['to_date'] - promotion_data['from_date']).dt.days + 1
    
    # Select promotions based on sorting option
    if sort_option == "most_used":
        selected_promotions = promotion_data.sort_values(by='ctb_discount_used', ascending=False).head(max_promotions)
    elif sort_option == "least_used":
        # Filter out unused promotions first (they would dominate the "least used" category)
        used_promotions = promotion_data[promotion_data['ctb_discount_used'] > 0]
        if used_promotions.empty:
            selected_promotions = promotion_data.head(max_promotions)  # Fallback if no used promotions
        else:
            selected_promotions = used_promotions.sort_values(by='ctb_discount_used', ascending=True).head(max_promotions)
    elif sort_option == "random":
        # Random selection
        if len(promotion_data) <= max_promotions:
            selected_promotions = promotion_data
        else:
            selected_promotions = promotion_data.sample(n=max_promotions)
    else:  # fallback
        selected_promotions = promotion_data.head(max_promotions)
        
    return selected_promotions


@st.cache_data(ttl=7200, show_spinner=False)
def create_promotion_chart(selected_promotions):
    """
    Create a bubble chart visualization from processed promotion data.
    
    Args:
        selected_promotions: DataFrame with processed promotion data
        
    Returns:
        Plotly figure object
    """
    # Create the bubble chart
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=selected_promotions['discount_amount'],
        y=selected_promotions['ctb_discount_used'],
        mode='markers',
        marker=dict(
            size=np.sqrt(selected_promotions['ctb_units_used']) + 5,  # Square root scaling for better visual
            sizemode='area',
            sizeref=2.*max(np.sqrt(selected_promotions['ctb_units_used'])+5)/(40.**2),
            color=selected_promotions['duration_days'],
            colorscale='Viridis',
            colorbar=dict(
                title='Duration (days)',
                thickness=15,
                len=0.7,
                tickfont=dict(size=12)
            ),
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=selected_promotions['name'],
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Discount Amount: %{x:.2f}<br>' +
            'Times Used: %{y}<br>' +
            'Units Sold: %{marker.size:.0f}<br>' +
            'Duration: %{marker.color:.0f} days<br>' +
            '<extra></extra>'
        )
    ))
    
    # Set up layout
    fig.update_layout(
        title="Promotion Usage Analysis",
        xaxis=dict(
            title='Discount Amount',
            gridcolor='rgba(230, 230, 230, 0.3)',
            # Use tickfont for text size
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='Number of Times Used',
            gridcolor='rgba(230, 230, 230, 0.3)',
            # Use tickfont for text size
            tickfont=dict(size=12)
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.05)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='rgba(70, 70, 70, 1)', size=12),
        height=800,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode='closest'
    )
    
    return fig


def create_promotion_usage_chart(promotion_details: pd.DataFrame):
    """
    Create a bubble chart visualization for promotion usage metrics with UI filters.
    
    Args:
        promotion_details: DataFrame with promotion details
        
    Returns:
        Plotly figure object
    """
    if promotion_details.empty:
        return None
    
    # Create filters container
    filter_col1, filter_col2 = st.columns([2, 1])
    
    with filter_col1:
        max_promotions = st.slider(
            "Number of promotions to analyze", 
            min_value=50, 
            max_value=500, 
            value=100,
            step=50
        )
    
    with filter_col2:
        sort_option = st.selectbox(
            "Promotion selection",
            options=["most_used", "least_used", "random"],
            format_func=lambda x: {
                "most_used": "Most used",
                "least_used": "Least used",
                "random": "Random"
            }[x],
            key="promotion_sort_option"
        )
    
    # Process and select promotions
    with st.spinner("Processing promotion data..."):
        # Create a copy to avoid modifying the original DataFrame
        promotion_data = promotion_details.copy()
        
        # For non-random selections, we can use session-state caching
        if sort_option != "random":
            # Generate a cache key based on the promotion details and parameters
            cache_key = f"promo_{hash(tuple(promotion_details['name'].tolist()))}_n{max_promotions}_sort{sort_option}"
            
            # Check if result is in session_state cache
            if cache_key not in st.session_state:
                # Process and store in session state
                st.session_state[cache_key] = process_promotion_data(
                    promotion_data, 
                    max_promotions, 
                    sort_option
                )
            
            selected_promotions = st.session_state[cache_key]
        else:
            # For random selection, don't cache
            selected_promotions = process_promotion_data(
                promotion_data, 
                max_promotions, 
                sort_option
            )
    
    # Create the visualization with selected promotions
    with st.spinner("Generating promotion usage chart..."):
        # Create the chart (using decorator-based caching)
        fig = create_promotion_chart(selected_promotions)
        
        # Display the chart
        if fig is None:
            st.info("Could not generate chart with the current data.")
            return None
            
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanatory text
        st.markdown("""
        **How to read this chart:**
        - **X-axis**: Discount amount applied in the promotion
        - **Y-axis**: Number of times the discount was used
        - **Bubble size**: Number of units sold with this promotion
        - **Color**: Duration of the promotion in days
        
        Larger bubbles with brighter colors indicate promotions that sold more units and ran for longer periods.
        """)
    
    return fig