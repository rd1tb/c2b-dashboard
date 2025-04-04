"""
Product rotation analysis components with improved caching.
Contains two main visualization functions:
1. Bubble Chart: Shows avg days between orders on x-axis, avg quantity on y-axis, and coefficient of variation as bubble color.
2. Violin Plots: Shows distribution of days between orders and quantity per order.
Each visualization has its own controls and can be displayed separately.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# Import our optimized functions
from dashboard.components.product_rotation_processor import calculate_product_rotation_metrics

# Set up logging
logger = logging.getLogger(__name__)


@st.cache_data(ttl=7200, show_spinner=False)
def create_bubble_chart(rotation_metrics, variance_type):
    """
    Create product rotation bubble chart - separated to cache the chart creation.
    
    Args:
        rotation_metrics: DataFrame containing the rotation metrics
        variance_type: Type of variance being displayed
        
    Returns:
        Plotly figure object
    """
    if rotation_metrics.empty:
        return None
        
    # Create the bubble chart
    fig = go.Figure()
    
    # Configure color scale - light red to light green
    colorscale = [
        [0, 'rgba(255, 153, 153, 0.8)'],    # Light red
        [0.5, 'rgba(255, 255, 153, 0.8)'],  # Light yellow
        [1, 'rgba(153, 255, 153, 0.8)']     # Light green
    ]
    
    # Add bubbles with square root scaling for better visual differentiation
    # Changed to use base_price instead of revenue for bubble size
    fig.add_trace(go.Scatter(
        x=rotation_metrics['avg_days_between_orders'],
        y=rotation_metrics['avg_quantity_per_order'],
        mode='markers',
        name='',
        marker=dict(
            size=np.sqrt(rotation_metrics['base_price'].astype(float) / rotation_metrics['base_price'].astype(float).max()) * 50 + 10,  # Increased multiplier and base size
            sizemode='diameter',
            color=rotation_metrics['coefficient_of_variation'],
            colorscale=colorscale,
            colorbar=dict(
                title='Coefficient of Variation',
                thickness=15,
                len=0.7,
                tickfont=dict(size=16, color='rgba(255, 255, 255, 0.8)')
            ),
            reversescale=True,  # Reverse to have green for low CV
            opacity=0.8,
            line=dict(width=1, color='rgba(255, 255, 255, 0.7)')  # Thicker and more visible border
        ),
        text=rotation_metrics['sku'],
        hovertemplate=(
            'Product: %{text}<br>' +
            'Avg Days Between Orders: %{x:.1f}<br>' +
            'Avg Quantity Per Order: %{y:.2f}<br>' +
            'Base Price: €%{customdata[1]:,.2f}<br>' +  # Made base price bold
            'Order Count: %{customdata[0]}<br>' +
            'Total Revenue: €%{customdata[2]:,.2f}<br>' +
            'Standard Deviation: %{customdata[3]:.2f}<br>' + 
            'Coefficient of Variation (CV): %{marker.color:.2f}'
        ),
        customdata=np.stack((
            rotation_metrics['order_count'], 
            rotation_metrics['base_price'],  # Added base price to customdata
            rotation_metrics['total_revenue'],
            rotation_metrics['std_deviation']
        ), axis=-1)
    ))
    
    # Set up layout with dark theme
    cv_title = "Order Timing CV (SD/Mean)" if variance_type == "days" else "Quantity CV (SD/Mean)"
    
    fig.update_layout(
        xaxis=dict(
            title='Average Days Between Orders',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            title='Average Quantity Per Order',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        legend=dict(
            font=dict(size=18, color='rgba(255, 255, 255, 0.8)'),
            bgcolor='rgba(0, 0, 0, 0.3)',
            bordercolor='rgba(255, 255, 255, 0.2)')
        ,
        plot_bgcolor='rgba(11, 14, 31, 1)',
        paper_bgcolor='rgba(11, 14, 31, 1)',
        hovermode='closest',
        hoverlabel=dict(font_size=16),
        height=800
    )
    
    # Update colorbar title based on variance type
    fig.update_layout(
        coloraxis=dict(
            colorbar=dict(
                title=cv_title
            )
        )
    )
    
    # Add quadrant annotations
    quadrant_annotations = [
        dict(
            x=0.25, y=0.85, 
            xref="paper", yref="paper",
            text="High Quantity<br>Frequent Orders",
            showarrow=False,
            font=dict(color='rgba(255, 255, 255, 0.7)', size=14)
        ),
        dict(
            x=0.75, y=0.85, 
            xref="paper", yref="paper",
            text="High Quantity<br>Infrequent Orders",
            showarrow=False,
            font=dict(color='rgba(255, 255, 255, 0.7)', size=14)
        ),
        dict(
            x=0.25, y=0.15, 
            xref="paper", yref="paper",
            text="Low Quantity<br>Frequent Orders",
            showarrow=False,
            font=dict(color='rgba(255, 255, 255, 0.7)', size=14)
        ),
        dict(
            x=0.75, y=0.15, 
            xref="paper", yref="paper",
            text="Low Quantity<br>Infrequent Orders",
            showarrow=False,
            font=dict(color='rgba(255, 255, 255, 0.7)', size=14)
        )
    ]
    
    fig.update_layout(annotations=quadrant_annotations)
    
    return fig


@st.cache_data(ttl=7200, show_spinner=False)
def create_distribution_plots(rotation_metrics):
    """
    Create distribution plots - separated to cache the plots creation.
    
    Args:
        rotation_metrics: DataFrame containing the rotation metrics
        
    Returns:
        Tuple of (days_fig, qty_fig) Plotly figure objects
    """
    if rotation_metrics.empty:
        return None, None
        
    # Create box plot for average days between orders
    fig_days = go.Figure()
    fig_qty = go.Figure()
    
    # Only use products with valid days between orders data
    valid_days_data = rotation_metrics['avg_days_between_orders'].dropna()
    
    if len(valid_days_data) >= 5:
        # Create mapping of value to SKU for hovering
        sku_mapping = dict(zip(rotation_metrics['avg_days_between_orders'], rotation_metrics['sku']))
        
        fig_days.add_trace(go.Box(
            y=valid_days_data,
            name="Days Between Orders",
            boxmean=True,  # Show mean as a dashed line
            marker_color='rgba(0, 100, 200, 0.7)',
            line_color='rgba(100, 200, 255, 0.8)',
            fillcolor='rgba(100, 200, 255, 0.3)',
            boxpoints='outliers',  # Only show outlier points for better performance
            text=[f"SKU: {sku_mapping.get(val, 'unknown')}" for val in valid_days_data],
            hovertemplate="Avg Days: %{y:.1f}<br>%{text}<extra></extra>"
        ))
        
        # Update layout
        fig_days.update_layout(
            title=f'Distribution of Avg Days Between Orders',
            yaxis_title='Average Days',
            plot_bgcolor='rgba(11, 14, 31, 1)',
            paper_bgcolor='rgba(11, 14, 31, 1)',
            font=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            height=800,
            hoverlabel=dict(font_size=16),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        
        # Add grid lines
        fig_days.update_yaxes(
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.3)'
        )
    else:
        fig_days = None
    
    # Only use products with valid quantity data
    valid_qty_data = rotation_metrics['avg_quantity_per_order'].dropna()
    
    if len(valid_qty_data) >= 5:
        # Create mapping of value to SKU for hovering
        sku_mapping_qty = dict(zip(rotation_metrics['avg_quantity_per_order'], rotation_metrics['sku']))
        
        fig_qty.add_trace(go.Box(
            y=valid_qty_data,
            name="Quantity Per Order",
            boxmean=True,  # Show mean as a dashed line
            marker_color='rgba(200, 100, 0, 0.7)',
            line_color='rgba(255, 150, 100, 0.8)',
            fillcolor='rgba(255, 150, 100, 0.3)',
            boxpoints='outliers',  # Only show outlier points for better performance
            text=[f"SKU: {sku_mapping_qty.get(val, 'unknown')}" for val in valid_qty_data],
            hovertemplate="Avg Quantity: %{y:.1f}<br>%{text}<extra></extra>"
        ))
        
        # Update layout
        fig_qty.update_layout(
            title=f'Distribution of Avg Quantity Per Order',
            yaxis_title='Average Quantity',
            plot_bgcolor='rgba(11, 14, 31, 1)',
            paper_bgcolor='rgba(11, 14, 31, 1)',
            font=dict(size=18, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            hoverlabel=dict(font_size=16),
            height=800,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        
        # Add grid lines
        fig_qty.update_yaxes(
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.3)'
        )
    else:
        fig_qty = None
        
    return fig_days, fig_qty


def display_product_rotation_bubble_chart(product_sales: pd.DataFrame, max_products: int = 50):
    """
    Display product rotation bubble chart with days between orders on x-axis and
    average quantity on y-axis. Bubble color represents coefficient of variation in order timing or quantity.
    Bubble size represents base price instead of revenue.
    
    Args:
        product_sales: DataFrame containing the filtered sales data
        max_products: Maximum number of products to display (default: 50)
    """
    if product_sales.empty:
        st.info("No sales data available for the selected criteria.")
        return
    
    # Create a container for filters with adjusted column widths
    filter_col1, filter_col2, filter_col3 = st.columns([4, 1, 1])
    
    # Add slider for number of products (takes 2/3 of the width)
    with filter_col1:
        num_products = st.slider(
            "Number of products for bubble chart", 
            min_value=20, 
            max_value=250, 
            value=50,
            step=10,
            key="bubble_num_products"
        )
    
    # Add selector for sorting (takes 1/6 of the width)
    with filter_col2:
        sort_option = st.selectbox(
            "Product selection",
            options=["random", "ascending", "descending"],
            format_func=lambda x: {
                "ascending": "Top revenue",
                "descending": "Lowest revenue",
                "random": "Random"
            }[x],
            key="bubble_sort_option"
        )
    
    # Add selector for variance type (takes 1/6 of the width)
    with filter_col3:
        variance_type = st.selectbox(
            "Variance metric",
            options=["days", "quantity"],
            format_func=lambda x: {
                "days": "Time between orders",
                "quantity": "Order quantity"
            }[x],
            key="bubble_variance_type"
        )
    
    # Calculate rotation metrics using our cached and optimized implementation
    with st.spinner("Calculating product rotation metrics..."):
        # Use the optimized function with caching
        rotation_metrics = calculate_product_rotation_metrics(
            product_sales, 
            top_n=num_products,
            sort_by=sort_option,
            variance_type=variance_type,
            use_cache=True
        )
    
    if rotation_metrics.empty or len(rotation_metrics) < 5:  # Ensure we have enough data for a meaningful chart
        st.info("Not enough product data with multiple orders to generate the rotation chart.")
        return
    
    # Create the chart (this will be cached)
    fig = create_bubble_chart(rotation_metrics, variance_type)
    
    if fig is None:
        st.info("Could not generate chart with the current data.")
        return
        
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Add explanatory text with variance type-specific explanation
    variance_explanation = "order timing pattern" if variance_type == "days" else "order quantity pattern"
    
    st.markdown(f"""
    <div style="background-color: rgba(11, 14, 31, 0.7); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">
            <strong>How to read this chart:</strong><br>
            • <span style="color: rgba(153, 255, 153, 0.8);">Green bubbles</span>: Products with consistent {variance_explanation}<br>
            • <span style="color: rgba(255, 153, 153, 0.8);">Red bubbles</span>: Products with irregular {variance_explanation}<br>
            • <strong>Bubble size represents base price</strong> - larger bubbles = higher priced products<br>
            • CV (Coefficient of Variation) = Standard Deviation / Mean
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_product_distributions(product_sales: pd.DataFrame):
    """
    Display box plots showing the distribution of average metrics across products.
    
    Args:
        product_sales: DataFrame containing the filtered sales data
    """
    if product_sales.empty:
        st.info("No sales data available for the selected criteria.")
        return
    
    # Add separate controls for box plots
    box_col1, box_col2 = st.columns([4, 2])
    
    with box_col1:
        num_products = st.slider(
            "Number of products for distribution analysis", 
            min_value=100, 
            max_value=500, 
            value=100,
            step=10,
            key="dist_num_products"
        )
    
    with box_col2:
        sort_option = st.selectbox(
            "Product selection",
            options=["random", "ascending", "descending"],
            format_func=lambda x: {
                "ascending": "Top revenue",
                "descending": "Lowest revenue",
                "random": "Random"
            }[x],
            key="dist_sort_option"
        )
    
    # Calculate rotation metrics for the selected products using our optimized function
    with st.spinner("Calculating distribution metrics..."):
        # Use the optimized function with caching
        rotation_metrics = calculate_product_rotation_metrics(
            product_sales, 
            top_n=num_products,
            sort_by=sort_option,
            variance_type="days",  # Default to days for distribution analysis
            use_cache=True
        )
    
    if rotation_metrics.empty or len(rotation_metrics) < 5:
        st.info("Not enough product data to generate distribution plots.")
        return
    
    # Create the plots (this will be cached)
    fig_days, fig_qty = create_distribution_plots(rotation_metrics)
    
    # Create two columns for the box plots
    col1, col2 = st.columns(2)
    
    with col1:
        if fig_days is not None:
            st.plotly_chart(fig_days, use_container_width=True)
        else:
            st.info("Not enough products with valid days between orders data.")
    
    with col2:
        if fig_qty is not None:
            st.plotly_chart(fig_qty, use_container_width=True)
        else:
            st.info("Not enough products with valid quantity data.")