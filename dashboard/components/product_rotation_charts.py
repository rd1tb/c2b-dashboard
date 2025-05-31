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

from dashboard.components.product_rotation_processor import calculate_product_rotation_metrics

logger = logging.getLogger(__name__)


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

    fig = go.Figure()

    colorscale = [
        [0, 'rgba(139, 0, 0, 0.85)'],
        [0.5, 'rgba(184, 134, 11, 0.85)'],
        [1, 'rgba(0, 100, 0, 0.85)']
    ]

    fig.add_trace(go.Scatter(
        x=rotation_metrics['avg_days_between_orders'],
        y=rotation_metrics['avg_quantity_per_order'],
        mode='markers',
        name='',
        marker=dict(
            size=np.sqrt(rotation_metrics['base_price'].astype(float) / rotation_metrics['base_price'].astype(float).max()) * 80 + 20,
            sizemode='diameter',
            color=rotation_metrics['coefficient_of_variation'],
            colorscale=colorscale,
            colorbar=dict(
                title=dict(
                    text='CV',
                    font=dict(size=30, color='rgba(0, 0, 0, 1)', weight='bold')
                ),
                thickness=20,
                len=0.7,
                tickfont=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold')
            ),
            reversescale=True,
            opacity=0.85,
            line=dict(width=1, color='rgba(0, 0, 0, 0.7)')
        ),
        text=rotation_metrics['sku'],
        hovertemplate=(
            '<b>Product: %{text}</b><br>' +
            'Avg Days Between Orders: %{x:.1f}<br>' +
            'Avg Quantity Per Order: %{y:.2f}<br>' +
            'Base Price: €%{customdata[1]:,.2f}<br>' +
            'Order Count: %{customdata[0]}<br>' +
            'Total Revenue: €%{customdata[2]:,.2f}<br>' +
            'Standard Deviation: %{customdata[3]:.2f}<br>' + 
            'Coefficient of Variation (CV): %{marker.color:.2f}'
        ),
        customdata=np.stack((
            rotation_metrics['order_count'], 
            rotation_metrics['base_price'],
            rotation_metrics['total_revenue'],
            rotation_metrics['std_deviation']
        ), axis=-1)
    ))

    cv_title = "Order Timing CV (SD/Mean)" if variance_type == "days" else "Quantity CV (SD/Mean)"

    fig.update_layout(
        xaxis=dict(
            title='Average Days Between Orders',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            gridcolor='rgba(0, 0, 0, 0.1)'
        ),
        yaxis=dict(
            title='Average Quantity Per Order',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            gridcolor='rgba(0, 0, 0, 0.1)'
        ),
        legend=dict(
            font=dict(size=30, color='rgba(0, 0, 0, 1)', weight='bold'),
            bgcolor='rgba(245, 245, 245, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)'
        ),
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',
        hovermode='closest',
        hoverlabel=dict(
            font_size=28,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            font=dict(color='rgba(0, 0, 0, 0.9)')
        ),
        height=1000
    )

    fig.update_layout(
        coloraxis=dict(
            colorbar=dict(
                title=cv_title
            )
        )
    )

    quadrant_annotations = [
        dict(
            x=0.25, y=0.75, 
            xref="paper", yref="paper",
            text="High Quantity<br>Frequent Orders",
            showarrow=False,
            font=dict(color='rgba(0, 0, 0, 0.3)', size=32, weight='bold'),
            xanchor='center',
            yanchor='middle'
        ),
        dict(
            x=0.75, y=0.75, 
            xref="paper", yref="paper",
            text="High Quantity<br>Infrequent Orders",
            showarrow=False,
            font=dict(color='rgba(0, 0, 0, 0.3)', size=32, weight='bold'),
            xanchor='center',
            yanchor='middle'
        ),
        dict(
            x=0.25, y=0.25, 
            xref="paper", yref="paper",
            text="Low Quantity<br>Frequent Orders",
            showarrow=False,
            font=dict(color='rgba(0, 0, 0, 0.3)', size=32, weight='bold'),
            xanchor='center',
            yanchor='middle'
        ),
        dict(
            x=0.75, y=0.25, 
            xref="paper", yref="paper",
            text="Low Quantity<br>Infrequent Orders",
            showarrow=False,
            font=dict(color='rgba(0, 0, 0, 0.3)', size=32, weight='bold'),
            xanchor='center',
            yanchor='middle'
        )
    ]

    fig.update_layout(annotations=quadrant_annotations)

    return fig

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

    fig_days = go.Figure()
    fig_qty = go.Figure()

    valid_days_data = rotation_metrics['avg_days_between_orders'].dropna()

    if len(valid_days_data) >= 5:
        sku_mapping = dict(zip(rotation_metrics['avg_days_between_orders'], rotation_metrics['sku']))

        fig_days.add_trace(go.Violin(
            y=valid_days_data,
            name="Days Between Orders",
            box_visible=True,
            meanline_visible=True,
            points='outliers',
            text=[f"SKU: {sku_mapping.get(val, 'unknown')}" for val in valid_days_data],
            hoverinfo='y+text',
            marker=dict(
                size=14,
                color='rgba(0, 20, 60, 0.95)',
                line=dict(width=1, color='rgba(0, 0, 0, 0.95)')
            ),
            line=dict(width=5, color='rgba(0, 30, 80, 0.95)'),
            fillcolor='rgba(0, 30, 80, 0.4)',
            opacity=0.95
        ))

        fig_days.update_layout(
            title=dict(
                text='Distribution of Avg Days Between Orders',
                font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold')
            ),
            xaxis=dict(
                title='',
                tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace')
            ),
            yaxis=dict(
                title='Average Days',
                title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
                tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
                showgrid=True,
                gridcolor='rgba(0, 0, 0, 0.3)',
                showticklabels=True,
                nticks=15
            ),
            plot_bgcolor='#F5F5F5',
            paper_bgcolor='#F5F5F5',
            font=dict(size=25, color='rgba(0, 0, 0, 1)', family='Monospace'),
            height=1000,
            hoverlabel=dict(
                font_size=30,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                font=dict(color='rgba(0, 0, 0, 0.9)', weight='normal')
            ),
            margin=dict(l=80, r=30, t=100, b=60)
        )

        fig_days.update_yaxes(
            gridcolor='rgba(0, 0, 0, 0.1)',
            zerolinecolor='rgba(0, 0, 0, 0.3)'
        )
    else:
        fig_days = None

    valid_qty_data = rotation_metrics['avg_quantity_per_order'].dropna()

    if len(valid_qty_data) >= 5:
        sku_mapping_qty = dict(zip(rotation_metrics['avg_quantity_per_order'], rotation_metrics['sku']))

        fig_qty.add_trace(go.Violin(
            y=valid_qty_data,
            name="Quantity Per Order",
            box_visible=True,
            meanline_visible=True,
            points='outliers',
            text=[f"SKU: {sku_mapping_qty.get(val, 'unknown')}" for val in valid_qty_data],
            hoverinfo='y+text',
            marker=dict(
                size=14,
                color='rgba(160, 80, 0, 0.95)',
                line=dict(width=1, color='rgba(0, 0, 0, 0.95)')
            ),
            line=dict(width=5, color='rgba(180, 90, 0, 0.95)'),
            fillcolor='rgba(180, 90, 0, 0.4)',
            opacity=0.95
        ))

        fig_qty.update_layout(
            title=dict(
                text='Distribution of Avg Quantity Per Order',
                font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold')
            ),
            xaxis=dict(
                title='',
                tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace')
            ),
            yaxis=dict(
                title='Average Quantity',
                title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
                tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
                showgrid=True,
                gridcolor='rgba(0, 0, 0, 0.3)',
                showticklabels=True,
                nticks=10
            ),
            plot_bgcolor='#F5F5F5',
            paper_bgcolor='#F5F5F5',
            font=dict(size=25, color='rgba(0, 0, 0, 1)', family='Monospace'),
            hoverlabel=dict(
                font_size=30,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                font=dict(color='rgba(0, 0, 0, 0.9)', weight='normal')
            ),
            height=1000,
            margin=dict(l=80, r=30, t=100, b=60)
        )

        fig_qty.update_yaxes(
            gridcolor='rgba(0, 0, 0, 0.1)',
            zerolinecolor='rgba(0, 0, 0, 0.3)'
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

    filter_col1, filter_col2, filter_col3 = st.columns([4, 1, 1])

    with filter_col1:
        num_products = st.slider(
            "Number of products for bubble chart", 
            min_value=50, 
            max_value=250, 
            value=150,
            step=10,
            key="bubble_num_products"
        )

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

    with st.spinner("Calculating product rotation metrics..."):
        rotation_metrics = calculate_product_rotation_metrics(
            product_sales, 
            top_n=num_products,
            sort_by=sort_option,
            variance_type=variance_type,
            use_cache=True
        )

    if rotation_metrics.empty or len(rotation_metrics) < 5:
        st.info("Not enough product data with multiple orders to generate the rotation chart.")
        return

    fig = create_bubble_chart(rotation_metrics, variance_type)

    if fig is None:
        st.info("Could not generate chart with the current data.")
        return
        
    st.plotly_chart(fig, use_container_width=True)

def display_product_distributions(product_sales: pd.DataFrame):
    """
    Display box plots showing the distribution of average metrics across products.
    
    Args:
        product_sales: DataFrame containing the filtered sales data
    """
    if product_sales.empty:
        st.info("No sales data available for the selected criteria.")
        return

    box_col1, box_col2 = st.columns([4, 2])

    with box_col1:
        num_products = st.slider(
            "Number of products for distribution analysis", 
            min_value=100, 
            max_value=500, 
            value=300,
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

    with st.spinner("Calculating distribution metrics..."):
        rotation_metrics = calculate_product_rotation_metrics(
            product_sales, 
            top_n=num_products,
            sort_by=sort_option,
            variance_type="days",
            use_cache=True
        )

    if rotation_metrics.empty or len(rotation_metrics) < 5:
        st.info("Not enough product data to generate distribution plots.")
        return

    fig_days, fig_qty = create_distribution_plots(rotation_metrics)

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
