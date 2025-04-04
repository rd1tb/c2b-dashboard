"""
Monthly sales trend visualization component.
Shows quantity ordered as bars and sales amount as line.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Tuple


def display_monthly_sales_chart(category_sales: pd.DataFrame):
    """
    Display monthly sales chart with quantity ordered bars and sales amount line.
    
    Args:
        category_sales: DataFrame containing the filtered sales data
    """
    if category_sales.empty:
        st.info("No sales data available for the selected criteria.")
        return
    
    # Create year_month column for grouping
    category_sales['year_month'] = category_sales['order_date'].dt.to_period('M')
    
    # Aggregate data by month
    monthly_data = category_sales.groupby(['year_month']).agg({
        'qty_ordered': 'sum',
        'sales_amount': 'sum'
    }).reset_index()
    monthly_data['year_month'] = monthly_data['year_month'].astype(str)
    
    # Sort by month
    monthly_data = monthly_data.sort_values(['year_month'])
    
    # Create a subplot with shared x-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add quantity bars
    fig.add_trace(
        go.Bar(
            x=monthly_data['year_month'],
            y=monthly_data['qty_ordered'],
            name='Units Sold',
            marker_color='rgba(22, 140, 163, 0.8)',  
            marker_line_color='rgba(22, 140, 163, 1)',
            marker_line_width=1
        ),
        secondary_y=False
    )
    
    # Add sales line
    fig.add_trace(
        go.Scatter(
            x=monthly_data['year_month'],
            y=monthly_data['sales_amount'],
            name='Sales Amount',
            line=dict(color='rgba(140, 196, 163, 1)', width=3),
            marker=dict(size=8, color='rgba(140, 196, 163, 1)')
        ),
        secondary_y=True
    )
    
    # Set up layout with dark theme
    fig.update_layout(
        xaxis=dict(
            title='Month',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis1=dict(
            title='Units sold',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis2=dict(
            title='Sales Amount',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        legend=dict(
            font=dict(size=18, color='rgba(255, 255, 255, 0.8)', weight=3),
            bgcolor='rgba(0, 0, 0, 0.3)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            orientation='h',
            yanchor='top',
            y=1.1,
            xanchor='right',
            x=1
        ),
        plot_bgcolor='rgba(11, 14, 31, 1)',
        paper_bgcolor='rgba(11, 14, 31, 1)',
        hovermode='x unified',
        hoverlabel=dict(
        font_size=16),
        height=800
    )
    
    # Add hover template with formatted values
    fig.update_traces(
        hovertemplate='Month: %{x}<br>%{y:,.2f}',
        selector=dict(type='scatter')
    )
    fig.update_traces(
        hovertemplate='Month: %{x}<br>%{y:,.0f}',
        selector=dict(type='bar')
    )
    
    # Format y-axis tick labels
    fig.update_yaxes(
        tickformat=',d',  # No decimal places for quantity
        secondary_y=False
    )
    fig.update_yaxes(
        tickprefix='€',  # Changed from $ to €
        tickformat=',.2f',  # Two decimal places for currency
        secondary_y=True
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)