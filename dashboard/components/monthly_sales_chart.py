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

    category_sales['year_month'] = category_sales['order_date'].dt.to_period('M')
    monthly_data = category_sales.groupby(['year_month']).agg({
        'qty_ordered': 'sum',
        'sales_amount': 'sum'
    }).reset_index()
    monthly_data['year_month'] = monthly_data['year_month'].astype(str)
    monthly_data = monthly_data.sort_values(['year_month'])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
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
    fig.add_trace(
        go.Scatter(
            x=monthly_data['year_month'],
            y=monthly_data['sales_amount'],
            name='Sales Amount',
            line=dict(color='#D35400', width=6),
            marker=dict(size=9, color='#D35400')
        ),
        secondary_y=True
    )
    fig.update_layout(
        xaxis=dict(
            title='Month',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            gridcolor='rgba(0, 0, 0, 0.1)'
        ),
        yaxis1=dict(
            title='Units sold',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            gridcolor='rgba(0, 0, 0, 0.1)'
        ),
        yaxis2=dict(
            title='Sales Amount',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            gridcolor='rgba(0, 0, 0, 0.1)'
        ),
        legend=dict(
            font=dict(size=30, color='rgba(0, 0, 0, 1)', weight='bold'),
            bgcolor='rgba(245, 245, 245, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            orientation='h',
            yanchor='top',
            y=1.1,
            xanchor='right',
            x=1
        ),
        plot_bgcolor='#F5F5F5',
        paper_bgcolor='#F5F5F5',
        hovermode='x unified',
        hoverlabel=dict(
            font_size=32,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            font=dict(color='rgba(0, 0, 0, 0.9)', weight='normal')
        ),
        height=1200
    )
    fig.update_traces(
        hovertemplate='%{y:,.2f}',
        selector=dict(type='scatter')
    )
    fig.update_traces(
        hovertemplate='%{y:,.0f}',
        selector=dict(type='bar')
    )
    fig.update_yaxes(
        tickformat=',d',
        secondary_y=False
    )
    fig.update_yaxes(
        tickprefix='€',
        tickformat=',.2f',
        secondary_y=True
    )
    st.plotly_chart(fig, use_container_width=True)
