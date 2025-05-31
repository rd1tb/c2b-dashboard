"""
Sales analysis component with time series decomposition and growth rate analysis.
Includes adjustable granularity for growth rate calculations.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Tuple


def display_sales_analysis(category_sales: pd.DataFrame):
    """
    Display sales analysis with decomposition and growth rate.
    
    Args:
        category_sales: DataFrame containing the filtered sales data
    """
    if category_sales.empty:
        st.info("No sales data available for the selected criteria.")
        return

    display_time_series_decomposition(category_sales)
    display_sales_growth_rate(category_sales)


def display_time_series_decomposition(category_sales: pd.DataFrame):
    """
    Display time series decomposition of sales data.
    
    Args:
        category_sales: DataFrame containing sales data
    """
    if not pd.api.types.is_datetime64_dtype(category_sales['order_date']):
        category_sales['order_date'] = pd.to_datetime(category_sales['order_date'])

    category_sales['year_month'] = category_sales['order_date'].dt.to_period('M')

    monthly_data = category_sales.groupby(['year_month']).agg({
        'sales_amount': 'sum'
    }).reset_index()

    monthly_data['year_month_str'] = monthly_data['year_month'].astype(str)
    monthly_data = monthly_data.sort_values('year_month')

    if len(monthly_data) < 12:
        st.error("Insufficient data for decomposition analysis (less than 12 months available).")
        return

    time_series = pd.Series(
        monthly_data['sales_amount'].values,
        index=pd.to_datetime(monthly_data['year_month'].astype(str))
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        model = st.selectbox(
            "Decomposition model", 
            options=["additive", "multiplicative"],
            format_func=lambda x: {
                "additive": "Additive model",
                "multiplicative": "Multiplicative model"
            }[x],
            index=1
        )

    try:
        period = min(12, len(monthly_data) // 2)
        decomposition = seasonal_decompose(
            time_series, 
            model=model, 
            period=period,
            extrapolate_trend='freq'
        )

        fig = make_subplots(
            rows=4, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
        )

        dates = monthly_data['year_month_str'].tolist()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=decomposition.observed,
                mode='lines+markers',
                name='Observed',
                line=dict(color='rgba(22, 140, 163, 1)', width=6),
                marker=dict(size=7, color='rgba(22, 140, 163, 1)')
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=decomposition.trend,
                mode='lines',
                name='Trend',
                line=dict(color='rgba(255, 153, 51, 1)', width=6)
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=decomposition.seasonal,
                mode='lines+markers',
                name='Seasonal',
                line=dict(color='rgba(0, 80, 0, 1)', width=6),
                marker=dict(size=8, color='rgba(0, 80, 0, 1)')
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=decomposition.resid,
                mode='markers',
                name='Residual',
                marker=dict(
                    size=7,
                    color='rgba(0, 0, 0, 0.6)',
                    line=dict(width=3, color='rgba(0, 0, 0, 0.8)')
                )
            ),
            row=4, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'),
                showlegend=False
            ),
            row=4, col=1
        )

        fig.update_layout(
            height=1200,
            template="plotly_dark",
            plot_bgcolor='#F5F5F5',
            paper_bgcolor='#F5F5F5',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="right",
                x=1,
                font=dict(size=30, color='rgba(0, 0, 0, 1)', weight='bold'),
                bgcolor='rgba(245, 245, 245, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)'
            ),
            hoverlabel=dict(
                font_size=30,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                font=dict(color='rgba(0, 0, 0, 0.9)', weight='normal')
            ),
            hovermode="x unified",
            xaxis4=dict(
                title='Date',
                title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
                tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
                gridcolor='rgba(0, 0, 0, 0.1)'
            )
        )

        fig.update_yaxes(
            gridcolor='rgba(0, 0, 0, 0.1)',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            tickprefix='€',
            tickformat=',.0f',
            row=1
        )

        fig.update_yaxes(
            gridcolor='rgba(0, 0, 0, 0.1)',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            tickprefix='€',
            tickformat=',.0f',
            row=2
        )

        if model == "additive":
            fig.update_yaxes(
                gridcolor='rgba(0, 0, 0, 0.1)',
                title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
                tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
                tickprefix='€',
                tickformat=',.0f',
                row=3
            )
        else:
            fig.update_yaxes(
                gridcolor='rgba(0, 0, 0, 0.1)',
                title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
                tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
                tickformat='.2f',
                row=3
            )

        if model == "additive":
            fig.update_yaxes(
                gridcolor='rgba(0, 0, 0, 0.1)',
                title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
                tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
                tickprefix='€',
                tickformat=',.0f',
                row=4
            )
        else:
            fig.update_yaxes(
                gridcolor='rgba(0, 0, 0, 0.1)',
                title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
                tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
                tickformat='.2f',
                row=4
            )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error performing time series decomposition: {str(e)}")
        st.info("Try selecting a different decomposition model or check if your data has sufficient time points.")


def display_sales_growth_rate(category_sales: pd.DataFrame):
    """
    Display sales growth rate analysis.
    
    Args:
        category_sales: DataFrame containing sales data
    """
    if not pd.api.types.is_datetime64_dtype(category_sales['order_date']):
        category_sales['order_date'] = pd.to_datetime(category_sales['order_date'])

    granularity_options = {
        "monthly": "Monthly",
        "quarterly": "Quarterly"
    }

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_granularity = st.selectbox(
            "Time granularity",
            options=list(granularity_options.keys()),
            format_func=lambda x: granularity_options[x],
            index=0
        )

    if selected_granularity == "monthly":
        category_sales['period'] = category_sales['order_date'].dt.to_period('M')
    else:
        category_sales['period'] = category_sales['order_date'].dt.to_period('Q')

    period_data = category_sales.groupby(['period']).agg({
        'sales_amount': 'sum'
    }).reset_index()

    if 'sales_amount' in period_data.columns:
        period_data['sales_amount'] = period_data['sales_amount'].astype(float)

    period_data['period_str'] = period_data['period'].astype(str)
    period_data = period_data.sort_values('period')

    if len(period_data) > 1:
        period_data['previous_sales'] = period_data['sales_amount'].shift(1)
        period_data['growth_rate'] = (period_data['sales_amount'] - period_data['previous_sales']) / period_data['previous_sales'] * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=period_data['period_str'],
                y=period_data['sales_amount'],
                name='<b>Sales Amount</b>',
                marker_color='rgba(70, 130, 180, 0.7)',
                hovertemplate='Sales: €%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=period_data['period_str'],
                y=period_data['growth_rate'],
                name='<b>Growth Rate</b>',
                mode='lines+markers',
                line=dict(color='#DC143C', width=6),
                marker=dict(size=9, color='#DC143C'),
                hovertemplate='Growth Rate: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=True
        )

        fig.update_layout(
            height=1200,
            template="plotly_dark",
            plot_bgcolor='#F5F5F5',
            paper_bgcolor='#F5F5F5',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="right",
                x=1,
                font=dict(size=30, color='rgba(0, 0, 0, 1)', weight='bold'),
                bgcolor='rgba(245, 245, 245, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)'
            ),
            hoverlabel=dict(
                font_size=32,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                font=dict(color='rgba(0, 0, 0, 0.9)', weight='normal')
            ),
            hovermode="x unified"
        )

        fig.update_yaxes(
            title_text="Sales Amount",
            gridcolor='rgba(0, 0, 0, 0.1)',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            tickprefix='€',
            tickformat=',.0f',
            secondary_y=False
        )

        fig.update_yaxes(
            title_text="Growth Rate (%)",
            gridcolor='rgba(0, 0, 0, 0.1)',
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            ticksuffix='%',
            secondary_y=True
        )

        tick_spacing = max(1, len(period_data) // 10)

        x_axis_title = "Month" if selected_granularity == "monthly" else "Quarter"

        fig.update_xaxes(
            gridcolor='rgba(0, 0, 0, 0.1)',
            title=x_axis_title,
            title_font=dict(size=28, color='rgba(0, 0, 0, 1)', weight='bold'),
            tickfont=dict(size=24, color='rgba(0, 0, 0, 1)', family='Monospace'),
            tickmode='array',
            tickvals=period_data['period_str'][::tick_spacing],
            ticktext=period_data['period_str'][::tick_spacing]
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"Insufficient data for growth rate analysis. At least two {selected_granularity} periods are required.")
