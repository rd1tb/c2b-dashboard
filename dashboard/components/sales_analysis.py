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
    
    # Display both analyses one after another instead of using tabs
    display_time_series_decomposition(category_sales)
    display_sales_growth_rate(category_sales)


def display_time_series_decomposition(category_sales: pd.DataFrame):
    """
    Display time series decomposition of sales data.
    
    Args:
        category_sales: DataFrame containing sales data
    """
    st.subheader("Sales Decomposition Analysis")
    
    # Convert dates to datetime if not already
    if not pd.api.types.is_datetime64_dtype(category_sales['order_date']):
        category_sales['order_date'] = pd.to_datetime(category_sales['order_date'])
    
    # Create year_month column for grouping
    category_sales['year_month'] = category_sales['order_date'].dt.to_period('M')
    
    # Aggregate data by month
    monthly_data = category_sales.groupby(['year_month']).agg({
        'sales_amount': 'sum'
    }).reset_index()
    
    # Convert period to string and sort
    monthly_data['year_month_str'] = monthly_data['year_month'].astype(str)
    monthly_data = monthly_data.sort_values('year_month')
    
    # Check if we have enough data for decomposition (at least 12 months)
    if len(monthly_data) < 12:
        st.error("Insufficient data for decomposition analysis (less than 12 months available).")
        return
    
    # Create time series for decomposition
    time_series = pd.Series(
        monthly_data['sales_amount'].values,
        index=pd.to_datetime(monthly_data['year_month'].astype(str))
    )
    
    # Add model selection
    col1, col2 = st.columns([1, 3])
    with col1:
        model = st.selectbox(
            "Decomposition model", 
            options=["additive", "multiplicative"],
            format_func=lambda x: {
                "additive": "Additive model",
                "multiplicative": "Multiplicative model"
            }[x],
            index=0
        )
    
    try:
        # Perform decomposition
        period = min(12, len(monthly_data) // 2)  # Use appropriate period based on data length
        decomposition = seasonal_decompose(
            time_series, 
            model=model, 
            period=period,
            extrapolate_trend='freq'
        )
        
        # Create a subplot with 4 rows
        fig = make_subplots(
            rows=4, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
        )
        
        # Get date strings for x-axis
        dates = monthly_data['year_month_str'].tolist()
        
        # Add observed data
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=decomposition.observed,
                mode='lines+markers',
                name='Observed',
                line=dict(color='rgba(22, 140, 163, 0.8)', width=2),
                marker=dict(size=6, color='rgba(22, 140, 163, 1)')
            ),
            row=1, col=1
        )
        
        # Add trend data
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=decomposition.trend,
                mode='lines',
                name='Trend',
                line=dict(color='rgba(255, 153, 51, 0.8)', width=3)
            ),
            row=2, col=1
        )
        
        # Add seasonal data
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=decomposition.seasonal,
                mode='lines+markers',
                name='Seasonal',
                line=dict(color='rgba(140, 196, 163, 0.8)', width=2),
                marker=dict(size=5, color='rgba(140, 196, 163, 1)')
            ),
            row=3, col=1
        )
        
        # Add residual data
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=decomposition.resid,
                mode='markers',
                name='Residual',
                marker=dict(
                    size=6, 
                    color='rgba(255, 255, 255, 0.5)',
                    line=dict(width=1, color='rgba(255, 255, 255, 0.8)')
                )
            ),
            row=4, col=1
        )
        
        # Add zero line for residuals
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
        
        # Update layout
        fig.update_layout(
            height=1000,
            template="plotly_dark",
            plot_bgcolor='rgba(11, 14, 31, 1)',
            paper_bgcolor='rgba(11, 14, 31, 1)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=18, color='rgba(255, 255, 255, 0.8)'),
                bgcolor='rgba(0, 0, 0, 0.3)',
                bordercolor='rgba(255, 255, 255, 0.2)'
            ),
            hoverlabel=dict(font_size=16),
            hovermode="x unified"
        )
        
        # Update y-axes
        fig.update_yaxes(
            gridcolor='rgba(255, 255, 255, 0.1)',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            tickprefix='€',
            tickformat=',.0f',
            row=1
        )
        
        fig.update_yaxes(
            gridcolor='rgba(255, 255, 255, 0.1)',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            tickprefix='€',
            tickformat=',.0f',
            row=2
        )
        
        # For seasonal, don't use currency format if multiplicative
        if model == "additive":
            fig.update_yaxes(
                gridcolor='rgba(255, 255, 255, 0.1)',
                title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
                tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
                tickprefix='€',
                tickformat=',.0f',
                row=3
            )
        else:
            fig.update_yaxes(
                gridcolor='rgba(255, 255, 255, 0.1)',
                title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
                tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
                tickformat='.2f',
                row=3
            )
        
        if model == "additive":
            fig.update_yaxes(
                gridcolor='rgba(255, 255, 255, 0.1)',
                title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
                tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
                tickprefix='€',
                tickformat=',.0f',
                row=4
            )
        else:
            fig.update_yaxes(
                gridcolor='rgba(255, 255, 255, 0.1)',
                title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
                tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
                tickformat='.2f',
                row=4
            )
        
        # Update x-axis
        fig.update_xaxes(
            gridcolor='rgba(255, 255, 255, 0.1)',
                title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
                tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
        )
        
        # Show every nth tick to avoid overcrowding
        tick_spacing = max(1, len(dates) // 10)
        fig.update_xaxes(
            tickmode='array',
            tickvals=dates[::tick_spacing],
            ticktext=dates[::tick_spacing]
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div style="background-color: rgba(11, 14, 31, 0.7); padding: 10px; border-radius: 5px; margin-top: 10px;">
        <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">
        <strong>How to interpret this decomposition:</strong><br>
        • <strong>Observed</strong>: The original sales data<br>
        • <strong>Trend</strong>: The long-term progression of the sales (increasing or decreasing)<br>
        • <strong>Seasonal</strong>: Repeating patterns over fixed periods (yearly cycles)<br>
        • <strong>Residual</strong>: Irregular fluctuations that cannot be attributed to trend or seasonality
        </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error performing time series decomposition: {str(e)}")
        st.info("Try selecting a different decomposition model or check if your data has sufficient time points.")


def display_sales_growth_rate(category_sales: pd.DataFrame):
    """
    Display sales growth rate analysis.
    
    Args:
        category_sales: DataFrame containing sales data
    """
    st.subheader("Sales Growth Rate Analysis")
    
    # Convert dates to datetime if not already
    if not pd.api.types.is_datetime64_dtype(category_sales['order_date']):
        category_sales['order_date'] = pd.to_datetime(category_sales['order_date'])
    
    # Add granularity selection
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
    
    # Create period column based on selected granularity
    if selected_granularity == "monthly":
        category_sales['period'] = category_sales['order_date'].dt.to_period('M')
    else:  # quarterly
        category_sales['period'] = category_sales['order_date'].dt.to_period('Q')
    
    # Aggregate data by period
    period_data = category_sales.groupby(['period']).agg({
        'sales_amount': 'sum'
    }).reset_index()
    
    # Convert decimal values to float
    if 'sales_amount' in period_data.columns:
        period_data['sales_amount'] = period_data['sales_amount'].astype(float)
    
    # Convert period to string and sort
    period_data['period_str'] = period_data['period'].astype(str)
    period_data = period_data.sort_values('period')
    
    # Calculate growth rates
    if len(period_data) > 1:
        period_data['previous_sales'] = period_data['sales_amount'].shift(1)
        period_data['growth_rate'] = (period_data['sales_amount'] - period_data['previous_sales']) / period_data['previous_sales'] * 100
        
        # Create figure with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add sales amount bars
        fig.add_trace(
            go.Bar(
                x=period_data['period_str'],
                y=period_data['sales_amount'],
                name='Sales Amount',
                marker_color='rgba(22, 140, 163, 0.7)',
                hovertemplate='%{x}<br>Sales: €%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add growth rate line
        fig.add_trace(
            go.Scatter(
                x=period_data['period_str'],
                y=period_data['growth_rate'],
                name='Growth Rate',
                mode='lines+markers',
                line=dict(color='rgba(255, 153, 51, 0.8)', width=3),
                marker=dict(size=8),
                hovertemplate='%{x}<br>Growth Rate: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Update layout
        title_text = {
            "monthly": "Monthly Sales Growth Rate",
            "quarterly": "Quarterly Sales Growth Rate"
        }
        
        fig.update_layout(
            height=800,
            template="plotly_dark",
            plot_bgcolor='rgba(11, 14, 31, 1)',
            paper_bgcolor='rgba(11, 14, 31, 1)',
            title=dict(
                text=title_text[selected_granularity],
                font=dict(size=20, color='rgba(255, 255, 255, 0.9)')
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=18, color='rgba(255, 255, 255, 0.8)'),
                bgcolor='rgba(0, 0, 0, 0.3)',
                bordercolor='rgba(255, 255, 255, 0.2)'
            ),
            hoverlabel=dict(font_size=16),
            hovermode="x unified"
        )
        
        # Update y-axes
        fig.update_yaxes(
            title_text="Sales Amount",
            gridcolor='rgba(255, 255, 255, 0.1)',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            tickprefix='€',
            tickformat=',.0f',
            range=[0, period_data['sales_amount'].max() * 1.2],  # Add 20% headroom
            secondary_y=False
        )
        
        fig.update_yaxes(
            title_text="Growth Rate (%)",
            gridcolor='rgba(255, 255, 255, 0.1)',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
            ticksuffix='%',
            range=[
                min(period_data['growth_rate'].min() * 1.2, -7) if not period_data['growth_rate'].isna().all() else -7,
                max(period_data['growth_rate'].max() * 1.2, 7) if not period_data['growth_rate'].isna().all() else 7
            ],
            secondary_y=True
        )
        
        # Update x-axis
        fig.update_xaxes(
            gridcolor='rgba(255, 255, 255, 0.1)',
            title_font=dict(size=18, color='rgba(255, 255, 255, 0.9)'),
            tickfont=dict(size=16, color='rgba(255, 255, 255, 0.9)', family='Monospace'),
        )
        
        # Show every nth tick to avoid overcrowding
        tick_spacing = max(1, len(period_data) // 10)
        fig.update_xaxes(
            tickmode='array',
            tickvals=period_data['period_str'][::tick_spacing],
            ticktext=period_data['period_str'][::tick_spacing]
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display average growth rate
        avg_growth = period_data['growth_rate'].mean()
        avg_growth_text = f"Average {selected_granularity[:-2] if selected_granularity.endswith('ly') else selected_granularity} growth rate: {avg_growth:.2f}%"
        
        # Determine color based on average growth
        if avg_growth > 0:
            avg_growth_color = "rgba(75, 192, 192, 0.9)"  # Green
        elif avg_growth < 0:
            avg_growth_color = "rgba(255, 99, 132, 0.9)"  # Red
        else:
            avg_growth_color = "rgba(255, 255, 255, 0.9)"  # White
            
        st.markdown(f"""
        <div style="background-color: rgba(11, 14, 31, 0.7); padding: 10px; border-radius: 5px; margin-top: 10px;">
        <p style="color: {avg_growth_color}; font-size: 18px; font-weight: bold; text-align: center; margin: 0;">
        {avg_growth_text}
        </p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.warning(f"Insufficient data for growth rate analysis. At least two {selected_granularity} periods are required.")