import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pypfopt import EfficientFrontier, risk_models, expected_returns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_data(etf_list, start, end):
    """Download ETF data from Yahoo Finance with error handling."""
    try:
        if isinstance(etf_list, str):
            etf_list = [etf_list]
        data = yf.download(etf_list, start=start, end=end)['Adj Close']
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data, columns=[etf_list[0]])
        return data
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        st.error(f"Failed to fetch data: {str(e)}")
        return None

def plot_price_movement(data, etf_list):
    """Plot price movement using Plotly for interactive charts."""
    fig = go.Figure()
    for etf in etf_list:
        if etf in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[etf],
                name=etf,
                mode='lines'
            ))
    fig.update_layout(
        title='Price Movement of ETFs',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        showlegend=True
    )
    st.plotly_chart(fig)

def plot_normalized_price(data, etf_list):
    """Plot normalized price movement using Plotly."""
    normalized_data = data.div(data.iloc[0])
    fig = go.Figure()
    for etf in etf_list:
        if etf in normalized_data.columns:
            fig.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[etf],
                name=etf,
                mode='lines'
            ))
    fig.update_layout(
        title='Normalized Price Movement',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        showlegend=True
    )
    st.plotly_chart(fig)

def plot_moving_average(data, etf_list):
    """Plot moving averages using Plotly with customizable window."""
    ma_window = st.slider('Moving Average Window (Days)', 20, 200, 50)
    fig = go.Figure()
    for etf in etf_list:
        if etf in data.columns:
            ma = data[etf].rolling(window=ma_window).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma,
                name=f'{etf} {ma_window}-Day MA',
                mode='lines'
            ))
    fig.update_layout(
        title=f'{ma_window}-Day Moving Average Trend of ETFs',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        showlegend=True
    )
    st.plotly_chart(fig)

def calculate_risk_metrics(data, etf_list):
    """Calculate and display risk metrics with error handling."""
    try:
        daily_returns = data.pct_change().dropna()
        metrics = pd.DataFrame(index=etf_list)
        
        # Calculate metrics
        metrics['Annualized Return (%)'] = (daily_returns.mean() * 252 * 100).round(2)
        metrics['Annualized Volatility (%)'] = (daily_returns.std() * np.sqrt(252) * 100).round(2)
        
        risk_free_rate = st.sidebar.number_input('Risk-free rate (%)', value=2.0) / 100
        
        metrics['Sharpe Ratio'] = ((metrics['Annualized Return (%)'] / 100 - risk_free_rate) / 
                                 (metrics['Annualized Volatility (%)'] / 100)).round(2)
        
        # Calculate Sortino Ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        metrics['Sortino Ratio'] = ((metrics['Annualized Return (%)'] / 100 - risk_free_rate) / 
                                  downside_deviation).round(2)
        
        st.write("### Risk Metrics")
        st.dataframe(metrics)
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        st.error("Failed to calculate risk metrics. Please check your data.")

def plot_efficient_frontier(data):
    """Plot efficient frontier with improved error handling and visualization."""
    try:
        # Calculate returns and covariance matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # Create efficient frontier object
        ef = EfficientFrontier(mu, S)
        
        # Calculate maximum Sharpe ratio portfolio
        weights_sharpe = ef.max_sharpe()
        ret_sharpe, vol_sharpe, _ = ef.portfolio_performance()
        
        # Generate efficient frontier points
        ef = EfficientFrontier(mu, S)
        returns = []
        volatilities = []
        for target_return in np.linspace(0.0, max(mu), 50):
            try:
                ef.efficient_return(target_return)
                ret, vol, _ = ef.portfolio_performance()
                returns.append(ret)
                volatilities.append(vol)
            except Exception:
                continue
        
        # Create interactive Plotly plot
        fig = go.Figure()
        
        # Plot efficient frontier
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='lines',
            name='Efficient Frontier'
        ))
        
        # Plot maximum Sharpe ratio point
        fig.add_trace(go.Scatter(
            x=[vol_sharpe],
            y=[ret_sharpe],
            mode='markers',
            marker=dict(size=15, symbol='star'),
            name='Maximum Sharpe Ratio Portfolio'
        ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Annualized Volatility',
            yaxis_title='Annualized Return',
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # Display optimal portfolio weights
        st.write("### Optimal Portfolio Weights (Maximum Sharpe Ratio)")
        weights_df = pd.DataFrame.from_dict(weights_sharpe, orient='index', columns=['Weight'])
        weights_df['Weight'] = (weights_df['Weight'] * 100).round(2)
        st.dataframe(weights_df)
        
    except Exception as e:
        logger.error(f"Error generating efficient frontier: {str(e)}")
        st.error("Failed to generate efficient frontier. Please check your data and ensure you have sufficient historical prices.")

def calculate_percentage_gains(data, etf_list):
    """Calculate and plot percentage gains/losses over time."""
    try:
        # Calculate percentage change from the start
        pct_changes = ((data - data.iloc[0]) / data.iloc[0] * 100)
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add traces for each ETF
        for etf in etf_list:
            if etf in pct_changes.columns:
                fig.add_trace(go.Scatter(
                    x=pct_changes.index,
                    y=pct_changes[etf],
                    name=etf,
                    mode='lines',
                    hovertemplate='%{y:.1f}%<extra></extra>'
                ))
        
        # Add a horizontal line at 0%
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_layout(
            title='Percentage Gains/Losses Over Time',
            xaxis_title='Date',
            yaxis_title='Percentage Change (%)',
            showlegend=True,
            hovermode='x unified',
            yaxis=dict(
                tickformat='.1f',
                ticksuffix='%'
            )
        )
        
        # Add range selector buttons
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
        
        # Display the plot
        st.plotly_chart(fig)
        
        # Display summary statistics
        st.write("### Performance Summary")
        summary = pd.DataFrame(index=etf_list)
        
        # Calculate different timeframe returns
        current_date = pct_changes.index[-1]
        
        # YTD return
        ytd_start = pd.Timestamp(current_date.year, 1, 1)
        ytd_data = pct_changes[pct_changes.index >= ytd_start]
        summary['YTD (%)'] = ytd_data.iloc[-1].round(2)
        
        # 1-month return
        one_month = current_date - pd.DateOffset(months=1)
        month_data = pct_changes[pct_changes.index >= one_month]
        summary['1 Month (%)'] = (pct_changes.iloc[-1] - month_data.iloc[0]).round(2)
        
        # 3-month return
        three_months = current_date - pd.DateOffset(months=3)
        three_month_data = pct_changes[pct_changes.index >= three_months]
        summary['3 Months (%)'] = (pct_changes.iloc[-1] - three_month_data.iloc[0]).round(2)
        
        # Total return
        summary['Total Return (%)'] = pct_changes.iloc[-1].round(2)
        
        st.dataframe(summary)
        
    except Exception as e:
        logger.error(f"Error calculating percentage gains: {str(e)}")
        st.error("Failed to calculate percentage gains. Please check your data.")

def main():
    st.set_page_config(page_title="Assets Performance Analysis", layout="wide")
    
    st.title("Assets Performance Analysis")
    st.sidebar.header("User Input Options")

    # User input for ETF symbols and period
    default_etfs = "UPRO, SPYG,SPLG,UBS,AMZN,VOO,JPM,BRK-A"
    etf_symbols = st.sidebar.text_input("Enter ETF Symbols (comma-separated)", default_etfs)
    
    # Date inputs with validation
    min_date = pd.to_datetime('2000-01-01')
    max_date = pd.to_datetime('today')
    start_date = st.sidebar.date_input("Select start date", 
                                     pd.to_datetime('2018-01-01'),
                                     min_value=min_date,
                                     max_value=max_date)
    end_date = st.sidebar.date_input("Select end date", 
                                   max_date,
                                   min_value=start_date,
                                   max_value=max_date)

    # Process user input
    etf_list = [symbol.strip().upper() for symbol in etf_symbols.split(',') if symbol.strip()]
    
    if len(etf_list) > 0:
        with st.spinner('Downloading ETF data...'):
            data = download_data(etf_list, start_date, end_date)
            
        if data is not None and not data.empty:
            st.success('Data downloaded successfully!')
            
            # Add data download button
            csv = data.to_csv()
            st.download_button(
                label="Download ETF Data as CSV",
                data=csv,
                file_name="etf_data.csv",
                mime="text/csv",
            )
            
            # Display analysis sections
            st.write("### Historical Price Data")
            st.dataframe(data.tail())
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Price Analysis", "Performance Analysis", "Portfolio Analysis"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Price Movement")
                    plot_price_movement(data, etf_list)
                    
                    st.write("### Moving Average Trend")
                    plot_moving_average(data, etf_list)
                
                with col2:
                    st.write("### Normalized Price Movement")
                    plot_normalized_price(data, etf_list)
                    
                    calculate_risk_metrics(data, etf_list)
            
            with tab2:
                st.write("### Percentage Gains Analysis")
                calculate_percentage_gains(data, etf_list)
            
            with tab3:
                st.write("### Portfolio Optimization")
                plot_efficient_frontier(data)
        else:
            st.error("No data available for the selected ETFs and time period.")

if __name__ == "__main__":
    main()
