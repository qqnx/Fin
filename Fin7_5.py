import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
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
        
        # Ensure data is a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=etf_list[0])
        
        if data.empty:
            st.error("No data retrieved. Please check your tickers or date range.")
            return None
        missing_tickers = [ticker for ticker in etf_list if ticker not in data.columns]
        if missing_tickers:
            st.warning(f"Data not available for: {', '.join(missing_tickers)}")
        return data
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        st.error(f"Failed to fetch data: {str(e)}")
        return None

def calculate_sharpe_sortino(returns, risk_free_rate=0.02):
    """Calculate Sharpe and Sortino ratios with robust handling for missing or insufficient data."""
    if returns.empty or returns.std() == 0:
        return None, None  # Avoid division by zero or empty returns
    
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    negative_returns = excess_returns[excess_returns < 0]
    if negative_returns.empty or negative_returns.std() == 0:
        sortino_ratio = None  # No negative returns to calculate Sortino
    else:
        sortino_ratio = excess_returns.mean() / negative_returns.std() * np.sqrt(252)
    
    return sharpe_ratio, sortino_ratio

def optimize_portfolio(data, method="max_sharpe"):
    """Optimize portfolio allocation using PyPortfolioOpt based on method."""
    if data is None or data.empty:
        st.error("No data available for optimization.")
        return None, None
    try:
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        
        if method == "max_sharpe":
            weights = ef.max_sharpe()
        elif method == "min_vol":
            weights = ef.min_volatility()
        
        cleaned_weights = ef.clean_weights()

        # Calculate % change for each asset
        pct_changes = data.apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if not x.isnull().any() else None)

        # Create DataFrame for weights and % change
        weights_df = pd.DataFrame({
            "Asset": cleaned_weights.keys(),
            "Weight": cleaned_weights.values(),
            "Total Change (%)": [pct_changes[ticker] for ticker in cleaned_weights.keys()]
        })

        return weights_df, cleaned_weights
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return None, None

def plot_individual_tickers(data):
    """Plot individual price movements for each ticker."""
    fig = go.Figure()
    for ticker in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=ticker))
    fig.update_layout(
        title='Individual Ticker Price Movements',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        showlegend=True
    )
    return fig

def plot_backtest(portfolio_growth, benchmarks):
    """Plot backtest comparison between portfolio and benchmarks."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_growth.index, y=portfolio_growth, mode='lines', name='Portfolio'))

    for ticker, benchmark_growth in benchmarks.items():
        fig.add_trace(go.Scatter(x=benchmark_growth.index, y=benchmark_growth, mode='lines', name=ticker))

    fig.update_layout(
        title='Backtest: Portfolio vs Benchmarks',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (USD)',
        showlegend=True
    )
    return fig

def backtest_portfolio(data, weights, benchmark_tickers, start_date, end_date, initial_capital):
    """Backtest optimized portfolio and compare with benchmarks."""
    portfolio_returns = (data.pct_change(fill_method=None).dropna() * pd.Series(weights)).sum(axis=1)
    portfolio_growth = (1 + portfolio_returns).cumprod() * initial_capital

    sharpe, sortino = calculate_sharpe_sortino(portfolio_returns)

    benchmarks = {}
    benchmark_ratios = []
    for ticker in benchmark_tickers:
        try:
            benchmark_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            if benchmark_data.empty:
                st.warning(f"No data for benchmark {ticker}")
                continue
            benchmark_returns = benchmark_data.pct_change(fill_method=None).dropna()
            benchmark_growth = (1 + benchmark_returns).cumprod() * initial_capital
            sharpe_b, sortino_b = calculate_sharpe_sortino(benchmark_returns)
            benchmarks[ticker] = benchmark_growth
            benchmark_ratios.append((sharpe_b, sortino_b))
        except Exception as e:
            st.error(f"Failed to fetch benchmark data for {ticker}: {e}")
            continue

    return portfolio_growth, sharpe, sortino, benchmarks, benchmark_ratios

# Streamlit Layout
st.set_page_config(layout="wide")

st.title('Portfolio Optimization and Backtesting')

# Sidebar for user input
with st.sidebar:
    st.header("User Input")
    etf_list = st.text_input('Enter ETF tickers separated by commas (e.g., SPY, QQQ, EEM)', 'SPY, QQQ')
    etf_list = [etf.strip() for etf in etf_list.split(',')]
    start_date = st.date_input('Start Date', value=datetime(2020, 1, 1))
    end_date = st.date_input('End Date', value=datetime(2023, 1, 1))
    if end_date > datetime.now().date():
        st.warning("End date cannot be in the future. Using today's date instead.")
        end_date = datetime.now().date()

    benchmark_tickers = st.text_input('Enter Benchmark Tickers separated by commas (e.g., SPY, VOO)', 'SPY, VOO')
    benchmark_tickers = [ticker.strip() for ticker in benchmark_tickers.split(',')]
    initial_capital = st.number_input('Enter Initial Investment Amount (USD)', min_value=1000, value=5000, step=100)

    optimization_method = st.radio("Select Optimization Method", ("Max Sharpe Ratio", "Min Volatility", "Both"))

if st.sidebar.button('Run Analysis'):
    data = download_data(etf_list, start_date, end_date)
    benchmarks = {}  # Inizializzazione corretta
    benchmark_ratios_sharpe = []
    benchmark_ratios_vol = []

    if data is not None:
        st.plotly_chart(plot_individual_tickers(data), use_container_width=True)

        results_data = []
        benchmark_results = []

        if optimization_method in ["Max Sharpe Ratio", "Both"]:
            weights_df_sharpe, weights_sharpe = optimize_portfolio(data, method="max_sharpe")
            if weights_df_sharpe is not None:
                portfolio_growth_sharpe, sharpe_ratio, sortino_ratio, benchmarks, benchmark_ratios_sharpe = backtest_portfolio(
                    data, weights_sharpe, benchmark_tickers, start_date, end_date, initial_capital
                )
                fig_sharpe = plot_backtest(portfolio_growth_sharpe, benchmarks)
                col_sharpe1, col_sharpe2 = st.columns(2)
                col_sharpe1.subheader("**Optimized Portfolio (Max Sharpe):**")
                col_sharpe1.dataframe(weights_df_sharpe.style.format({'Weight': '{:.2%}', 'Total Change (%)': '{:.2f}%'}))
                col_sharpe2.plotly_chart(fig_sharpe, use_container_width=True)
                results_data.append({
                    'Name': 'Portfolio (Sharpe)',
                    'Final Value (USD)': portfolio_growth_sharpe.iloc[-1],
                    'Total Change (%)': (portfolio_growth_sharpe.iloc[-1] / initial_capital - 1) * 100,
                    'Sharpe Ratio': sharpe_ratio,
                    'Sortino Ratio': sortino_ratio
                })
            else:
                st.error("Max Sharpe optimization failed.")

        if optimization_method in ["Min Volatility", "Both"]:
            weights_df_vol, weights_vol = optimize_portfolio(data, method="min_vol")
            if weights_df_vol is not None:
                portfolio_growth_vol, sharpe_ratio_vol, sortino_ratio_vol, benchmarks, benchmark_ratios_vol = backtest_portfolio(
                    data, weights_vol, benchmark_tickers, start_date, end_date, initial_capital
                )
                fig_vol = plot_backtest(portfolio_growth_vol, benchmarks)
                col_vol1, col_vol2 = st.columns(2)
                col_vol1.subheader("**Optimized Portfolio (Min Volatility):**")
                col_vol1.dataframe(weights_df_vol.style.format({'Weight': '{:.2%}', 'Total Change (%)': '{:.2f}%'}))
                col_vol2.plotly_chart(fig_vol, use_container_width=True)
                results_data.append({
                    'Name': 'Portfolio (Min Vol)',
                    'Final Value (USD)': portfolio_growth_vol.iloc[-1],
                    'Total Change (%)': (portfolio_growth_vol.iloc[-1] / initial_capital - 1) * 100,
                    'Sharpe Ratio': sharpe_ratio_vol,
                    'Sortino Ratio': sortino_ratio_vol
                })
            else:
                st.error("Min Volatility optimization failed.")

        # Evitare duplicati per i benchmark e costruire solo una tabella benchmark chiara
        for i, (ticker, growth) in enumerate(benchmarks.items()):
            if len(benchmark_ratios_sharpe) > i:
                sharpe_b, sortino_b = benchmark_ratios_sharpe[i]
            elif len(benchmark_ratios_vol) > i:
                sharpe_b, sortino_b = benchmark_ratios_vol[i]
            else:
                sharpe_b, sortino_b = None, None
            benchmark_results.append({
                'Name': ticker,
                'Final Value (USD)': growth.iloc[-1],
                'Total Change (%)': (growth.iloc[-1] / initial_capital - 1) * 100,
                'Sharpe Ratio': sharpe_b,
                'Sortino Ratio': sortino_b
            })

        # Creazione due DataFrame finali 
        results_df = pd.DataFrame(results_data)
        benchmark_df = pd.DataFrame(benchmark_results)

        st.subheader("**Backtesting Results:**")
        st.write("### Portfolio Results")
        st.dataframe(results_df.style.format({
            'Final Value (USD)': '${:,.2f}',
            'Total Change (%)': '{:.2f}%',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}'
        }))

        st.write("### Benchmark Results")
        st.dataframe(benchmark_df.style.format({
            'Final Value (USD)': '${:,.2f}',
            'Total Change (%)': '{:.2f}%',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}'
        }))
