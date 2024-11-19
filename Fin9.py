import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pypfopt import EfficientFrontier, risk_models, expected_returns
import logging
import yfinance as yf
from scipy.optimize import minimize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_data(etf_list, start, end):
    """Download ETF data from Yahoo Finance with error handling."""
    try:
        if isinstance(etf_list, str):
            etf_list = [etf_list]
        data = yf.download(etf_list, start=start, end=end)['Adj Close']
        data = data.interpolate(method='linear')
        
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
    if returns.empty or returns.std() < 1e-6:  # Tolerance threshold to avoid division by zero
        return np.nan, np.nan  # Return NaN instead of None for better data handling
    
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    negative_returns = excess_returns[excess_returns < 0]
    sortino_ratio = (excess_returns.mean() / negative_returns.std() * np.sqrt(252)) if not negative_returns.empty else np.nan
    
    return sharpe_ratio, sortino_ratio

def optimize_portfolio(data, method="max_sharpe"):
    """Optimize portfolio allocation using PyPortfolioOpt based on method."""
    if data is None or data.empty:
        st.error("No data available for optimization.")
        return None, None
    try:
        mu = expected_returns.mean_historical_return(data)
        if (mu <= 0.02).all():
            st.error("Optimization failed: at least one of the assets must have an expected return exceeding the risk-free rate.")
            return None, None
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        
        weights = None
        try:
            if method == "max_sharpe":
                weights = ef.max_sharpe()
            elif method == "min_vol":
                weights = ef.min_volatility()
            elif method == "multi_objective":
                weights = multi_objective_optimization(mu, S)
            else:
                st.error("Invalid optimization method selected.")
                return None, None
        except Exception as e:
            logger.error(f"Optimization with EfficientFrontier failed: {e}")
            st.error(f"Optimization failed: {e}")

        if weights is None:
            st.error("Optimization failed: Weights could not be computed.")
            logger.error("Optimization failed: Weights could not be computed. Using equal weight fallback.")
            # Fallback: Use equal weight allocation if optimization fails
            n_assets = len(data.columns)
            weights = {ticker: 1/n_assets for ticker in data.columns}
        else:
            # Clean weights only if they are successfully computed by EfficientFrontier
            weights = ef.clean_weights() if method != "multi_objective" else weights

        # Calculate % change for each asset
        pct_changes = data.apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if not x.isnull().any() else np.nan)

        # Create DataFrame for weights and % change
        weights_df = pd.DataFrame({
            "Asset": weights.keys(),
            "Weight": weights.values(),
            "Total Change (%)": [pct_changes[ticker] for ticker in weights.keys()]
        })

        return weights_df, weights
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        st.error(f"Optimization failed: {e}")
        return None, None

def multi_objective_optimization(mu, S):
    """Perform multi-objective optimization considering return, risk, and drawdown."""
    n_assets = len(mu)
    init_guess = np.array([1/n_assets] * n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    def objective(weights):
        portfolio_return = np.dot(weights, mu)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        return -portfolio_return / portfolio_volatility  # Negative Sharpe Ratio to minimize
    
    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        logger.error(f"Multi-objective optimization failed: {result.message}")
        return None
    return dict(zip(mu.index, result.x))

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

def plot_percentage_gains(data):
    """Plot percentage gains/losses over time using Plotly Express."""
    pct_changes = ((data - data.iloc[0]) / data.iloc[0] * 100)
    fig = px.line(pct_changes, x=pct_changes.index, y=pct_changes.columns, 
                  labels={'value': 'Percentage Change (%)', 'index': 'Date'}, 
                  title='Percentage Gains/Losses Over Time')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Percentage Change (%)',
        hovermode='x unified'
    )
    fig.update_yaxes(tickformat='.1f')
    return fig

def backtest_portfolio(data, weights, benchmark_tickers, start_date, end_date, initial_capital, risk_free_rate):
    """Backtest optimized portfolio and compare with benchmarks."""
    portfolio_returns = (data.pct_change(fill_method=None).dropna() * pd.Series(weights)).sum(axis=1)
    portfolio_growth = (1 + portfolio_returns).cumprod() * initial_capital

    sharpe, sortino = calculate_sharpe_sortino(portfolio_returns, risk_free_rate=risk_free_rate)

    benchmarks = {}
    benchmark_ratios = []
    for ticker in benchmark_tickers:
        try:
            benchmark_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            benchmark_data = benchmark_data.interpolate(method='linear')
            if benchmark_data.empty or benchmark_data.isnull().all():
                st.warning(f"No valid data for benchmark {ticker}")
                continue
            benchmark_returns = benchmark_data.pct_change(fill_method=None).dropna()
            if benchmark_returns.empty:
                st.warning(f"No valid returns data for benchmark {ticker}")
                continue
            benchmark_growth = (1 + benchmark_returns).cumprod() * initial_capital
            sharpe_b, sortino_b = calculate_sharpe_sortino(benchmark_returns, risk_free_rate=risk_free_rate)
            benchmarks[ticker] = benchmark_growth
            benchmark_ratios.append((sharpe_b, sortino_b))
        except Exception as e:
            logger.error(f"Failed to fetch benchmark data for {ticker}: {e}")
            st.error(f"Failed to fetch benchmark data for {ticker}: {e}")
            continue

    return portfolio_growth, sharpe, sortino, benchmarks, benchmark_ratios

# Streamlit Layout
st.set_page_config(page_title="Portfolio Optimization and Backtesting", layout="wide")

st.title('Portfolio Optimization and Backtesting')

# Sidebar for user input
with st.sidebar:
    st.header("User Input Options")
    etf_list = st.text_input('Enter ETF tickers separated by commas (e.g., SPY, QQQ, EEM)', 'SPY, QQQ')
    etf_list = [etf.strip() for etf in etf_list.split(',')]
    start_date = st.date_input('Select Start Date', value=datetime(2020, 1, 1))
    end_date = st.date_input('Select End Date', value=datetime(2023, 1, 1))
    if end_date > datetime.now().date():
        st.warning("End date cannot be in the future. Using today's date instead.")
        end_date = datetime.now().date()

    benchmark_tickers = st.text_input('Enter Benchmark Tickers separated by commas (e.g., SPY, VOO)', 'SPY, VOO')
    benchmark_tickers = [ticker.strip() for ticker in benchmark_tickers.split(',')]
    initial_capital = st.number_input('Enter Initial Investment Amount (USD)', min_value=1000, value=5000, step=100)

    st.header("Select Optimization Methods")
    opt_max_sharpe = st.checkbox("Max Sharpe Ratio")
    opt_min_vol = st.checkbox("Min Volatility")
    opt_multi_objective = st.checkbox("Multi-Objective")

    # Slider for risk-free rate adjustment
    st.header("Risk-Free Rate Adjustment")
    risk_free_rate = st.slider('Select Risk-Free Rate (%)', 0.0, 5.0, 2.0, step=0.1) / 100

    # Manual risk-free rate input
    st.header("Manual Risk-Free Rate Input")
    manual_risk_free_rate = st.number_input("Or enter a custom risk-free rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
    if manual_risk_free_rate != risk_free_rate:
        risk_free_rate = manual_risk_free_rate

    # Add a start button for data downloading and elaboration
    if st.button('Start Analysis'):
        run_analysis = True
    else:
        run_analysis = False

# Tabs for different analyses and outputs
tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis", "Percentage Gains Analysis", "Optimization & Backtesting", "Benchmark Comparison"])

with tab1:
    st.header("Price Movements of Selected ETFs")
    if run_analysis:
        data = download_data(etf_list, start_date, end_date)
        if data is not None:
            st.plotly_chart(plot_individual_tickers(data), use_container_width=True)
        else:
            st.error("No data available for the selected ETFs and time period.")
    else:
        st.info("Click 'Start Analysis' to begin data downloading and elaboration.")

with tab2:
    st.header("Percentage Gains Analysis")
    if run_analysis and 'data' in locals() and data is not None:
        st.plotly_chart(plot_percentage_gains(data), use_container_width=True)
    else:
        st.info("Click 'Start Analysis' to view percentage gains analysis.")

with tab3:
    st.header("Portfolio Optimization and Backtesting")
    if run_analysis and 'data' in locals() and data is not None:
        selected_methods = []
        if opt_max_sharpe:
            selected_methods.append("max_sharpe")
        if opt_min_vol:
            selected_methods.append("min_vol")
        if opt_multi_objective:
            selected_methods.append("multi_objective")

        portfolio_results = []
        for method in selected_methods:
            weights_df, weights = optimize_portfolio(data, method=method)
            if weights_df is not None:
                portfolio_growth, sharpe_ratio, sortino_ratio, benchmarks, benchmark_ratios = backtest_portfolio(
                    data, weights, benchmark_tickers, start_date, end_date, initial_capital, risk_free_rate
                )
                st.subheader(f"Optimized Portfolio ({method.replace('_', ' ').title()})")
                st.markdown("---")
                st.plotly_chart(plot_backtest(portfolio_growth, benchmarks), use_container_width=True)
                weights_df['Sharpe Ratio'] = sharpe_ratio
                weights_df['Sortino Ratio'] = sortino_ratio
                st.dataframe(weights_df.style.format({'Weight': '{:.2%}', 'Total Change (%)': '{:.2f}%', 'Sharpe Ratio': '{:.2f}', 'Sortino Ratio': '{:.2f}'}))
                portfolio_results.append({
                    'Name': f'Optimized Portfolio ({method.replace("_", " ").title()})',
                    'Final Value (USD)': portfolio_growth.iloc[-1],
                    'Total Change (%)': (portfolio_growth.iloc[-1] / initial_capital - 1) * 100,
                    'Sharpe Ratio': sharpe_ratio,
                    'Sortino Ratio': sortino_ratio
                })
            else:
                st.error(f"Optimization failed for method: {method}")

with tab4:
    st.header("Benchmark Comparison")
    if run_analysis and 'benchmarks' in locals():
        benchmark_results = portfolio_results.copy()  # Start with portfolio results
        for i, (ticker, growth) in enumerate(benchmarks.items()):
            sharpe_b, sortino_b = benchmark_ratios[i] if len(benchmark_ratios) > i else (np.nan, np.nan)
            benchmark_results.append({
                'Name': ticker,
                'Final Value (USD)': growth.iloc[-1],
                'Total Change (%)': (growth.iloc[-1] / initial_capital - 1) * 100,
                'Sharpe Ratio': sharpe_b,
                'Sortino Ratio': sortino_b
            })
        benchmark_df = pd.DataFrame(benchmark_results)
        st.write("### Benchmark Results")
        st.dataframe(benchmark_df.style.format({
            'Final Value (USD)': '${:,.2f}',
            'Total Change (%)': '{:.2f}%',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}'
        }))
    else:
        st.info("Click 'Start Analysis' to compare benchmarks.")
