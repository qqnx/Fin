
# Financial Portfolio Optimization Tool

This repository contains a Streamlit-based web application for financial portfolio optimization.
The application allows users to download historical ETF data, visualize it, and optimize their portfolio using modern portfolio theory.

## Features

- **Download Data**: Fetch historical adjusted close prices for ETFs from Yahoo Finance.
- **Portfolio Optimization**: 
  - Uses `PyPortfolioOpt` to construct the efficient frontier.
  - Calculates optimal portfolio weights based on risk-return trade-offs.
- **Interactive Visualization**:
  - Utilizes Plotly for dynamic financial plots.
  - Provides insights into portfolio performance and allocation.
- **Streamlit Integration**: 
  - User-friendly interface for financial analysis.
  - Real-time feedback and error handling.

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- yFinance
- Plotly
- PyPortfolioOpt

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/portfolio-optimizer.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit application:
    ```bash
    streamlit run Fin7_5.py
    ```

## Usage

1. Enter a list of ETF tickers and select a date range.
2. Download and visualize historical data.
3. Perform portfolio optimization to find the optimal allocation of assets.
4. Explore the efficient frontier and adjust allocations based on risk tolerance.

## File Structure

- `Fin7_5.py`: Main application script.
- `requirements.txt`: List of dependencies.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions

Contributions, issues, and feature requests are welcome!

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

---

Developed with ❤️ by [Your Name].
