import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize





#
# HELPER FUNCTIONS FOR SORTINO
#
@st.cache_data(show_spinner=False)
def downside_deviation(portfolio_monthly_returns, annual_rfr):
    """
    Compute annualized downside deviation based on a monthly portfolio returns series,
    relative to the monthly RFR (annual_rfr/12).
    """
    monthly_rfr = annual_rfr / 12.0
    diff = portfolio_monthly_returns - monthly_rfr
    negative_diff = diff[diff < 0]

    if len(negative_diff) == 0:
        return 0.0  # No negative returns => no downside

    # Monthly downside deviation
    dd_monthly = np.sqrt(np.mean(negative_diff**2))
    # Annualize
    return dd_monthly * np.sqrt(12.0)





@st.cache_data(show_spinner=False)
def portfolio_monthly_returns(returns_df, weights):
    """
    Multiply each column's monthly returns by the corresponding weight, then sum across columns.
    Returns a Series of portfolio returns over time (monthly).
    """
    return returns_df.dot(weights)




@st.cache_data(show_spinner=False)
def compute_max_drawdown(portfolio_returns: pd.Series) -> float:
    """
    Compute maximum drawdown from a portfolio return series (e.g. monthly returns).
    """
    # 1) Build cumulative return curve
    cumulative = (1 + portfolio_returns).cumprod()
    # 2) Track running peaks
    peak = cumulative.cummax()
    # 3) Drawdown series = (current value – peak) / peak
    drawdown = (cumulative - peak) / peak
    # 4) Worst drawdown
    return drawdown.min()





@st.cache_data(show_spinner=False)
def efficient_frontier(
    mean_returns,
    cov_matrix,
    min_weight,
    max_weight,
    annual_rfr,
    returns_df,
    dividend_yields
):
    """
    Calculate the efficient frontier, returning annualized returns, volatilities, 
    Sharpe ratios, Sortino ratios, and portfolio forward dividend yields.
    """
    num_assets = len(mean_returns)

    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def portfolio_dividend_yield(weights):
        return np.dot(weights, dividend_yields)

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = [(min_weight, max_weight) for _ in range(num_assets)]
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)

    efficient_results = {
        "returns": [],
        "volatilities": [],
        "weights": [],
        "sharpe_ratios": [],
        "sortino_ratios": [],
        "dividend_yields": [],
        "max_drawdown": [],    # ← new
    }


    for target_return in target_returns:
        constraints_eq = [
            constraints,
            {"type": "eq", "fun": lambda x, tr=target_return: portfolio_return(x) - tr},
        ]

        result = minimize(
            portfolio_volatility,
            x0=np.full(num_assets, 1.0 / num_assets),
            bounds=bounds,
            constraints=constraints_eq,
        )

        if result.success:
            w = result.x
            port_ret = portfolio_return(w) * 12.0
            port_vol = portfolio_volatility(w) * np.sqrt(12.0)
            sharpe_ratio = (port_ret - annual_rfr) / port_vol if port_vol != 0 else 0
            monthly_port = portfolio_monthly_returns(returns_df, w)
            dd_annual = downside_deviation(monthly_port, annual_rfr)
            sortino_ratio = (port_ret - annual_rfr) / dd_annual if dd_annual != 0 else 0
            port_div_yield = portfolio_dividend_yield(w)
            max_dd= compute_max_drawdown(monthly_port)
            

            efficient_results["returns"].append(port_ret)
            efficient_results["volatilities"].append(port_vol)
            efficient_results["weights"].append(w)
            efficient_results["sharpe_ratios"].append(sharpe_ratio)
            efficient_results["sortino_ratios"].append(sortino_ratio)
            efficient_results["dividend_yields"].append(port_div_yield)
            efficient_results["max_drawdown"].append(max_dd)

    min_vol_idx = int(np.argmin(efficient_results["volatilities"]))
    # Slice each list in efficient_results from the minimum volatility portfolio onward
    for key in efficient_results:
        efficient_results[key] = efficient_results[key][min_vol_idx:]

    return efficient_results





@st.cache_data(show_spinner=False)
def single_stock_portfolio(mean_returns: pd.Series,
                           cov_matrix: pd.DataFrame,
                           annual_rfr: float,
                           returns_df: pd.DataFrame,
                           dividend_yields):
    """
    Metrics for portfolios that are 100% invested in one stock.
    Returns a DataFrame with one row per stock.
    """
    # 1) Vectorisable pieces
    annualised_ret = mean_returns * 12.0
    annualised_vol = np.sqrt(np.diag(cov_matrix)) * np.sqrt(12.0)
    sharpe         = (annualised_ret - annual_rfr) / annualised_vol

    # 2) Sortino (needs downside deviation)
    sortino = []
    for ticker in mean_returns.index:
        dd = downside_deviation(returns_df[ticker], annual_rfr)
        sortino.append((annualised_ret[ticker] - annual_rfr) / dd if dd else 0)

    # 3) Dividend yields
    if isinstance(dividend_yields, (pd.Series, pd.DataFrame)):
        div_yield = dividend_yields.reindex(mean_returns.index).values
    else:
        div_yield = np.asarray(dividend_yields)

    # 4) Max Drawdown (new)
    max_dd = [
        compute_max_drawdown(returns_df[ticker])
        for ticker in mean_returns.index
    ]

    # 5) Assemble DataFrame
    single_stock_portfolio_data = pd.DataFrame({
        "Stock":           mean_returns.index,
        "Return":          annualised_ret.values,
        "Volatility":      annualised_vol,
        "Sharpe Ratio":    sharpe,
        "Sortino Ratio":   sortino,
        "Dividend Yield":  div_yield,
        "Max Drawdown":    max_dd,    # ← new column
        "Weight":          1.0
    })

    return single_stock_portfolio_data










@st.cache_data(show_spinner=False)
def calculate_custom_portfolio_data(
    custom_asset_weights,
    mean_returns,
    cov_matrix,
    annual_rfr,
    returns_df,
    dividend_yields
):
    """
    Calculate metrics for a user-defined custom portfolio, including max drawdown.
    """
    # 1) Validate & extract
    tickers = list(custom_asset_weights.keys())
    weights = np.array([custom_asset_weights[t] for t in tickers])

    if not all(t in mean_returns.index for t in tickers):
        raise ValueError("Some tickers in custom_asset_weights are missing from mean_returns.")
    if not all(t in dividend_yields.index for t in tickers):
        raise ValueError("Some tickers in custom_asset_weights are missing from dividend_yields.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("The custom portfolio weights must sum to 1.0.")

    # 2) Annualized return & volatility
    expected_return      = np.dot(weights, mean_returns.loc[tickers]) * 12.0
    var                  = np.dot(weights.T, np.dot(cov_matrix.loc[tickers, tickers], weights))
    portfolio_volatility = np.sqrt(var) * np.sqrt(12.0)
    sharpe_ratio         = (
        (expected_return - annual_rfr) / portfolio_volatility
        if portfolio_volatility != 0 else 0
    )

    # 3) Monthly portfolio returns series
    monthly_port = returns_df[tickers].dot(weights)

    # 4) Downside deviation & Sortino
    monthly_rfr     = annual_rfr / 12.0
    diff            = monthly_port - monthly_rfr
    neg_diff        = diff[diff < 0]
    downside_dev    = np.sqrt(np.mean(neg_diff**2)) * np.sqrt(12.0) if len(neg_diff) else 0.0
    sortino_ratio   = (
        (expected_return - annual_rfr) / downside_dev
        if downside_dev != 0 else 0
    )

    # 5) Dividend yield
    c_dividend_yield = np.dot(weights, dividend_yields.loc[tickers])

    # 6) Max drawdown via shared helper
    #    (compute_max_drawdown assumes a monthly-return series)
    c_max_drawdown = compute_max_drawdown(monthly_port)

    return {
        "c_expected_return":      expected_return,
        "c_portfolio_volatility": portfolio_volatility,
        "c_sharpe_ratio":         sharpe_ratio,
        "c_sortino_ratio":        sortino_ratio,
        "c_dividend_yield":       c_dividend_yield,
        "c_max_drawdown":         c_max_drawdown,   # ← new
    }






@st.cache_data(show_spinner=False)
def dividend_focused_portfolio(
    mean_returns,
    cov_matrix,
    min_weight,
    max_weight,
    annual_rfr,
    returns_df,
    dividend_yields
):
    """
    Calculate the portfolio with the highest forward dividend yield, adhering to user constraints.

    Now also computes maximum drawdown using the shared helper.
    """
    num_assets = len(mean_returns)

    # Portfolio forward dividend yield
    def portfolio_dividend_yield(weights):
        return np.dot(weights, dividend_yields)

    # Constraints & bounds
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds      = [(min_weight, max_weight) for _ in range(num_assets)]
    initial_guess = np.full(num_assets, 1.0 / num_assets)

    # Optimize: maximize dividend yield
    result = minimize(
        fun=lambda w: -portfolio_dividend_yield(w),
        x0=initial_guess,
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        st.error("Dividend portfolio optimization failed.")
        return None

    # Unpack optimized weights
    weights = result.x

    # Annualized return & volatility
    expected_return      = np.dot(weights, mean_returns) * 12.0
    portfolio_variance   = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(12.0)

    # Sharpe
    sharpe_ratio = (
        (expected_return - annual_rfr) / portfolio_volatility
        if portfolio_volatility != 0
        else 0
    )

    # Monthly portfolio returns series
    monthly_port = returns_df.dot(weights)

    # Sortino
    dd = downside_deviation(monthly_port, annual_rfr)
    sortino_ratio = (
        (expected_return - annual_rfr) / dd
        if dd != 0
        else 0
    )

    # Dividend yield
    div_yield = portfolio_dividend_yield(weights)

    # --- NEW: max drawdown via shared helper ---
    max_dd = compute_max_drawdown(monthly_port)

    return {
        "d_weights":         weights,
        "d_expected_return": expected_return,
        "d_volatility":      portfolio_volatility,
        "d_sharpe_ratio":    sharpe_ratio,
        "d_sortino_ratio":   sortino_ratio,
        "d_dividend_yield":  div_yield,
        "d_max_drawdown":    max_dd,
    }







@st.cache_data(show_spinner=False)
def calculate_ai_refined_portfolio_data(
    ai_weights,
    mean_returns,
    cov_matrix,
    annual_rfr,
    returns_df,
    dividend_yields
):
    """
    Calculate financial metrics for AI-refined portfolios, now including max drawdown.
    """
    portfolio_metrics = {}

    # Ensure dividend_yields is 1-D
    dividend_yields = np.ravel(dividend_yields)

    for portfolio_type, weights in ai_weights.items():
        weights = np.array(weights)

        # 1) Annualized expected return
        expected_return = np.dot(weights, mean_returns) * 12.0

        # 2) Annualized volatility
        var = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(var) * np.sqrt(12.0)

        # 3) Sharpe ratio
        sharpe_ratio = (
            (expected_return - annual_rfr) / portfolio_volatility
            if portfolio_volatility != 0 else 0
        )

        # 4) Monthly portfolio returns + Sortino
        monthly_port = returns_df.dot(weights)
        dd_annual = downside_deviation(monthly_port, annual_rfr)
        sortino_ratio = (
            (expected_return - annual_rfr) / dd_annual
            if dd_annual != 0 else 0
        )

        # 5) Dividend yield
        computed_dividend_yield = float(np.dot(weights, dividend_yields))

        # 6) Max drawdown (new)
        max_dd = compute_max_drawdown(monthly_port)

        portfolio_metrics[portfolio_type] = {
            "expected_return":  expected_return,
            "volatility":       portfolio_volatility,
            "sharpe_ratio":     sharpe_ratio,
            "sortino_ratio":    sortino_ratio,
            "dividend_yield":   computed_dividend_yield,
            "max_drawdown":     max_dd,              # ← new
        }

    return portfolio_metrics


