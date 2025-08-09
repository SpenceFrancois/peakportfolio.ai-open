# dashboard/b_download.py

import pandas as pd
import io


def prepare_efficient_frontier_download(efficient_results, tickers):
    """
    Prepares Efficient Frontier data for download.

    :param efficient_results: Dictionary containing efficient frontier data ('returns', 'volatilities', 'sharpe_ratios', 'sortino_ratios', 'dividend_yields', and 'weights').
    :param tickers: List of stock tickers.
    :return: CSV string for download.
    """
    # Create a DataFrame directly from the dictionary
    ef_df = pd.DataFrame({
        "Expected Return": efficient_results["returns"],
        "Standard Deviation": efficient_results["volatilities"],
        "Sharpe Ratio": efficient_results["sharpe_ratios"],
        "Sortino Ratio": efficient_results["sortino_ratios"],
        "Dividend Yield": efficient_results["dividend_yields"]  # Add Dividend Yield
    })

    # Append weights as separate columns
    weights_df = pd.DataFrame(efficient_results["weights"], columns=tickers).round(2)
    ef_df = pd.concat([ef_df, weights_df], axis=1)

    # Convert to CSV and return
    return ef_df.to_csv(index=False)



def prepare_portfolio_download(
    max_sharpe_allocation_df,
    max_sharpe_summary_df,
    max_return_allocation_df,
    max_return_summary_df,
    returns,
    tickers,
    min_volatility_allocation_df=None,
    min_volatility_summary_df=None,
    dividend_allocation_df=None,        # Renamed
    dividend_summary_df=None,           # Renamed
    correlation_matrix=None,
    covariance_matrix=None,
    custom_allocation_df=None,
    custom_summary_df=None,
    portfolio_returns_df=None,
    cumulative_returns_df=None 
):
    """
    Prepares portfolio data for download as an Excel file, combining allocation and summary data 
    for each portfolio type into single sheets.

    Parameters:
        max_sharpe_allocation_df: DataFrame for Max Sharpe Ratio Portfolio Allocation.
        max_sharpe_summary_df: DataFrame for Max Sharpe Ratio Portfolio Summary.
        max_return_allocation_df: DataFrame for Max Return Portfolio Allocation.
        max_return_summary_df: DataFrame for Max Return Portfolio Summary.
        returns: DataFrame containing asset returns with dates as the index.
        tickers: List of stock tickers.
        min_volatility_allocation_df: DataFrame for Min Volatility Portfolio Allocation (optional).
        min_volatility_summary_df: DataFrame for Min Volatility Portfolio Summary (optional).
        dividend_allocation_df: DataFrame for Dividend-Focused Portfolio Allocation (optional).
        dividend_summary_df: DataFrame for Dividend-Focused Portfolio Summary (optional).
        correlation_matrix: Correlation matrix DataFrame (optional).
        covariance_matrix: Covariance matrix DataFrame (optional).
        custom_allocation_df: DataFrame for Custom Portfolio Allocation (optional).
        custom_summary_df: DataFrame for Custom Portfolio Summary (optional).

    Returns:
        Binary Excel file for download.
    """
    output = io.BytesIO()

    # Copy returns, adding a 'dates' column for clarity in the Excel output
    returns_with_dates = returns.copy()
    returns_with_dates["dates"] = returns_with_dates.index.astype(str)
    cols = ["dates"] + [col for col in returns_with_dates.columns if col != "dates"]
    returns_with_dates = returns_with_dates[cols]

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Helper function to write allocation + summary to a single sheet
        def write_combined_sheet(sheet_name, allocation_df, summary_df):
            startrow = 0
            if allocation_df is not None and not allocation_df.empty:
                allocation_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=startrow)
                startrow += len(allocation_df.index) + 3  # space between tables
            if summary_df is not None and not summary_df.empty:
                summary_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=startrow)

        # Conservative (Min Volatility)
        write_combined_sheet("Conservative", min_volatility_allocation_df, min_volatility_summary_df)

        # Balanced (Max Sharpe)
        write_combined_sheet("Balanced", max_sharpe_allocation_df, max_sharpe_summary_df)

        # Aggressive (Max Return)
        write_combined_sheet("Aggressive", max_return_allocation_df, max_return_summary_df)

        # Dividend-Focused
        write_combined_sheet("Dividend Focused", dividend_allocation_df, dividend_summary_df)

        # Custom Portfolio
        write_combined_sheet("Custom Portfolio", custom_allocation_df, custom_summary_df)

        # --- NEW tab with cumulative returns -----------------------------
        if cumulative_returns_df is not None and not cumulative_returns_df.empty:
            cr = cumulative_returns_df.copy()
            cr.insert(0, "dates", cr.index.astype(str))
            cr.to_excel(writer, index=False, sheet_name="Cumulative Returns")

        # --- NEW sheet: portfolio & benchmark returns ------------------
        if portfolio_returns_df is not None and not portfolio_returns_df.empty:
            pr = portfolio_returns_df.copy()
            pr.insert(0, "dates", pr.index.astype(str))
            pr.to_excel(writer, index=False, sheet_name="Portfolio Returns")

        # Correlation Matrix
        if correlation_matrix is not None:
            correlation_matrix.to_excel(writer, index=True, sheet_name="Correlation Matrix")

        # Covariance Matrix
        if covariance_matrix is not None:
            covariance_matrix.to_excel(writer, index=True, sheet_name="Covariance Matrix")

        # Returns Sheet
        if returns_with_dates is not None and not returns_with_dates.empty:
            returns_with_dates.to_excel(writer, index=False, sheet_name="Returns")

        # Summary Notes Sheet
        summary_notes = pd.DataFrame({
            "Note": [
                "This file contains portfolio analysis data, including allocations, summaries, correlations, and covariance matrix.",
                "Use the data to evaluate portfolio diversification, risk-return profiles, and result accuracy."
            ]
        })
        summary_notes.to_excel(writer, index=False, sheet_name="Summary Notes")

    output.seek(0)
    return output.getvalue()
