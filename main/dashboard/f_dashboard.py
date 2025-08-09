import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
from .tiingo_client import client
from pathlib import Path
import base64
from io import BytesIO
from dashboard.b_data import fetch_and_resample_data, fetch_benchmark_data
from dashboard.b_optim import (
    efficient_frontier,
    single_stock_portfolio,
    calculate_custom_portfolio_data,
    dividend_focused_portfolio,
)
from dashboard.b_backtest import portfolio_returns, cumulative_growth
from dashboard.b_plot import (
    plot_efficient_frontier,
    plot_min_volatility_allocation,
    plot_max_sharpe_allocation,
    plot_max_return_allocation,
    plot_dividend_allocation,
    plot_custom_portfolio_allocation,
    plot_portfolio_vs_benchmark
)
from dashboard.b_chat_engine import query_gpt, fetch_recent_events, prepare_simulation_data_for_prompt
from dashboard.b_ai_allocator import ai_portfolio_refinement
from dashboard.b_download import prepare_efficient_frontier_download, prepare_portfolio_download
from dashboard.particle_component import render_particles
from dashboard.f_style import apply_custom_css
from streamlit_autorefresh import st_autorefresh  # ⇠ grouped with other imports
from concurrent.futures import ThreadPoolExecutor






# Apply custom CSS
apply_custom_css()


@st.cache_data(show_spinner=True)
def ticker_validity_check(symbol: str,
                          start: str = "1980-01-01",
                          end: str | None = None) -> bool:
    """
    True ⇢ Tiingo returns at least two adjClose rows between *start* and *end*,
    and the latest one is no more than 60 days old.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cutoff = datetime.today() - timedelta(days=60)

    try:
        data = client.get_ticker_price(
            symbol,
            startDate=start,
            endDate=end,
            frequency="monthly",
            fmt="json",
            columns=["adjClose", "date"],
        )
        if not data or len(data) < 2:
            return False

        dates = [datetime.fromisoformat(d["date"].split("T")[0]) for d in data]
        latest = max(dates)
        return latest >= cutoff

    except Exception:
        return False






@st.cache_data(show_spinner=False)
def get_earliest_date(stock: str):
    """
    Retrieves the earliest available date for a stock using Tiingo's price history.
    """
    try:
        price_data = client.get_ticker_price(
            stock,
            startDate="1980-01-01",
            endDate=datetime.today().strftime("%Y-%m-%d"),
            frequency="monthly"
        )
        if price_data:
            dates = pd.to_datetime([entry['date'] for entry in price_data])
            return dates.min().date()
        return None
    except Exception:
        return None




@st.cache_data(show_spinner=False)
def cached_fetch_benchmark_data(benchmark_symbol: str, start_date: str, end_date: str):
    return fetch_benchmark_data(benchmark_symbol, start_date, end_date)




# -------------------------------
# End of Caching Enhancements
# -------------------------------


def dashboard_page(product_id=None):
    
    def is_pro(product_id: str | None) -> bool:
        return True  # Temporarily allow full access for everyone

    

    # Display logo if available
    logo_path = Path("logo1.svg")
    if logo_path.exists():
        with open(logo_path, "rb") as file:
            svg_base64 = base64.b64encode(file.read()).decode("utf-8")
        st.markdown(
            f"""
            <div style="text-align: center; padding-top: 0px; margin-top: 15px;">
                <img src="data:image/svg+xml;base64,{svg_base64}" alt="PeakPortfolio Logo" style="width: 500px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(f"Logo not found at {logo_path.resolve()}")

    # Placeholders for initial instructions and particles
    subtitle_placeholder = st.empty()
    particle_placeholder = st.empty()

    _, left, center, right, _ = st.columns([3.8, 4.7, 45, 4.7, 3.8])

    # Initialize session state variables
    if 'parameters' not in st.session_state:
        st.session_state.parameters = {}
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = {}
    if 'simulation_ran' not in st.session_state:
        st.session_state.simulation_ran = False







    with center:
        st.sidebar.markdown(
            "<h1 style='margin-top: 85px; font-size: 1.3em;'>Design Your Portfolio</h1>",
            unsafe_allow_html=True
        )
        with st.sidebar.expander("Craft Your Investment Strategy", expanded=True):
            
            # --- Enter Account Size ---
            import re, locale
            locale.setlocale(locale.LC_ALL, '')

            @st.fragment
            def account_size_input():
                raw = st.text_input("Portfolio size", "$10,000.00", placeholder="$0.00")
                clean = re.sub(r"[^\d.]", "", raw)
                account_size = float(clean) if clean else 0
                return raw, account_size

            raw, account_size = account_size_input()









            # --- Enter Asset Symbols ---
            @st.fragment
            def asset_symbol_input(raw_tickers_default, product_id):
                raw_tickers_input = st.text_input(
                    "Enter Asset Symbols",
                    placeholder="Ex: SPY, AAPL, GLD, JEPI",
                    help="Enter assets you want to invest in (Yahoo/Google Finance symbols).",
                    value=raw_tickers_default
                )

                tickers = sorted({
                    stock.strip().upper()
                    for stock in raw_tickers_input.replace(" ", ",").split(",")
                    if stock.strip()
                })

                max_free_tickers = 5
                max_pro_tickers = 50
                invalid_symbols = []

                if not is_pro(product_id) and len(tickers) > max_free_tickers:
                    st.error(f"Free users can enter up to {max_free_tickers} tickers.")
                    st.markdown(
                        """
                        <a href="https://peakportfolio.ai/#Pricing" target="_blank" style="
                            text-decoration: none;
                            color: #0a4daa;
                            font-weight: 600;
                            font-size: 16px;
                            display: inline-flex;
                            align-items: center;
                            margin-left: 10px;
                        ">
                            🔓 Upgrade to Pro
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
                    tickers = []
                elif is_pro(product_id) and len(tickers) > max_pro_tickers:
                    st.error(f"Pro users can enter up to {max_pro_tickers} tickers.")
                    tickers = []

                if tickers:
                    with st.spinner("Validating symbols..."):
                        def validate_symbols(tickers_list):
                            with ThreadPoolExecutor() as executor:
                                results = list(executor.map(ticker_validity_check, tickers_list))
                            return [stock for stock, valid in zip(tickers_list, results) if not valid]

                        invalid_symbols = validate_symbols(tickers)

                    if invalid_symbols:
                        st.warning(f"Invalid symbols detected: {', '.join(invalid_symbols)}")
                    else:
                        st.success("All symbols are valid.")

                return tickers

            raw_tickers_default = ""  # Or use saved/default portfolio tickers
            tickers = asset_symbol_input(raw_tickers_default, product_id)


            # --- AI Strategy Context ---
            @st.fragment
            def ai_strategy_context(tickers, product_id):
                is_user_pro = is_pro(product_id)
                ai_context_input = ""

                if is_user_pro:
                    if len(tickers) > 20:
                        st.text_area(
                            label="Craft Your Strategy",
                            value="AI portfolios are limited to 20 assets. Reduce your selection to unlock AI-powered refinement.",
                            disabled=True,
                            height=195
                        )
                    else:
                        ai_context_input = st.text_area(
                            label="Craft Your Strategy",
                            placeholder="Provide the details about what you're looking for in your portfolio.",
                            help="Provide any details to help refine your portfolio using AI.",
                            height=195
                        )
                else:
                    st.text_area(
                        label="Craft Your Strategy (Pro Only)",
                        value="Tell us what you're aiming for and get a portfolio that's built to get you there — tailored to you.\n",
                        disabled=True,
                        height=195
                    )

                    st.markdown(
                        """
                        <a href="https://peakportfolio.ai/#Pricing" target="_blank" style="
                            text-decoration: none;
                            color: #0a4daa;
                            font-weight: 600;
                            font-size: 16px;
                            display: inline-flex;
                            align-items: center;
                            margin-left: 10px;
                        ">
                            🔓 Upgrade to Pro
                        </a>
                        """,
                        unsafe_allow_html=True
                    )

                return ai_context_input

            # --- Call Fragment ---
            ai_context_input = ai_strategy_context(tickers, product_id)










        # ---- Portfolio controls ---------------------------
        @st.fragment
        def portfolio_allocation_fragment():
            st.markdown(
                "<h1 style='margin-top: 65px; font-size: 1.3em;'>Fine-Tune Your Portfolio</h1>",
                unsafe_allow_html=True,
            )

            with st.expander("Portfolio Allocation Settings", expanded=False):
                min_weight = st.slider(
                    "Minimum Allocation Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    help="Minimum allocation per asset."
                )
                max_weight = st.slider(
                    "Maximum Allocation Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.75,
                    step=0.01,
                    help="Maximum allocation per asset."
                )
                annual_rfr = st.slider(
                    "Risk-Free Rate (Annualized)",
                    min_value=0.00,
                    max_value=0.08,
                    value=0.03,
                    step=0.001,
                    format="%.3f",
                    help="Annualized risk-free return (e.g., government bonds)."
                )

                include_single_stock_str = st.selectbox(
                    "Plot Single Assets?",
                    options=["Yes", "No"],
                    index=1
                )
                include_single_stock = (include_single_stock_str == "Yes")

            return min_weight, max_weight, annual_rfr, include_single_stock


        # ---- Benchmark control -----------------------------
        @st.fragment
        def benchmark_fragment(product_id):
            with st.expander("Set Benchmark", expanded=False):
                if is_pro(product_id):
                    return st.text_input(
                        "Enter Benchmark Symbol",
                        value="SPY",
                        placeholder="Ex: SPY",
                        help="Enter the benchmark asset symbol for comparison (e.g., SPY for S&P 500)."
                    ).strip().upper()
                else:
                    benchmark = "SPY"
                    st.text_input(
                        "Benchmark (Pro Only)",
                        value=benchmark,
                        disabled=True,
                        help="Want to compare your portfolio to QQQ, AAPL, or international indexes? [**Unlock Now**](https://peakportfolio.ai/#Pricing)"
                    )
                    st.markdown(
                        """
                        <a href="https://peakportfolio.ai/#Pricing" target="_blank" style="
                            text-decoration: none;
                            color: #0a4daa;
                            font-weight: 600;
                            font-size: 16px;
                            display: inline-flex;
                            align-items: center;
                            margin-left: 10px;
                        ">
                            🔓 Upgrade to Pro
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
                    return benchmark


        # ---- Sidebar assembly ------------------------------
        with st.sidebar:
            min_weight, max_weight, annual_rfr, include_single_stock = portfolio_allocation_fragment()
            benchmark = benchmark_fragment(product_id)







            # ---------------- Set Your Timeline ----------------
            @st.fragment
            def timeline_fragment(tickers=None, benchmark=None, product_id=None):
                """Render the timeline selector and return the chosen start/end dates with paywall logic."""
                tickers = tickers or []
                benchmark = benchmark or ""

                with st.expander("Set Your Timeline", expanded=False):
                    # --- Helper to generate month list ---
                    def generate_months(start_date, end_date):
                        if isinstance(start_date, datetime):
                            start_date = start_date.date()
                        if isinstance(end_date, datetime):
                            end_date = end_date.date()

                        months = []
                        current = start_date.replace(day=1)
                        while current <= end_date:
                            months.append(current.strftime("%B %Y"))
                            current = (
                                current.replace(year=current.year + 1, month=1)
                                if current.month == 12
                                else current.replace(month=current.month + 1)
                            )
                        return months

                    # --- Determine earliest data date ---
                    earliest_dates = []

                    if tickers:
                        earliest_dates += [d for d in (get_earliest_date(t) for t in tickers) if d]

                    bd = get_earliest_date(benchmark)
                    if bd:
                        earliest_dates.append(bd)

                    base_start_date = max(earliest_dates) if earliest_dates else datetime(2010, 1, 1).date()

                    today_date = datetime.today().date()

                    # --- Apply paywall for free users ---
                    if not is_pro(product_id):
                        five_years_ago = today_date.replace(year=today_date.year - 5)
                        if base_start_date < five_years_ago:
                            base_start_date = five_years_ago
                            st.error("Upgrade to Pro for full timeline access.")
                            st.markdown(
                                """
                                <a href="https://peakportfolio.ai/#Pricing" target="_blank" style="
                                    text-decoration: none;
                                    color: #0a4daa;
                                    font-weight: 600;
                                    font-size: 16px;
                                    display: inline-flex;
                                    align-items: center;
                                    margin-left: 10px;
                                ">
                                    🔓 Upgrade to Pro
                                </a>
                                """,
                                unsafe_allow_html=True
                            )

                    # --- Build month list & defaults ---
                    months = generate_months(base_start_date, today_date)
                    default_end = len(months) - 2 if len(months) >= 2 else len(months) - 1
                    default_start = max(0, default_end - 60)

                    start_date_str = st.selectbox("Start Date", months, index=default_start)
                    start_date = datetime.strptime(start_date_str, "%B %Y").strftime("%Y-%m-%d")

                    end_date_str = st.selectbox("End Date", months, index=default_end)
                    end_raw = datetime.strptime(end_date_str, "%B %Y").replace(day=1)
                    end_date = (end_raw + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")

                    return start_date, end_date


            # ───────────── Call Fragment in Sidebar ─────────────
            with st.sidebar:
                start_date, end_date = timeline_fragment(tickers=tickers, benchmark=benchmark, product_id=product_id)




        # ---------------- Custom Asset Weights ----------------
        @st.fragment
        def custom_asset_weights_fragment(tickers, product_id):
            weights = {}
            total_custom_weight = 0.0

            if tickers:
                if is_pro(product_id):
                    for stock in tickers:
                        w = st.number_input(
                            label=f"{stock} Weight (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            step=5.0,
                            format="%.2f",
                            help=f"Enter the weight for {stock} as a percentage (e.g., 20.0 = 20%)."
                        )
                        weights[stock] = w / 100.0

                    total_custom_weight = sum(weights.values())
                    st.write(f"**Total Weight Assigned:** {total_custom_weight:.2f}")

                    if total_custom_weight != 1.0:
                        if total_custom_weight > 1.0:
                            st.error("Total weight exceeds 100%. Please adjust the weights so that they sum to 100%.")
                        else:
                            st.info("Total weight is less than 100%. Please adjust the weights so that they sum to 100%.")
                    else:
                        st.success("Total weight equals 100%. Your portfolio is ready!")
                else:
                    # Preview-only (disabled) inputs for free users
                    for stock in tickers:
                        st.number_input(
                            label=f"{stock} Weight (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=0.0,
                            step=5.0,
                            format="%.2f",
                            disabled=True,
                            help="Pro users can fine-tune allocations manually."
                        )

                    st.markdown("Unlock full control over your portfolio.")
                    st.markdown(
                        """
                        <a href="https://peakportfolio.ai/#Pricing" target="_blank" style="
                            text-decoration: none;
                            color: #0a4daa;
                            font-weight: 600;
                            font-size: 16px;
                            display: inline-flex;
                            align-items: center;
                            margin-left: 10px;
                        ">
                            🔓 Upgrade to Pro
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No assets selected. Please select tickers in the 'Asset Input' section to assign weights.")

            return weights, total_custom_weight


        # ---------------- Call inside sidebar ----------------
        with st.sidebar.expander("Enter The Details", expanded=False):
            custom_asset_weights, total_custom_weight = custom_asset_weights_fragment(tickers, product_id)










        # --- Create My Portfolio Button ---
        if st.sidebar.button("Create My Portfolio"):
            subtitle_placeholder.empty()
            particle_placeholder.empty()

            # Set AI state only now
            st.session_state.ai_context_input = ai_context_input
            st.session_state.ai_refinement_option = (
                "Yes" if ai_context_input.strip() else "No"
            )

            # Save other parameters
            st.session_state.parameters = {
                "tickers": tickers,
                "include_single_stock": include_single_stock,
                "start_date": start_date,
                "end_date": end_date,
                "annual_rfr": annual_rfr,
                "min_weight": min_weight,
                "max_weight": max_weight,
                "ai_context_input": ai_context_input,
            }



            # Fetch data (tickers is trimmed in‑place to the survivors)
            returns, ticker_info = fetch_and_resample_data(tickers, start_date, end_date)

            # Store for later use
            st.session_state.simulation_data["ticker_info"] = ticker_info

            # Lists that are now automatically aligned with the (mutated) tickers list
            company_names   = [info["name"]           for info in ticker_info]
            dividend_yields = [info["dividend_yield"] for info in ticker_info]








            # Fetch benchmark data
            benchmark_returns, benchmark_name = fetch_benchmark_data(benchmark, start_date, end_date)

            # Store benchmark data in session state
            st.session_state.simulation_data["benchmark_returns"] = benchmark_returns
            st.session_state.simulation_data["benchmark_name"] = benchmark_name









            # Compute Efficient Frontier
            efficient_frontier_data = efficient_frontier(
                mean_returns=returns.mean(),
                cov_matrix=returns.cov(),
                min_weight=min_weight,
                max_weight=max_weight,
                annual_rfr=annual_rfr,
                returns_df=returns,  # For Sortino ratio
                dividend_yields=dividend_yields,  # Forward dividend yields
            )












            # Always compute the single stock portfolio data with additional metrics
            computed_single_stock_data = single_stock_portfolio(
                mean_returns=returns.mean(),
                cov_matrix=returns.cov(),
                annual_rfr=annual_rfr,         # new parameter if needed
                returns_df=returns,            # new parameter if needed for downside deviation
                dividend_yields=pd.Series(dividend_yields, index=returns.columns)  # ensure correct indexing
            )

            # Decide what to pass based on the user selection
            if include_single_stock:
                single_stock_portfolio_data = computed_single_stock_data
            else:
                # Adjust fallback DataFrame to include the new columns if necessary
                single_stock_portfolio_data = pd.DataFrame(
                    columns=['Stock', 'Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Dividend Yield','Max Drawdown', 'Weights']
                )






            # Custom Portfolio
            if custom_asset_weights and total_custom_weight == 1.0:
                try:
                    # Ensure inputs are properly formatted for the backend
                    mean_returns_series = pd.Series(returns.mean(), index=returns.columns)  # Ensure pandas.Series
                    dividend_yields_series = pd.Series(dividend_yields, index=returns.columns)  # Ensure pandas.Series

                    # Call backend function
                    custom_portfolio_data = calculate_custom_portfolio_data(
                        custom_asset_weights=custom_asset_weights,
                        mean_returns=mean_returns_series,
                        cov_matrix=returns.cov(),
                        annual_rfr=annual_rfr,
                        returns_df=returns,  # Historical returns for Sortino
                        dividend_yields=dividend_yields_series,  # Forward dividend yields
                    )

                except Exception as e:
                    st.error(f"Error calculating custom portfolio data: {e}")
                    custom_portfolio_data = None
            else:
                custom_portfolio_data = None






            # Dividend-Focused Portfolio
            try:
                mean_returns_series = pd.Series(returns.mean(), index=returns.columns)
                dividend_yields_series = pd.Series(dividend_yields, index=returns.columns)

                dividend_portfolio_data = dividend_focused_portfolio(
                    mean_returns=mean_returns_series,
                    cov_matrix=returns.cov(),
                    min_weight=min_weight,
                    max_weight=max_weight,
                    annual_rfr=annual_rfr,
                    returns_df=returns,
                    dividend_yields=dividend_yields_series,
                )
            except Exception as e:
                st.error(f"Error calculating dividend-focused portfolio data: {e}")
                dividend_portfolio_data = None




            # Identify the indices of special portfolios
            max_sharpe_idx = np.argmax(efficient_frontier_data["sharpe_ratios"])
            max_return_idx = np.argmax(efficient_frontier_data["returns"])
            min_volatility_idx = np.argmin(efficient_frontier_data["volatilities"])




            # Extract metrics for the Max Sharpe portfolio         
            max_sharpe_weights = efficient_frontier_data["weights"][max_sharpe_idx]
            max_sharpe_return = efficient_frontier_data["returns"][max_sharpe_idx]
            max_sharpe_volatility = efficient_frontier_data["volatilities"][max_sharpe_idx]
            max_sharpe_ratio = efficient_frontier_data["sharpe_ratios"][max_sharpe_idx]
            max_sharpe_sortino = efficient_frontier_data["sortino_ratios"][max_sharpe_idx]
            max_sharpe_yield = efficient_frontier_data["dividend_yields"][max_sharpe_idx]
            max_sharpe_max_drawdown = efficient_frontier_data["max_drawdown"][max_sharpe_idx] 


            # Extract metrics for the Max Return portfolio
            max_return_weights = efficient_frontier_data["weights"][max_return_idx]
            max_return_return = efficient_frontier_data["returns"][max_return_idx]
            max_return_volatility = efficient_frontier_data["volatilities"][max_return_idx]
            max_return_ratio = efficient_frontier_data["sharpe_ratios"][max_return_idx]
            max_return_sortino = efficient_frontier_data["sortino_ratios"][max_return_idx]
            max_return_yield = efficient_frontier_data["dividend_yields"][max_return_idx]
            max_return_max_drawdown = efficient_frontier_data["max_drawdown"][max_return_idx]  


            # Extract metrics for the Min Volatility portfolio
            min_volatility_weights = efficient_frontier_data["weights"][min_volatility_idx]
            min_volatility_return = efficient_frontier_data["returns"][min_volatility_idx]
            min_volatility_volatility = efficient_frontier_data["volatilities"][min_volatility_idx]
            min_volatility_ratio = efficient_frontier_data["sharpe_ratios"][min_volatility_idx]
            min_volatility_sortino = efficient_frontier_data["sortino_ratios"][min_volatility_idx]
            min_volatility_yield = efficient_frontier_data["dividend_yields"][min_volatility_idx]
            min_volatility_max_drawdown = efficient_frontier_data["max_drawdown"][min_volatility_idx]



            # Convert weights from np.ndarray to Python lists
            ef_weights = [w.tolist() for w in efficient_frontier_data["weights"]]


            # Extract company names and forward dividend yields from ticker_info
            company_names = [info["name"] for info in ticker_info]
            dividend_yields = [info["dividend_yield"] for info in ticker_info]





            # --- time seres returns (0-based like your chart) ----------------------------
            conservative_r = portfolio_returns(returns, min_volatility_weights)
            balanced_r     = portfolio_returns(returns, max_sharpe_weights)
            aggressive_r   = portfolio_returns(returns, max_return_weights)

            if isinstance(benchmark_returns, pd.DataFrame):
                benchmark_r = benchmark_returns.iloc[:, 0]      # first (only) column → Series
            else:
                benchmark_r = benchmark_returns                 # already a Series

            benchmark_r = benchmark_r.rename("benchmark")

            portfolio_returns_df = pd.concat(
                [
                    conservative_r.rename("conservative"),
                    balanced_r.rename("balanced"),
                    aggressive_r.rename("aggressive"),
                    benchmark_r,
                ],
                axis=1,
            )


            # --- cumulative time series returns (0-based like your chart) ----------------------------
            conservative_cum = cumulative_growth(conservative_r)   # Series
            balanced_cum     = cumulative_growth(balanced_r)
            aggressive_cum   = cumulative_growth(aggressive_r)
            benchmark_cum    = cumulative_growth(benchmark_r)

            cumulative_returns_df = pd.concat(
                [
                    conservative_cum.rename("conservative"),
                    balanced_cum.rename("balanced"),
                    aggressive_cum.rename("aggressive"),
                    benchmark_cum.rename("benchmark"),
                ],
                axis=1,
            )



















            st.session_state.simulation_data = {
                "account_size": account_size,
                "tickers": tickers,

                # ↓ NEW: portfolio-level + benchmark return series
                "portfolio_returns_df": portfolio_returns_df,
                "cumulative_returns_df": cumulative_returns_df,  

                "returns": returns,  # DataFrame, not string
                "Annual Risk Free Rate": annual_rfr,
                "company_names": company_names,
                "dividend_yields": dividend_yields,
                "benchmark_returns": benchmark_returns,
                "benchmark_name": benchmark_name,
                "ai_context_input": ai_context_input,

                "efficient_frontier_data": {
                    "returns": efficient_frontier_data["returns"],
                    "volatilities": efficient_frontier_data["volatilities"],
                    "sharpe_ratios": efficient_frontier_data["sharpe_ratios"],
                    "sortino_ratios": efficient_frontier_data["sortino_ratios"],
                    "dividend_yields": efficient_frontier_data["dividend_yields"],
                    "weights": ef_weights
                },
                "single_stock_portfolio_data": single_stock_portfolio_data,
                "computed_single_stock_data": computed_single_stock_data,  # Store the computed data for single stock portfolio-


                # Metrics for Max Sharpe portfolio
                "max_sharpe_idx": max_sharpe_idx,
                "max_sharpe_return": max_sharpe_return,
                "max_sharpe_volatility": max_sharpe_volatility,
                "max_sharpe_ratio": max_sharpe_ratio,
                "max_sharpe_sortino": max_sharpe_sortino,
                "max_sharpe_yield": max_sharpe_yield,
                "max_sharpe_max_drawdown": max_sharpe_max_drawdown,
                "max_sharpe_weights": max_sharpe_weights.tolist() if isinstance(max_sharpe_weights, np.ndarray) else max_sharpe_weights,

                # Metrics for Max Return portfolio
                "max_return_idx": max_return_idx,
                "max_return_return": max_return_return,
                "max_return_volatility": max_return_volatility,
                "max_return_ratio": max_return_ratio,
                "max_return_sortino": max_return_sortino,
                "max_return_yield": max_return_yield,
                "max_return_max_drawdown": max_return_max_drawdown,
                "max_return_weights": max_return_weights.tolist() if isinstance(max_return_weights, np.ndarray) else max_return_weights,

                # Metrics for Min Volatility portfolio
                "min_volatility_idx": min_volatility_idx,
                "min_volatility_return": min_volatility_return,
                "min_volatility_volatility": min_volatility_volatility,
                "min_volatility_ratio": min_volatility_ratio,
                "min_volatility_sortino": min_volatility_sortino,
                "min_volatility_yield": min_volatility_yield,
                "min_volatility_max_drawdown": min_volatility_max_drawdown,
                "min_volatility_weights": min_volatility_weights.tolist() if isinstance(min_volatility_weights, np.ndarray) else min_volatility_weights,

                # Dividend and Custom portfolios
                "dividend_portfolio_data": dividend_portfolio_data,
                "custom_portfolio_data": custom_portfolio_data,

                # Placeholder DataFrames for future use
                "max_sharpe_allocation_df": None,
                "max_sharpe_summary_df": None,
                "max_return_allocation_df": None,
                "max_return_summary_df": None,
                "min_volatility_allocation_df": None,
                "min_volatility_summary_df": None,
                "custom_allocation_df": None,
                "top_5_allocations_df": None,
                "custom_summary_df": None
            }
            st.session_state.simulation_data["ticker_info"] = ticker_info
            st.session_state.simulation_ran = True


        if st.session_state.get("ai_refinement_option") != "Yes":
            for key in ["ai_refined_weights", "ai_refined_explanations", "ai_refined_portfolios", "ai_portfolio_metrics"]:
                st.session_state.simulation_data.pop(key, None)











        # --- Display Results if Simulation Ran ---
        if st.session_state.get("simulation_ran", False):
            parameters = st.session_state.parameters
            simulation_data = st.session_state.simulation_data

            # Extract variables
            tickers = parameters["tickers"]
            company_names = simulation_data["company_names"]
            efficient_frontier_data = simulation_data["efficient_frontier_data"]
            single_stock_portfolio_data = simulation_data["single_stock_portfolio_data"]
            computed_single_stock_data = simulation_data["computed_single_stock_data"]
            custom_portfolio_data = simulation_data.get("custom_portfolio_data")
            dividend_portfolio_data = simulation_data.get("dividend_portfolio_data")
            annual_rfr = parameters["annual_rfr"]
            ticker_info = simulation_data.get("ticker_info", [])


            st.session_state.pop('ai_insights', None)
            # Now, if AI refinement is enabled, run it:








            ai_status = st.empty()                           # banner placeholder

            # ------------------------------------------------------------------
            # AI refinement
            # ------------------------------------------------------------------
            if st.session_state.get("ai_refinement_option") == "Yes":
                done = ai_portfolio_refinement()  # True once weights final
                # ───────────────────── running – live 5-s status ───────────────
                if not done:

                    @st.fragment                             # Streamlit ≥ 1.34
                    def _ai_status_fragment() -> None:
                        sim  = st.session_state.get("simulation_data", {})
                        logs = sim.get("ai_logs", [])
                        msg  = logs[-1]["msg"] if logs else "multi agent portfolio construction in progress…"
                        ai_status.info(msg)

                        # force immediate rerun once the AI thread finishes
                        if sim.get("ai_done", False):
                            st.rerun()

                        # normal 5-second poll while waiting
                        st_autorefresh(interval=5_000, key="ai_autorefresh_inner")

                    _ai_status_fragment() 
                # ───────────────────── finished branch – final page state ──────
                else:
                    ai_status.empty()




























            # --- Efficient Frontier Visualization ---
            st.markdown('<h2 style=>Risk Reward Map</h2>', unsafe_allow_html=True)

            # If you have 'dividend_portfolio_data' stored somewhere (e.g., session_state),
            # simply pass it into 'plot_efficient_frontier' as below. 
            # Otherwise, remove the parameter if you don't have the variable available yet.
            fig_frontier = plot_efficient_frontier(
                efficient_results=efficient_frontier_data,
                max_sharpe_idx=simulation_data["max_sharpe_idx"],
                max_return_idx=simulation_data["max_return_idx"],
                min_volatility_idx=simulation_data["min_volatility_idx"],
                tickers=tickers,
                single_stock_portfolio_data=single_stock_portfolio_data,
                annual_rfr=annual_rfr,
                custom_portfolio_data=custom_portfolio_data,
                dividend_portfolio_data=dividend_portfolio_data 
            )


            buffer = BytesIO()
            fig_frontier.savefig(buffer, format="png", bbox_inches='tight', pad_inches=0.0)
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.read()).decode()
            buffer.close()

            st.markdown(f"""
            <table class="styled-frontier"; border-collapse: collapse; width: 100%; text-align: center;">
                <tr>
                    <td colspan="2" style="padding: 5px; text-align: center;">
                        <img src="data:image/png;base64,{chart_base64}" 
                            alt="Efficient Frontier Chart" 
                            style="max-width: 100%; height: auto;">
                    </td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

            sim = st.session_state.get("simulation_data", {})
            ai_done = sim.get("ai_done", False)

            legend_symbols = {
                "Efficient Frontier": '<svg width="100" height="10" xmlns="http://www.w3.org/2000/svg">'
                                    '<line x1="0" y1="5" x2="100" y2="5" stroke="blue" stroke-width="4" stroke-dasharray="6,4" />'
                                    '</svg>',
                "Conservative": '<svg width="20" height="20" xmlns="http://www.w3.org/2000/svg">'
                                        '<polygon points="10,1 13,7 19,7 14,11 16,17 10,13 4,17 6,11 1,7 7,7" fill="#18af28" />'
                                        '</svg>',
                ("Personalized" if ai_done else "Balanced"): '<svg width="20" height="20" xmlns="http://www.w3.org/2000/svg">'
                                                                                '<polygon points="10,1 13,7 19,7 14,11 16,17 10,13 4,17 6,11 1,7 7,7" fill="#f5cd05" />'
                                                                                '</svg>',
                "Aggressive": '<svg width="20" height="20" xmlns="http://www.w3.org/2000/svg">'
                                        '<polygon points="10,1 13,7 19,7 14,11 16,17 10,13 4,17 6,11 1,7 7,7" fill="#8321ff" />'
                                        '</svg>',
                "Dividend": '<svg width="20" height="20" xmlns="http://www.w3.org/2000/svg">'
                                    '<polygon points="10,1 13,7 19,7 14,11 16,17 10,13 4,17 6,11 1,7 7,7" fill="#f697ff" />'
                                    '</svg>',
                "Custom": '<svg width="20" height="20" xmlns="http://www.w3.org/2000/svg"><polygon points="10,1 13,7 19,7 14,11 16,17 10,13 4,17 6,11 1,7 7,7" fill="red" /></svg>',
                                                   

            }



            data = {
                "Portfolio": list(legend_symbols.keys()),
                "Symbol": list(legend_symbols.values())
            }
            table_df = pd.DataFrame(data)

            st.markdown(
                table_df.to_html(index=False, classes="styled-table", escape=False, border=1),
                unsafe_allow_html=True
            )

            with st.expander("Risk Reward Map Overview"):
                st.markdown("""
                <div style="font-size: 16px; line-height: 1.8; text-align: justify;">
                    <p>The Risk Reward Map helps you find the best balance between risk and return when investing. It shows which portfolios give you the highest possible returns for the risk you’re willing to take.</p>
                    </p>
                </div>
                """, unsafe_allow_html=True)

























































            # --- AI Insights Header ---
            st.markdown('<h2 style="margin-top: 20px;">AI Powered Portfolio Insights</h2>', unsafe_allow_html=True)

            @st.fragment
            def ai_chat(product_id, tickers):
                if st.session_state.get("simulation_ran", False):
                    if is_pro(product_id):
                        user_question = st.chat_input("Ask a question about your portfolio:")
                    else:
                        st.chat_input(
                            "🔒 Upgrade to Pro to ask questions about your portfolio.",
                            disabled=True
                        )
                        user_question = None

                    if user_question:
                        with st.spinner("Generating insights..."):
                            try:
                                prepared_data = prepare_simulation_data_for_prompt(st.session_state.simulation_data)
                                news_context = fetch_recent_events(tickers)
                                recent_events_summary = news_context.get("news_summary", "No recent events available.")

                                response = query_gpt(
                                    user_question=user_question,
                                    prepared_data=prepared_data,
                                    recent_events_summary=recent_events_summary,
                                )

                                st.markdown(f"""
                                    <div class="custom-text-box">
                                        <div class="custom-chat-message">{response}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                            except Exception as e:
                                st.markdown(f"""
                                    <div class="custom-text-box">
                                        <div class="custom-chat-message">❌ Failed to generate a response: {e}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("Please run the simulation first to generate insights.")

            # --- Call the fragment ---
            ai_chat(product_id, tickers)






















            st.markdown(
                """
                <style>
                /* Styling the tabs */
                div[data-testid="stTabs"] button {
                    font-size: 1.1em;
                    font-weight: bold;
                    color: #FFFFFF !important;
                    background-color: #007BFF;
                    border: 1px solid transparent;
                    border-radius: 5px;
                    padding: 8px 16px;
                    margin: 5px;
                    white-space: nowrap; /* Ensures text does not wrap */
                    transition: background-color 0.2s ease; /* Smooth transition for background color */
                }
                
                /* Selected tab style */
                div[data-testid="stTabs"] button[aria-selected="true"] {
                    background-color: #0a4daa !important;
                    color: #FFFFFF !important;
                }
                
                /* Hover, focus, and active states for immediate color change */
                div[data-testid="stTabs"] button:hover,
                div[data-testid="stTabs"] button:focus,
                div[data-testid="stTabs"] button:active {
                    background-color: #0a4daa !important;
                    color: #FFFFFF !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Portfolio Tabs with Dynamic "Custom" Tab Name
            st.markdown('<h2>Portfolio Breakdowns</h2>', unsafe_allow_html=True)

            # 1) Tabs creation – replace the True-branch label
            # 1) Tabs creation – replace the True-branch label
            portfolio_tabs = st.tabs(
                ["Personalized", "Conservative", "Aggressive", "Dividend Focused", "Custom"]
                if st.session_state.get("simulation_data", {}).get("ai_done", False)
                else
                ["Conservative", "Balanced", "Aggressive", "Dividend Focused", "Custom"]
            )



            simulation_data = st.session_state.simulation_data
            # Pull the stored ticker_info here
            ticker_info = simulation_data["ticker_info"]



            # Low Volatility Tab  (Conservative Portfolio – MVO only)
            with portfolio_tabs[
                1 if st.session_state.get("simulation_data", {}).get("ai_done", False) else 0
            ]:
                st.markdown(
                    '<h2>Conservative Portfolio</h2>',
                    unsafe_allow_html=True,
                )


                simulation_data = st.session_state.simulation_data

                # --- always MVO weights ---
                min_volatility_weights    = simulation_data["min_volatility_weights"]
                min_volatility_return     = simulation_data["min_volatility_return"]
                min_volatility_ratio      = simulation_data["min_volatility_ratio"]
                min_volatility_sortino    = simulation_data["min_volatility_sortino"]
                min_volatility_volatility = simulation_data["min_volatility_volatility"]
                min_volatility_yield      = simulation_data["min_volatility_yield"]
                min_volatility_max_drawdown = simulation_data["min_volatility_max_drawdown"]

                df_min_vol = pd.DataFrame({
                    "Ticker":       tickers,
                    "Asset Name":   company_names,
                    "Asset Type":   [info.get("type", "Unknown") for info in ticker_info],
                    "$ Allocation": [f"{round(w * account_size):,}" for w in min_volatility_weights],
                    "% Allocation": min_volatility_weights,
                })

                threshold   = 1e-5
                df_min_vol  = df_min_vol[df_min_vol["% Allocation"] >= threshold]
                df_min_vol  = df_min_vol.sort_values(by="% Allocation", ascending=False)
                df_min_vol["% Allocation"] = df_min_vol["% Allocation"].apply(lambda w: f"{w * 100:.2f}%")

                table_html = df_min_vol.to_html(index=False, classes="styled-table", escape=False, border=1)
                st.markdown(f'{table_html}</div>', unsafe_allow_html=True)

                # Simple explanatory banner (no AI explanations)
                fallback_html = (
                    '<div style="border: 1px solid #0a4daa; border-radius: 8px;'
                    'box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 10px; margin-bottom: 60px">'
                    '<p style="margin: 8px 0;">This portfolio is produced purely from mean-variance optimization (MVO). '
                    'AI refinement is not applied here.</p>'
                    '</div>'
                )
                st.markdown(fallback_html, unsafe_allow_html=True)

                # --- donut chart ---
                st.markdown(
                    '<h2>Allocation Snapshot</h2>',
                    unsafe_allow_html=True
                )

                fig_donut_min_vol = plot_min_volatility_allocation(tickers, min_volatility_weights)
                buffer_min_vol = BytesIO()
                fig_donut_min_vol.savefig(buffer_min_vol, format="png", bbox_inches="tight")
                buffer_min_vol.seek(0)
                chart_image_min_vol = base64.b64encode(buffer_min_vol.read()).decode()
                buffer_min_vol.close()

                st.markdown(f"""
                    <style>
                        .chart-container {{
                            max-width: 600px;
                            width: 100%;
                            margin: auto;
                            margin-bottom: 30px;
                        }}
                        .styled-table.non-scrollable tr:hover {{
                            background: none !important;
                        }}
                    </style>
                    <div class="chart-container">
                        <table class="styled-table non-scrollable">
                            <tr>
                                <td>
                                    <img src="data:image/png;base64,{chart_image_min_vol}" alt="Conservative Allocation Chart"/>
                                </td>
                            </tr>
                        </table>
                    </div>
                """, unsafe_allow_html=True)


                # --- summary table ---
                summary_labels  = ["Expected Return", "Sharpe Ratio", "Sortino Ratio", "Volatility", "Dividend Yield", "Max Drawdown"]
                summary_values  = [
                    f"{min_volatility_return:.2%}",
                    f"{min_volatility_ratio:.2f}",
                    f"{min_volatility_sortino:.2f}",
                    f"{min_volatility_volatility:.2%}",
                    f"{min_volatility_yield:.2%}",
                    f"{min_volatility_max_drawdown:.2%}"
                ]
                min_volatility_summary_df = pd.DataFrame({
                    "Portfolio Summary": summary_labels,
                    "Results": summary_values
                })
                st.markdown(
                    min_volatility_summary_df.to_html(index=False, classes="styled-table", border=1),
                    unsafe_allow_html=True
                )

                # cache in session_state
                simulation_data["min_volatility_allocation_df"] = df_min_vol
                simulation_data["min_volatility_summary_df"]    = min_volatility_summary_df







                st.markdown(
                    '<h2 style="margin-top: 60px; margin-bottom: 0px;">Conservative Portfolio vs. Benchmark</h2>',
                    unsafe_allow_html=True
                )

                def conservative_line_chart(figsize=(12, 8), scale=1):
                    sim = st.session_state.simulation_data

                    port_r  = sim["returns"].dot(sim["min_volatility_weights"])
                    bench_r = sim["benchmark_returns"]

                    port_c  = cumulative_growth(port_r)
                    bench_c = cumulative_growth(bench_r)

                    styles = {
                        "portfolio": dict(line="green",    dot="green",
                                        box="#18af28",   box_alpha=0.9, dot_size=35),
                        "benchmark": dict(line="#448dea",  dot="#0068ff",
                                        box="#016aab",   box_alpha=0.9, dot_size=25)
                    }

                    fig = plot_portfolio_vs_benchmark(
                        port_c, bench_c,
                        labels   = {"portfolio": "Conservative Portfolio",
                                    "benchmark": sim["benchmark_name"]},
                        styles   = styles,
                        figsize  = figsize,
                        scale    = scale
                    )
                    return fig








                def get_base64_image(image_path):
                    """Converts an image file (PNG) to a base64 encoded data URI."""
                    with open(image_path, "rb") as img_file:
                        encoded = base64.b64encode(img_file.read()).decode("utf-8")
                    return f"data:image/png;base64,{encoded}"


                def generate_html_report(line_chart_url, allocation_df, summary_df, logo_data_uri):
                    """Generates an HTML report using the provided data."""
                    env = Environment(loader=FileSystemLoader("main/templates"))
                    template = env.get_template("auto_template.html")
                    
                    context = {
                        "allocation_table": allocation_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                        "summary_table": summary_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                        "line_chart_url": line_chart_url,
                        "logo_data_uri": logo_data_uri,
                    }
                    
                    rendered_html = template.render(context)
                    output_path = "report_preview.html"
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(rendered_html)
                        
                    return output_path
                def fig_to_base64(fig):
                    """Converts a matplotlib figure to a base64 encoded PNG image string."""
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.2)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                    buffer.close()
                    return image_base64

                # --- Web App Section ---




                # Create and display the web app line chart
                fig_web = conservative_line_chart(figsize=(12, 8))
                line_chart_base64_web = fig_to_base64(fig_web)

                st.markdown(
                    f"""
                    <style>
                        .styled-table.non-scrollable tr:hover {{
                            background: none !important;
                        }}
                    </style>
                    <table class="styled-table non-scrollable" style="margin: 20px 0; background-color: #fff; width: 100%; text-align: center;">
                        <tr>
                            <td style="padding: 10px; text-align: center;">
                                <img src="data:image/png;base64,{line_chart_base64_web}" alt="Cumulative Returns" style="width:100%; height:auto;"/>
                            </td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown('<h2 style="margin-top: 60px; margin-bottom: 10px;">Portfolio Report</h2>', unsafe_allow_html=True)



                # Create the report line chart
                fig_report = conservative_line_chart(figsize=(12, 6), scale=1)
                line_chart_base64_report = fig_to_base64(fig_report)
                line_chart_url = f"data:image/png;base64,{line_chart_base64_report}"


                # Load the PNG logo from a relative path
                logo_data_uri = get_base64_image("logo.png")

                # Use only the top 5 allocations for the report
                top_5_allocations_df = st.session_state.simulation_data["min_volatility_allocation_df"].head(5)

                # Generate and display the HTML report
                report_html_path = generate_html_report(
                    line_chart_url=line_chart_url,
                    allocation_df=top_5_allocations_df,
                    summary_df=st.session_state.simulation_data["min_volatility_summary_df"],
                    logo_data_uri=logo_data_uri,
                )

                with open(report_html_path, "r", encoding="utf-8") as file:
                    report_html = file.read()

                st.components.v1.html(report_html, height=1000, scrolling=True)























            # 2) Header for the AI tab – replace “Balanced Portfolio” with “Personalized Portfolio”
            with portfolio_tabs[
                0 if st.session_state.get("simulation_data", {}).get("ai_done", False) else 1
            ]:
                st.markdown(
                    '<h2>{}</h2>'.format(
                        "Personalized Portfolio"
                        if st.session_state.get("simulation_data", {}).get("ai_done", False)
                        else "Balanced Portfolio"
                    ),
                    unsafe_allow_html=True,
                )

                simulation_data = st.session_state.simulation_data  # Use the nested simulation_data
                if st.session_state.get("ai_refinement_option") == "Yes" and simulation_data.get("ai_refined_weights"):

                    # Use AI-refined data for the Balanced portfolio
                    ai_weights = simulation_data["ai_refined_weights"]["balanced"]
                    balanced_metrics = simulation_data["ai_portfolio_metrics"]["balanced"]
                    max_sharpe_weights = ai_weights
                    max_sharpe_return = balanced_metrics["expected_return"]
                    max_sharpe_ratio = balanced_metrics["sharpe_ratio"]
                    max_sharpe_sortino = balanced_metrics["sortino_ratio"]
                    max_sharpe_volatility = balanced_metrics["volatility"]
                    max_sharpe_yield = balanced_metrics["dividend_yield"]
                    max_sharpe_max_drawdown = balanced_metrics.get("max_drawdown", None)
                    df_alloc = simulation_data["ai_refined_portfolios"]["balanced"]

                    # Format the AI-refined DataFrame the same as the fallback
                    threshold = 1e-5
                    df_alloc = df_alloc[df_alloc["% Allocation"] >= threshold]
                    df_alloc = df_alloc.sort_values(by="% Allocation", ascending=False)
                    df_alloc["% Allocation"] = df_alloc["% Allocation"].apply(
                        lambda x: f"{x * 100:.2f}%" if isinstance(x, float) else x
                    )
                else:
                    # Fallback to the original MVO data from simulation_data
                    max_sharpe_weights = simulation_data["max_sharpe_weights"]
                    max_sharpe_return = simulation_data["max_sharpe_return"]
                    max_sharpe_ratio = simulation_data["max_sharpe_ratio"]
                    max_sharpe_volatility = simulation_data["max_sharpe_volatility"]
                    max_sharpe_sortino = simulation_data.get("max_sharpe_sortino", None)
                    max_sharpe_yield = simulation_data.get("max_sharpe_yield", None)
                    max_sharpe_max_drawdown = simulation_data.get("max_sharpe_max_drawdown", None)

                    # Create the allocation DataFrame from fallback data
                    df_alloc = pd.DataFrame({
                        "Ticker": tickers,
                        "Asset Name": company_names,
                        "Asset Type": [info.get("type", "Unknown") for info in ticker_info],
                        "$ Allocation": [f"{round(weight * account_size):,}" for weight in max_sharpe_weights],
                        "% Allocation": max_sharpe_weights,
                    })

                    threshold = 1e-5
                    df_alloc = df_alloc[df_alloc["% Allocation"] >= threshold]
                    df_alloc = df_alloc.sort_values(by="% Allocation", ascending=False)
                    df_alloc["% Allocation"] = df_alloc["% Allocation"].apply(lambda x: f"{x * 100:.2f}%")

                # Generate the HTML table with styling and display it
                table_html = df_alloc.to_html(index=False, classes="styled-table", escape=False, border=1)
                container_html = f'{table_html}</div>'
                st.markdown(container_html, unsafe_allow_html=True)

                # Append the Why This Portfolio for the Balanced portfolio (if available)
                if st.session_state.get("ai_refinement_option") == "Yes" and simulation_data.get("ai_refined_explanations"):

                    balanced_explanations = simulation_data["ai_refined_explanations"].get("balanced", [])
                    if balanced_explanations:
                        st.markdown('<h2 style="">Why This Portfolio</h2>', unsafe_allow_html=True)
                        # Build the HTML content for available explanations
                        explanation_html = (
                            '<div style="border: 1px solid #0a4daa; border-radius: 8px;'
                            'box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 10px; margin-bottom: 40px;">'
                        )
                        for asset in sorted(balanced_explanations, key=lambda a: a.get("weight", 0), reverse=True):
                            ticker      = asset.get("ticker", "N/A")
                            weight      = asset.get("weight", "N/A")
                            explanation = asset.get("explanation", "N/A")
                            explanation_html += (
                                f'<p style="margin: 8px 0;"><strong>Ticker:</strong> {ticker} '
                                f'| <strong>Weight:</strong> {weight} '
                                f'| <strong>Explanation:</strong> {explanation}</p>'
                            )
                        explanation_html += "</div>"
                        st.markdown(explanation_html, unsafe_allow_html=True)
                    else:
                        # Styled fallback message when no explanations are available
                        fallback_html = (
                            '<div style="border: 1px solid #0a4daa; border-radius: 8px;'
                            'box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 10px; margin-bottom: 40px;">'
                            '<p style="margin: 8px 0;">This portfolio is based on mathematical optimization (MVO) without AI adjustments. While it follows a logical formula, it doesn’t account for market trends or risk balance—sometimes leading to unusual allocations. Upgrade for AI-powered insights and smarter portfolio refinements.</p>'
                            '</div>'
                        )
                        st.markdown(fallback_html, unsafe_allow_html=True)
                else:
                    # Styled fallback message when AI refinement is off or data is missing
                    fallback_html = (
                        '<div style="border: 1px solid #0a4daa; border-radius: 8px;'
                        'box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 10px; margin-bottom: 40px;">'
                        '<p style="margin: 8px 0;">This portfolio is based on mathematical optimization (MVO) without AI adjustments. While it follows a logical formula, it doesn’t account for market trends or risk balance—sometimes leading to unusual allocations. Upgrade for AI-powered insights and smarter portfolio refinements.</p>'
                        '</div>'
                    )
                    st.markdown(fallback_html, unsafe_allow_html=True)


                st.markdown(
                    '<h2>Allocation Snapshot</h2>',
                    unsafe_allow_html=True
                )

                fig_donut_sharpe = plot_max_sharpe_allocation(tickers, max_sharpe_weights)
                buffer_sharpe = BytesIO()
                fig_donut_sharpe.savefig(buffer_sharpe, format="png", bbox_inches="tight")
                buffer_sharpe.seek(0)
                chart_image_sharpe = base64.b64encode(buffer_sharpe.read()).decode()
                buffer_sharpe.close()

                st.markdown(f"""
                    <style>
                        .chart-container {{
                            max-width: 600px;
                            width: 100%;  /* Adjust this percentage to make it smaller */
                            margin: auto; /* Centers the chart */
                            margin-bottom: 30px;
                        }}
                        .styled-table.non-scrollable tr:hover {{
                            background: none !important;
                        }}
                    </style>
                    <div class="chart-container">
                        <table class="styled-table non-scrollable">
                            <td>
                                <img src="data:image/png;base64,{chart_image_sharpe}" 
                                    alt="Balanced Growth Allocation Chart"/>
                            </td>
                        </tr>
                    </table>
                """, unsafe_allow_html=True)


                # Build summary columns
                summary_labels = ["Expected Return", "Sharpe Ratio", "Sortino Ratio", "Volatility", "Dividend Yield", "Max Drawdown"]
                summary_values = [
                    f"{max_sharpe_return:.2%}",                     # Format as percentage
                    f"{max_sharpe_ratio:.2f}",                      # Format as float
                    f"{max_sharpe_sortino:.2f}",  # Handle missing Sortino
                    f"{max_sharpe_volatility:.2%}",                 # Format as percentage
                    f"{max_sharpe_yield:.2%}",     # Handle missing Yield
                    f"{max_sharpe_max_drawdown:.2%}",   # Handle missing Max Drawdown
                ]

                # Create summary DataFrame
                max_sharpe_summary_df = pd.DataFrame({
                    "Portfolio Summary": summary_labels,
                    "Results": summary_values
                })

                # Display summary table
                st.markdown(
                    max_sharpe_summary_df.to_html(index=False, classes="styled-table", escape=False, border=1),
                    unsafe_allow_html=True
                )


                # Store DataFrames in session state
                st.session_state.simulation_data["max_sharpe_allocation_df"] = df_alloc
                st.session_state.simulation_data["max_sharpe_summary_df"] = max_sharpe_summary_df









                # --- Helper Functions ---
                st.markdown(
                    f'<h2 style="margin-top: 60px;">{"Personalized Portfolio vs. Benchmark" if st.session_state.get("simulation_data", {}).get("ai_done", False) else "Balanced Portfolio vs. Benchmark"}</h2>',
                    unsafe_allow_html=True
                )

                def balanced_line_chart(figsize=(12, 8), scale=1):
                    sim     = st.session_state.simulation_data
                    ai_done = sim.get("ai_final_refresh_done", False)

                    # choose weights -------------------------------------------------
                    if (sim.get("ai_refinement_option", "Yes") == "Yes"
                            and "ai_refined_weights" in sim):
                        weights = sim["ai_refined_weights"]["balanced"]
                    else:
                        weights = sim["max_sharpe_weights"]

                    port_r  = sim["returns"].dot(weights)
                    bench_r = sim["benchmark_returns"]

                    port_c  = cumulative_growth(port_r)
                    bench_c = cumulative_growth(bench_r)

                    styles = {
                        "portfolio": dict(line="#f5cd05",  dot="#f1c900",
                                        box="#f5cd05",   box_alpha=0.9, dot_size=35),
                        "benchmark": dict(line="#448dea",  dot="#0068ff",
                                        box="#016aab",   box_alpha=0.9, dot_size=25)
                    }

                    fig = plot_portfolio_vs_benchmark(
                        port_c, bench_c,
                        labels   = {"portfolio": "Personalized Portfolio" if ai_done else "Balanced Portfolio",
                                    "benchmark": sim["benchmark_name"]},
                        styles   = styles,
                        figsize  = figsize,
                        scale    = scale
                    )
                    return fig









                def get_base64_image(image_path):
                    """
                    Converts an image file to a base64 encoded string.
                    """
                    with open(image_path, "rb") as img_file:
                        encoded = base64.b64encode(img_file.read()).decode("utf-8")
                    return f"data:image/png;base64,{encoded}"

                def generate_html_report(line_chart_url, allocation_df, summary_df, logo_data_uri):
                    """
                    Generates an HTML report using portfolio data.

                    Parameters:
                    - allocation_df: DataFrame with asset allocation details.
                    - summary_df: DataFrame with portfolio summary metrics.
                    - line_chart_url: URL or base64 string for the benchmark line chart.
                    - logo_data_uri: Base64 data URI for the logo.

                    Returns:
                    - output_path: The file path of the generated HTML report.
                    """
                    env = Environment(loader=FileSystemLoader("main/templates"))
                    template = env.get_template("auto_template.html")

                    context = {
                        "allocation_table": allocation_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                        "summary_table": summary_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                        "line_chart_url": line_chart_url,
                        "logo_data_uri": logo_data_uri,
                    }

                    rendered_html = template.render(context)
                    output_path = "report_preview.html"
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(rendered_html)

                    return output_path

                def fig_to_base64(fig):
                    """
                    Converts a matplotlib figure to a base64 encoded PNG image string.
                    """
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.2)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode()
                    buffer.close()
                    return image_base64

                # --- Web App Section ---

                # Create the line chart for the web app using a larger figure size (12, 8)
                fig_web = balanced_line_chart(figsize=(12, 8))
                line_chart_base64_web = fig_to_base64(fig_web)

                # Display the chart in Streamlit using markdown with an embedded image
                st.markdown(
                    f"""
                    <style>
                        .styled-table.non-scrollable tr:hover {{
                            background: none !important;
                        }}
                    </style>
                    <table class="styled-table non-scrollable" 
                        style="margin: 0px 0; background-color: #fff; 
                                width: 100%; text-align: center;">
                        <tr>
                            <td style="padding: 10px; text-align: center;">
                                <img src="data:image/png;base64,{line_chart_base64_web}" 
                                    alt="Cumulative Returns" 
                                    style="width:100%; height:auto;"/>
                            </td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )

                # --- Report Generation Section ---
                st.markdown('<h2 style="margin-top: 60px; margin-bottom: 10px;">Portfolio Report</h2>', unsafe_allow_html=True)

                # Create the line chart for the report using a larger figure size
                fig_report = balanced_line_chart(figsize=(12, 6), scale=1)
                line_chart_base64_report = fig_to_base64(fig_report)

                # Prepare the full data URL string for the line chart image for the report
                line_chart_url = f"data:image/png;base64,{line_chart_base64_report}"

                # Convert the logo PNG to a base64 data URI
                logo_data_uri = get_base64_image("logo.png")

                # Filter to include only the top 5 allocations by weight (assuming the data is already sorted)
                top_5_allocations_df = st.session_state.simulation_data["max_sharpe_allocation_df"].head(5)

                # Generate the report using the filtered simulation data
                report_html_path = generate_html_report(
                    line_chart_url=line_chart_url,
                    allocation_df=top_5_allocations_df,
                    summary_df=st.session_state.simulation_data["max_sharpe_summary_df"],
                    logo_data_uri=logo_data_uri,
                )

                # Read and display the generated HTML report in Streamlit
                with open(report_html_path, "r", encoding="utf-8") as file:
                    report_html = file.read()

                st.components.v1.html(report_html, height=1000, scrolling=True)


















            # Aggressive Growth Tab  (MVO only)
            with portfolio_tabs[2]:
                st.markdown(
                    '<h2>Aggressive Portfolio </h2>',
                    unsafe_allow_html=True
                )

                simulation_data = st.session_state.simulation_data

                # --- always MVO weights ---
                max_return_weights     = simulation_data["max_return_weights"]
                max_return_return      = simulation_data["max_return_return"]
                max_return_ratio       = simulation_data["max_return_ratio"]
                max_return_sortino     = simulation_data["max_return_sortino"]
                max_return_volatility  = simulation_data["max_return_volatility"]
                max_return_yield       = simulation_data["max_return_yield"]
                max_return_max_drawdown = simulation_data["max_return_max_drawdown"]

                # Build allocation DataFrame
                max_return_allocation_df = pd.DataFrame({
                    "Ticker":       tickers,
                    "Asset Name":   company_names,
                    "Asset Type":   [info.get("type", "Unknown") for info in ticker_info],
                    "$ Allocation": [f"{round(w * account_size):,}" for w in max_return_weights],
                    "% Allocation": max_return_weights,
                })

                threshold = 1e-5
                max_return_allocation_df = max_return_allocation_df[max_return_allocation_df["% Allocation"] >= threshold]
                max_return_allocation_df = max_return_allocation_df.sort_values(by="% Allocation", ascending=False)
                max_return_allocation_df["% Allocation"] = max_return_allocation_df["% Allocation"].apply(
                    lambda x: f"{x * 100:.2f}%"
                )

                # Display table
                table_html = max_return_allocation_df.to_html(index=False, classes="styled-table", escape=False, border=1)
                st.markdown(f'{table_html}</div>', unsafe_allow_html=True)

                # Static explanatory banner
                fallback_html = (
                    '<div style="border: 1px solid #0a4daa; border-radius: 8px;'
                    'box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 10px; margin-bottom: 40px;">'
                    '<p style="margin: 8px 0;">This portfolio is produced purely from mean-variance optimization (MVO). '
                    'AI refinement is not applied here.</p>'
                    '</div>'
                )
                st.markdown(fallback_html, unsafe_allow_html=True)

                # Allocation snapshot chart
                st.markdown(
                    '<h2>Allocation Snapshot</h2>',
                    unsafe_allow_html=True
                )

                fig_donut_return = plot_max_return_allocation(tickers, max_return_weights)
                buffer_return = BytesIO()
                fig_donut_return.savefig(buffer_return, format="png", bbox_inches="tight")
                buffer_return.seek(0)
                chart_image_return = base64.b64encode(buffer_return.read()).decode()
                buffer_return.close()

                st.markdown(f"""
                    <style>
                        .chart-container {{
                            max-width: 600px;
                            width: 100%;
                            margin: auto;
                            margin-bottom: 30px;
                        }}
                        .styled-table.non-scrollable tr:hover {{
                            background: none !important;
                        }}
                    </style>
                    <div class="chart-container">
                        <table class="styled-table non-scrollable">
                            <tr>
                                <td>
                                    <img src="data:image/png;base64,{chart_image_return}" 
                                        alt="Aggressive Allocation Chart"/>
                                </td>
                            </tr>
                        </table>
                    </div>
                """, unsafe_allow_html=True)

                # Summary metrics
                summary_labels  = ["Expected Return", "Sharpe Ratio", "Sortino Ratio", "Volatility", "Dividend Yield", "Max Drawdown"]
                summary_values  = [
                    f"{max_return_return:.2%}",
                    f"{max_return_ratio:.2f}",
                    f"{max_return_sortino:.2f}",
                    f"{max_return_volatility:.2%}",
                    f"{max_return_yield:.2%}",
                    f"{max_return_max_drawdown:.2%}"
                ]

                max_return_summary_df = pd.DataFrame({
                    "Portfolio Summary": summary_labels,
                    "Results": summary_values
                })
                st.markdown(
                    max_return_summary_df.to_html(index=False, classes="styled-table", escape=False, border=1),
                    unsafe_allow_html=True
                )

                # Cache DataFrames
                st.session_state.simulation_data["max_return_allocation_df"] = max_return_allocation_df
                st.session_state.simulation_data["max_return_summary_df"]    = max_return_summary_df







                def aggressive_growth_line_chart(figsize=(12, 8), scale=1):
                    sim      = st.session_state.simulation_data
                    port_r   = sim["returns"].dot(sim["max_return_weights"])
                    bench_r  = sim["benchmark_returns"]
                    port_c   = cumulative_growth(port_r)
                    bench_c  = cumulative_growth(bench_r)

                    styles = {
                        "portfolio": dict(line="#8321ff", dot="#8321ff", box="#8321ff",
                                        box_alpha=0.9, dot_size=35),
                        "benchmark": dict(line="#448dea", dot="#0068ff", box="#016aab",
                                        box_alpha=0.9, dot_size=25)
                    }

                    fig = plot_portfolio_vs_benchmark(
                        port_c, bench_c,
                        labels={"portfolio": "Aggressive Growth Portfolio",
                                "benchmark": sim["benchmark_name"]},
                        styles=styles,
                        figsize=figsize,
                        scale=scale,
                    )
                    return fig







                def get_base64_image(image_path):
                    """
                    Converts an image file to a base64 encoded string.
                    """
                    with open(image_path, "rb") as img_file:
                        encoded = base64.b64encode(img_file.read()).decode("utf-8")
                    return f"data:image/png;base64,{encoded}"

                def generate_html_report(line_chart_url, allocation_df, summary_df, logo_data_uri):
                    """
                    Generates an HTML report using portfolio data.

                    Parameters:
                    - allocation_df: DataFrame with asset allocation details.
                    - summary_df: DataFrame with portfolio summary metrics.
                    - line_chart_url: URL or base64 string for the benchmark line chart.
                    - logo_data_uri: Base64 data URI for the logo.

                    Returns:
                    - output_path: The file path of the generated HTML report.
                    """
                    env = Environment(loader=FileSystemLoader("main/templates"))
                    template = env.get_template("auto_template.html")

                    context = {
                        "allocation_table": allocation_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                        "summary_table": summary_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                        "line_chart_url": line_chart_url,
                        "logo_data_uri": logo_data_uri,
                    }

                    rendered_html = template.render(context)
                    output_path = "report_preview.html"
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(rendered_html)

                    return output_path

                def fig_to_base64(fig):
                    """
                    Converts a matplotlib figure to a base64 encoded PNG image string.
                    """
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.2)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode()
                    buffer.close()
                    return image_base64

                # --- Web App Section ---

                # Create the line chart for the web app using a larger figure size (12, 8)
                fig_web = aggressive_growth_line_chart(figsize=(12, 8))
                line_chart_base64_web = fig_to_base64(fig_web)

                # Display the chart in Streamlit using markdown with an embedded image
                st.markdown(
                    f"""
                    <style>
                        .styled-table.non-scrollable tr:hover {{
                            background: none !important;
                        }}
                    </style>
                    <table class="styled-table non-scrollable" 
                        style="margin: 0px 0; background-color: #fff; 
                                width: 100%; text-align: center;">
                        <tr>
                            <td style="padding: 10px; text-align: center;">
                                <img src="data:image/png;base64,{line_chart_base64_web}" 
                                    alt="Cumulative Returns" 
                                    style="width:100%; height:auto;"/>
                            </td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )
                # --- Report Generation Section ---
                st.markdown('<h2 style="margin-top: 60px; margin-bottom: 10px;">Portfolio Report</h2>', unsafe_allow_html=True)

                # Create the line chart for the report using a larger figure size
                fig_report = aggressive_growth_line_chart(figsize=(12, 6), scale=1)
                line_chart_base64_report = fig_to_base64(fig_report)

                # Prepare the full data URL string for the line chart image for the report
                line_chart_url = f"data:image/png;base64,{line_chart_base64_report}"

                # Convert the logo PNG to a base64 data URI
                logo_data_uri = get_base64_image("logo.png")

                # Filter to include only the top 5 allocations by weight (assuming the data is already sorted)
                top_5_allocations_df = st.session_state.simulation_data["max_return_allocation_df"].head(5)

                # Generate the report using the filtered simulation data
                report_html_path = generate_html_report(
                    line_chart_url=line_chart_url,
                    allocation_df=top_5_allocations_df,
                    summary_df=st.session_state.simulation_data["max_return_summary_df"],
                    logo_data_uri=logo_data_uri,
                )

                # Read and display the generated HTML report in Streamlit
                with open(report_html_path, "r", encoding="utf-8") as file:
                    report_html = file.read()

                st.components.v1.html(report_html, height=1000, scrolling=True)























            # Dividend-Focused Portfolio Tab
            with portfolio_tabs[3]:
                returns = simulation_data["returns"]                 
                ticker_info = simulation_data["ticker_info"]          
                company_names = simulation_data["company_names"]
                tickers = simulation_data["tickers"]
                dividend_portfolio_data = simulation_data["dividend_portfolio_data"]

                if dividend_portfolio_data:
                        st.markdown('<h2>Dividend-Focused Portfolio </h2>', unsafe_allow_html=True)

                    
                        # 1) Prepare portfolio allocation data
                        df_dividend_alloc = pd.DataFrame({
                            "Ticker": returns.columns,  # 'returns.columns' is now valid
                            "Asset Name": [company_names[tickers.index(tkr)] for tkr in returns.columns],
                            "Asset Type": [info.get("type", "Unknown") for info in ticker_info],
                            "$ Allocation": [f"{round(weight * account_size):,}" for weight in dividend_portfolio_data["d_weights"]],  # Round and format as currency
                            "% Allocation": dividend_portfolio_data["d_weights"],  # Keep numeric floats for weights
                        })

                        # 2) Filter by threshold
                        threshold = 1e-5
                        df_dividend_alloc = df_dividend_alloc[df_dividend_alloc["% Allocation"] >= threshold]

                        # 3) Sort allocations from greatest to least
                        df_dividend_alloc = df_dividend_alloc.sort_values(by="% Allocation", ascending=False)

                        # 4) Convert to percentage strings
                        df_dividend_alloc["% Allocation"] = df_dividend_alloc["% Allocation"].apply(lambda w: f"{w * 100:.2f}%")

                        # Generate HTML table with styling
                        table_html = df_dividend_alloc.to_html(index=False, classes="styled-table", escape=False, border=1)
                        container_html = f'{table_html}</div>'
                        st.markdown(container_html, unsafe_allow_html=True)


                        st.markdown(
                            '<h2>Allocation Snapshot</h2>',
                            unsafe_allow_html=True
                        )
                        fig_donut_dividend = plot_dividend_allocation(tickers, dividend_portfolio_data["d_weights"])

                        # Convert the chart to a base64 string for inline display
                        buffer_dividend = BytesIO()
                        fig_donut_dividend.savefig(buffer_dividend, format="png", bbox_inches="tight")
                        buffer_dividend.seek(0)
                        chart_image_dividend = base64.b64encode(buffer_dividend.read()).decode()
                        buffer_dividend.close()

                        st.markdown(f"""
                            <style>
                                .chart-container {{
                                    max-width: 600px;
                                    width: 100%;  /* Adjust this percentage to make it smaller */
                                    margin: auto; /* Centers the chart */
                                }}
                                .styled-table.non-scrollable tr:hover {{
                                    background: none !important;
                                }}
                            </style>
                            <div class="chart-container">
                                <table class="styled-table non-scrollable">
                                    <td>
                                        <img src="data:image/png;base64,{chart_image_dividend}" 
                                            alt="Dividend-Focused Portfolio Allocation Chart"/>
                                    </td>
                                </tr>
                            </table>
                        """, unsafe_allow_html=True)






                        # Retrieve the dividend portfolio’s metrics
                        d_return = dividend_portfolio_data["d_expected_return"]
                        d_sharpe = dividend_portfolio_data["d_sharpe_ratio"]
                        d_volatility = dividend_portfolio_data["d_volatility"]
                        d_sortino = dividend_portfolio_data.get("d_sortino_ratio", None)
                        d_yield = dividend_portfolio_data.get("d_dividend_yield", None)
                        d_max_drawdown = dividend_portfolio_data.get("d_max_drawdown", None)


                        # Create & display a summary table
                        summary_labels = ["Expected Return", "Sharpe Ratio", "Sortino Ratio", "Volatility", "Dividend Yield", "Max Drawdown"]
                        summary_values = [
                            f"{d_return:.2%}",
                            f"{d_sharpe:.2f}",
                            f"{d_sortino:.2f}",
                            f"{d_volatility:.2%}",
                            f"{d_yield:.2%}", 
                            f"{d_max_drawdown:.2%}",
                        ]

                        dividend_summary_df = pd.DataFrame({
                            "Portfolio Summary": summary_labels,
                            "Results": summary_values
                        })
                        st.markdown(
                            dividend_summary_df.to_html(index=False, classes="styled-table", escape=False, border=1),
                            unsafe_allow_html=True
                        )

                        # Store the results in session state
                        st.session_state.simulation_data["dividend_allocation_df"] = df_dividend_alloc
                        st.session_state.simulation_data["dividend_summary_df"] = dividend_summary_df






                
                def dividend_focused_line_chart(figsize=(12, 8), scale=1):
                    sim   = st.session_state.simulation_data
                    port_r  = sim["returns"].dot(
                                np.asarray(sim["dividend_portfolio_data"]["d_weights"]))
                    bench_r = sim["benchmark_returns"]
                    port_c  = cumulative_growth(port_r)
                    bench_c = cumulative_growth(bench_r)

                    styles = {
                        "portfolio": dict(line="#f697ff", dot="#f697ff", box="#f697ff",
                                        box_alpha=0.9, dot_size=35),
                        "benchmark": dict(line="#448dea", dot="#0068ff", box="#016aab",
                                        box_alpha=0.9, dot_size=25)
                    }

                    fig = plot_portfolio_vs_benchmark(
                        port_c, bench_c,
                        labels={"portfolio": "Dividend Focused Portfolio",
                                "benchmark": sim["benchmark_name"]},
                        styles=styles,
                        figsize=figsize,
                        scale=scale,
                    )
                    return fig





                def get_base64_image(image_path):
                    """
                    Converts an image file to a base64 encoded string.
                    """
                    with open(image_path, "rb") as img_file:
                        encoded = base64.b64encode(img_file.read()).decode("utf-8")
                    return f"data:image/png;base64,{encoded}"

                def generate_html_report(line_chart_url, allocation_df, summary_df, logo_data_uri):
                    """
                    Generates an HTML report using portfolio data.

                    Parameters:
                    - allocation_df: DataFrame with asset allocation details.
                    - summary_df: DataFrame with portfolio summary metrics.
                    - line_chart_url: URL or base64 string for the benchmark line chart.
                    - logo_data_uri: Base64 data URI for the logo.

                    Returns:
                    - output_path: The file path of the generated HTML report.
                    """
                    env = Environment(loader=FileSystemLoader("main/templates"))
                    template = env.get_template("auto_template.html")

                    context = {
                        "allocation_table": allocation_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                        "summary_table": summary_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                        "line_chart_url": line_chart_url,
                        "logo_data_uri": logo_data_uri,
                    }

                    rendered_html = template.render(context)
                    output_path = "report_preview.html"
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(rendered_html)

                    return output_path

                def fig_to_base64(fig):
                    """
                    Converts a matplotlib figure to a base64 encoded PNG image string.
                    """
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.2)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode()
                    buffer.close()
                    return image_base64

                # --- Web App Section ---

                # Create the line chart for the web app using a larger figure size (12, 8)
                fig_web = dividend_focused_line_chart(figsize=(12, 8))
                line_chart_base64_web = fig_to_base64(fig_web)

                # Display the chart in Streamlit using markdown with an embedded image
                st.markdown(
                    f"""
                    <style>
                        .styled-table.non-scrollable tr:hover {{
                            background: none !important;
                        }}
                    </style>
                    <table class="styled-table non-scrollable" 
                        style="margin: 0px 0; background-color: #fff; 
                                width: 100%; text-align: center;">
                        <tr>
                            <td style="padding: 10px; text-align: center;">
                                <img src="data:image/png;base64,{line_chart_base64_web}" 
                                    alt="Cumulative Returns" 
                                    style="width:100%; height:auto;"/>
                            </td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )

                # --- Report Generation Section ---
                st.markdown('<h2 style="margin-top: 60px; margin-bottom: 10px;">Portfolio Report</h2>', unsafe_allow_html=True)

                # Create the line chart for the report using a larger figure size
                fig_report = dividend_focused_line_chart(figsize=(12, 6), scale=1)
                line_chart_base64_report = fig_to_base64(fig_report)

                # Prepare the full data URL string for the line chart image for the report
                line_chart_url = f"data:image/png;base64,{line_chart_base64_report}"

                # Convert the logo PNG to a base64 data URI
                logo_data_uri = get_base64_image("logo.png")

                # Filter to include only the top 5 allocations by weight (assuming the data is already sorted)
                top_5_allocations_df = st.session_state.simulation_data["dividend_allocation_df"].head(5)

                # Generate the report using the filtered simulation data
                report_html_path = generate_html_report(
                    line_chart_url=line_chart_url,
                    allocation_df=top_5_allocations_df,
                    summary_df=st.session_state.simulation_data["dividend_summary_df"],
                    logo_data_uri=logo_data_uri,
                )

                # Read and display the generated HTML report in Streamlit
                with open(report_html_path, "r", encoding="utf-8") as file:
                    report_html = file.read()

                st.components.v1.html(report_html, height=1000, scrolling=True)























            # Custom Portfolio Tab
            with portfolio_tabs[4]:
                if custom_portfolio_data:
                        st.markdown("<h2>Custom Portfolio</h2>", unsafe_allow_html=True)

                        # 1) Keep numeric allocations
                        # Custom Portfolio

                        # 1) Prepare portfolio allocation data
                        df_custom_alloc = pd.DataFrame({
                            "Ticker": list(custom_asset_weights.keys()),
                            "Asset Name": [company_names[tickers.index(tkr)] for tkr in custom_asset_weights.keys()],
                            "Asset Type": [info.get("type", "Unknown") for info in ticker_info],  # Extract Asset Type
                            "$ Allocation": [f"{round(weight * account_size):,}" for weight in custom_asset_weights.values()],  # Round and format as currency
                            "% Allocation": list(custom_asset_weights.values()),  # Keep numeric floats for weights
                        })

                        # 2) Filter by threshold
                        threshold = 1e-5
                        df_custom_alloc = df_custom_alloc[df_custom_alloc["% Allocation"] >= threshold]

                        # 3) Sort allocations from greatest to least
                        df_custom_alloc = df_custom_alloc.sort_values(by="% Allocation", ascending=False)

                        # 4) Convert to percentage strings
                        df_custom_alloc["% Allocation"] = df_custom_alloc["% Allocation"].apply(lambda w: f"{w * 100:.2f}%")

                        # Generate HTML table with styling
                        table_html = df_custom_alloc.to_html(index=False, classes="styled-table", escape=False, border=1)
                        container_html = f'{table_html}</div>'
                        st.markdown(container_html, unsafe_allow_html=True)




                        st.markdown(
                            '<h2>Allocation Snapshot</h2>',
                            unsafe_allow_html=True
                        )
                        fig_donut_custom = plot_custom_portfolio_allocation(tickers, custom_asset_weights)
                        buffer_custom = BytesIO()
                        fig_donut_custom.savefig(buffer_custom, format="png", bbox_inches="tight")
                        buffer_custom.seek(0)
                        chart_image_custom = base64.b64encode(buffer_custom.read()).decode()
                        buffer_custom.close()

                        st.markdown(f"""
                            <style>
                                .chart-container {{
                                    max-width: 600px;
                                    width: 100%;  /* Adjust this percentage to make it smaller */
                                    margin: auto; /* Centers the chart */
                                    margin-bottom: 30px;
                                }}
                                .styled-table.non-scrollable tr:hover {{
                                    background: none !important;
                                }}
                            </style>
                            <div class="chart-container">
                                <table class="styled-table non-scrollable">
                                    <td>
                                        <img src="data:image/png;base64,{chart_image_custom}" 
                                            alt="Custom Portfolio Allocation Chart"/>
                                    </td>
                                </tr>
                            </table>
                        """, unsafe_allow_html=True)









                        # Retrieve the custom portfolio’s metrics
                        c_return = custom_portfolio_data["c_expected_return"]
                        c_sharpe = custom_portfolio_data["c_sharpe_ratio"]
                        c_volatility = custom_portfolio_data["c_portfolio_volatility"]
                        c_sortino = custom_portfolio_data.get("c_sortino_ratio", None)  # Ensure it exists
                        c_yield = custom_portfolio_data.get("c_dividend_yield", None)  # Ensure it exists
                        c_max_drawdown = custom_portfolio_data.get("c_max_drawdown", None)  # Ensure it exists



                        # Summary table: add "Sortino Ratio" and "Dividend Yield" if provided
                        summary_labels = ["Expected Return", "Sharpe Ratio", "Sortino Ratio", "Volatility", "Dividend Yield","Max Drawdown"]
                        summary_values = [
                            f"{c_return:.2%}",
                            f"{c_sharpe:.2f}",
                            f"{c_sortino:.2f}" if c_sortino is not None else "N/A",  # Handle missing Sortino
                            f"{c_volatility:.2%}",
                            f"{c_yield:.2%}" if c_yield is not None else "N/A",      # Handle missing Dividend Yield
                            f"{c_max_drawdown:.2%}" if c_max_drawdown is not None else "N/A"  # Handle missing Max Drawdown
                        ]

                        custom_summary_df = pd.DataFrame({
                            "Portfolio Summary": summary_labels,
                            "Results": summary_values
                        })
                        st.markdown(
                            custom_summary_df.to_html(index=False, classes="styled-table", escape=False, border=1),
                            unsafe_allow_html=True
                        )

                        # Store in session state
                        st.session_state.simulation_data["custom_allocation_df"] = df_custom_alloc
                        st.session_state.simulation_data["custom_summary_df"] = custom_summary_df




                        # --- Custom Portfolio Helper Function ---
                        st.markdown(
                            f'<h2 style="margin-top: 30px;Custom Portfolio vs. Benchmark</h2>',
                            unsafe_allow_html=True
                        )

                        def custom_portfolio_line_chart(
                                figsize,
                                custom_asset_weights: dict,
                                returns: pd.DataFrame,
                                benchmark_returns: pd.Series,
                                benchmark_name: str,
                                portfolio_display_name: str,
                                scale=1):

                            w = np.array([custom_asset_weights.get(tkr, 0) for tkr in returns.columns])
                            if np.allclose(w, 0):
                                st.warning("No custom portfolio weights set.")
                                return None

                            port_r  = returns.dot(w)
                            bench_r = benchmark_returns
                            port_c  = cumulative_growth(port_r)
                            bench_c = cumulative_growth(bench_r)

                            styles = {
                                "portfolio": dict(line="red", dot="red", box="red",
                                                box_alpha=0.9, dot_size=35),
                                "benchmark": dict(line="#448dea", dot="#0068ff", box="#016aab",
                                                box_alpha=0.9, dot_size=25)
                            }

                            fig = plot_portfolio_vs_benchmark(
                                port_c, bench_c,
                                labels={"portfolio": portfolio_display_name,
                                        "benchmark": benchmark_name},
                                styles=styles,
                                figsize=figsize,
                                scale=scale,
                            )
                            return fig


                        def get_base64_image(image_path):
                            """
                            Converts an image file to a base64 encoded string.
                            """
                            with open(image_path, "rb") as img_file:
                                encoded = base64.b64encode(img_file.read()).decode("utf-8")
                            return f"data:image/png;base64,{encoded}"

                        def generate_html_report(line_chart_url, allocation_df, summary_df, logo_data_uri):
                            """
                            Generates an HTML report using portfolio data.

                            Parameters:
                            - allocation_df: DataFrame with asset allocation details.
                            - summary_df: DataFrame with portfolio summary metrics.
                            - line_chart_url: URL or base64 string for the benchmark line chart.
                            - logo_data_uri: Base64 data URI for the logo.

                            Returns:
                            - output_path: The file path of the generated HTML report.
                            """
                            env = Environment(loader=FileSystemLoader("main/templates"))
                            template = env.get_template("auto_template.html")

                            context = {
                                "allocation_table": allocation_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                                "summary_table": summary_df.to_html(classes="styled-table", index=False, border=1, escape=False),
                                "line_chart_url": line_chart_url,
                                "logo_data_uri": logo_data_uri,
                            }

                            rendered_html = template.render(context)
                            output_path = "report_preview.html"
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(rendered_html)

                            return output_path

                        def fig_to_base64(fig):
                            """
                            Converts a matplotlib figure to a base64 encoded PNG image string.
                            """
                            buffer = BytesIO()
                            fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.2)
                            buffer.seek(0)
                            image_base64 = base64.b64encode(buffer.read()).decode()
                            buffer.close()
                            return image_base64

                        # --- Web App Section ---

                        if custom_asset_weights:
                            # Create the line chart for the web app using a larger figure size (12, 8)
                            fig_web_custom = custom_portfolio_line_chart(
                                figsize=(12, 8),
                                custom_asset_weights=custom_asset_weights,
                                returns=returns,
                                benchmark_returns=st.session_state.simulation_data.get("benchmark_returns"),
                                benchmark_name=st.session_state.simulation_data.get("benchmark_name"),
                                portfolio_display_name="Custom Portfolio",
                            )
                            if fig_web_custom is not None:
                                line_chart_base64_custom_web = fig_to_base64(fig_web_custom)

                                # Display the chart in Streamlit using an embedded image
                                st.markdown(
                                    f"""
                                    <style>
                                        .styled-table.non-scrollable tr:hover {{
                                            background: none !important;
                                        }}
                                    </style>
                                    <table class="styled-table non-scrollable" 
                                        style="margin: 0px 0; background-color: #fff; 
                                                width: 100%; text-align: center;">
                                        <tr>
                                            <td style="padding: 10px; text-align: center;">
                                                <img src="data:image/png;base64,{line_chart_base64_custom_web}" 
                                                    alt="Cumulative Returns" 
                                                    style="width:100%; height:auto;"/>
                                            </td>
                                        </tr>
                                    </table>
                                    """,
                                    unsafe_allow_html=True
                                )


                                # --- Report Generation Section ---
                                st.markdown('<h2 style="margin-top: 60px; margin-bottom: 10px;">Portfolio Report</h2>', unsafe_allow_html=True)

                                # Create the line chart for the report using a larger figure size and an appropriate scale factor.
                                fig_report_custom = custom_portfolio_line_chart(
                                    figsize=(12, 6),
                                    custom_asset_weights=custom_asset_weights,
                                    returns=returns,
                                    benchmark_returns=st.session_state.simulation_data.get("benchmark_returns"),
                                    benchmark_name=st.session_state.simulation_data.get("benchmark_name"),
                                    portfolio_display_name= "Custom Portfolio",
                                    scale=1
                                )

                                if fig_report_custom is not None:
                                    line_chart_base64_custom_report = fig_to_base64(fig_report_custom)
                                    line_chart_url_custom = f"data:image/png;base64,{line_chart_base64_custom_report}"

                                    top_5_allocations_df = st.session_state.simulation_data["custom_allocation_df"].head(5)

                                    report_html_path_custom = generate_html_report(
                                        line_chart_url=line_chart_url_custom,
                                        allocation_df=top_5_allocations_df,
                                        summary_df=st.session_state.simulation_data["custom_summary_df"],
                                        logo_data_uri=get_base64_image("logo.png")
                                    )

                                    with open(report_html_path_custom, "r", encoding="utf-8") as file:
                                        report_html_custom = file.read()

                                    st.components.v1.html(report_html_custom, height=1000, scrolling=True)



 





































            st.markdown('<h2>Single Stock Portfolios</h2>', unsafe_allow_html=True)


            raw_df = simulation_data.get("computed_single_stock_data", pd.DataFrame())
            if raw_df.empty:
                st.info("No single-stock portfolio data available.")


            # Always work on a *fresh copy* so the cached object stays pristine
            df = raw_df.copy(deep=True)

            # ——————————————————————————————————————————————————————————
            # 2. Build a Styler for pretty, non-destructive formatting
            # ——————————————————————————————————————————————————————————
            percent_cols = ['Return', 'Volatility', 'Dividend Yield','Max Drawdown']
            ratio_cols   = ['Sharpe Ratio', 'Sortino Ratio']

            # Apply specific formatting
            fmt_map = (
                {c: '{:.1%}' for c in percent_cols if c in df.columns} |
                ({'Dividend Yield': '{:.2%}'} if 'Dividend Yield' in df.columns else {}) |
                {c: '{:.3f}' for c in ratio_cols if c in df.columns} |
                ({'Weight': '{:.1f}'} if 'Weight' in df.columns else {})
            )


            table_html = (
                df.style
                .format(fmt_map, na_rep="")            # only affects HTML, not the data
                .set_table_attributes('class="styled-table" border="0"')
                .hide(axis="index")                    # same effect as index=False
                .to_html(escape=False)
            )

            # ——————————————————————————————————————————————————————————
            # 3. Display (single markdown call = zero extra vertical space)
            # ——————————————————————————————————————————————————————————
            st.markdown(table_html, unsafe_allow_html=True)




































            def display_efficient_frontier_table(efficient_frontier_data, tickers):
                # Header
                st.markdown(
                    '<h2 style="margin-top: 80px; color: #0a4daa;">Risk Reward Portfolio Table</h2>',
                    unsafe_allow_html=True
                )

                # 1) Extract per-stock weights as numeric
                ticker_weights_data = pd.DataFrame(efficient_frontier_data["weights"], columns=tickers)

                # 2) Combine with portfolio metrics
                efficient_data_table = pd.DataFrame({
                    "Portfolio Return": [round(val, 3) for val in efficient_frontier_data["returns"]],
                    "Portfolio Volatility": [round(val, 3) for val in efficient_frontier_data["volatilities"]],
                    "Sharpe Ratio": [round(val, 3) for val in efficient_frontier_data["sharpe_ratios"]],
                    "Sortino Ratio": [round(val, 3) for val in efficient_frontier_data["sortino_ratios"]],
                    "Dividend Yield": [round(val, 3) for val in efficient_frontier_data["dividend_yields"]],
                }).join(ticker_weights_data)

                # 3) Ensure all columns are numeric
                for column in ["Portfolio Return", "Portfolio Volatility", "Sharpe Ratio", "Sortino Ratio", "Dividend Yield"] + tickers:
                    efficient_data_table[column] = pd.to_numeric(efficient_data_table[column], errors="coerce")

                # 4) Sort by Sharpe Ratio
                efficient_data_table = efficient_data_table.sort_values(by="Sharpe Ratio", ascending=False)

                # 4.5) Add Row number column
                efficient_data_table.insert(0, "Row", range(len(efficient_data_table)))

                # 5) Format numeric values
                display_df = efficient_data_table.copy()
                display_df[tickers] = display_df[tickers].applymap(lambda x: f"{x:.2%}")
                display_df["Dividend Yield"] = display_df["Dividend Yield"].map("{:.2%}".format)
                display_df[["Portfolio Return", "Portfolio Volatility", "Sharpe Ratio", "Sortino Ratio"]] = \
                    display_df[["Portfolio Return", "Portfolio Volatility", "Sharpe Ratio", "Sortino Ratio"]].round(3)

                # Convert table to HTML
                html_table = display_df.to_html(classes="custom-portfolio-table", escape=False, index=False)

                # Add scrollable wrapper around it
                st.markdown(
                    f"""
                    <div class="custom-table-wrapper">
                        {html_table}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Extract the data we need
            efficient_data = simulation_data["efficient_frontier_data"]  
            ticker_list = simulation_data["tickers"]

            # Display the Risk Reward Portfolio Table
            display_efficient_frontier_table(efficient_data, ticker_list)




































            # In your main code, after the simulation runs:
            if st.session_state.get("simulation_ran", False):
                simulation_data = st.session_state.simulation_data


                # Extract stored data for display
                parameters = st.session_state.parameters
                simulation_data = st.session_state.simulation_data

                # Extract variables from simulation_data
                tickers = simulation_data.get("tickers", [])
                returns = simulation_data.get("returns", pd.DataFrame())

                # Proceed with correlation matrix only if we have valid data
                if tickers and not returns.empty and all(stock in returns.columns for stock in tickers):
                    correlation_matrix = returns[tickers].corr()

                    st.markdown('<h2 style="margin-top: 80px;">Correlation Table</h2>', unsafe_allow_html=True)

                    correlation_matrix_df = correlation_matrix.copy()
                    correlation_matrix_df.insert(0, "Asset", tickers)

                    def get_background_color(value, vmin, vmax):
                        """
                        Calculate a dynamically scaled heatmap background color based on the correlation value.

                        The heatmap uses a gradient from light red (#EC7063, negative values) to light blue (#85C1E9, neutral) 
                        to light green (#97E68C, positive values). The range is dynamically adjusted based on the min (vmin) 
                        and max (vmax) values of the dataset.
                        """
                        if not vmin <= value <= vmax:
                            raise ValueError("Value must be within the specified range (vmin to vmax).")

                        # Normalize the value to a 0-1 range based on vmin and vmax
                        normalized = (value - vmin) / (vmax - vmin)
                        # Negative values - soft pastel coral
                        red_neg, green_neg, blue_neg = (255, 200, 200 )  # #FFB7B2 (soft pastel coral)

                        # Neutral values - light pastel blue
                        red_neutral, green_neutral, blue_neutral = (200, 234, 255)  # #ADD8E6 (light pastel blue)

                        # Positive values - pastel mint green
                        red_pos, green_pos, blue_pos = (228, 255, 192)  # #98FB98 (pastel mint green)

                        if value < 0:
                            # Interpolate between light red and neutral (light blue) for negative values
                            red = int(red_neg * (1 - normalized) + red_neutral * normalized)
                            green = int(green_neg * (1 - normalized) + green_neutral * normalized)
                            blue = int(blue_neg * (1 - normalized) + blue_neutral * normalized)
                        else:
                            # Interpolate between neutral (light blue) and light green for positive values
                            red = int(red_neutral * (1 - normalized) + red_pos * normalized)
                            green = int(green_neutral * (1 - normalized) + green_pos * normalized)
                            blue = int(blue_neutral * (1 - normalized) + blue_pos * normalized)

                        return f"background-color: rgba({red}, {green}, {blue}, 0.8);"



                    def get_left_column_style():
                        """Return the style for the left column (blue background)."""
                        return "background-color:white; color:rgb(19, 87, 181); font-weight: bold;"  # Light blue

                    def get_top_column_style():
                        """Return the style for the top column (white background, blue text)."""
                        return "background-color:white; color:rgb(19, 87, 181); font-weight: bold;"  # White background, blue text


                    # Compute vmin and vmax from the correlation matrix
                    vmin = correlation_matrix_df[tickers].min().min()
                    vmax = correlation_matrix_df[tickers].max().max()

                    # Generate the HTML table with sticky left column
                    html_table = f"""
                    <div class="correlation-table-container" style="overflow-x: auto;">
                        <table class="correlation-table" style="margin-bottom: 0px; border-collapse: collapse; text-align: center;">
                            <thead>
                                <tr>
                                    <th style="border: 1px solid #d0d7eb; padding: 8px; {get_left_column_style()} position: sticky; left: 0; background: #fff; z-index: 2;">Asset</th>
                                    {''.join(f'<th style="border: 1px solid #d0d7eb; padding: 8px; {get_top_column_style()}">{stock}</th>' for stock in tickers)}
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(
                                    f"<tr><td style='border: 1px solid #d0d7eb; padding: 8px; {get_left_column_style()} position: sticky; left: 0; background: #fff; z-index: 1'>{row['Asset']}</td>" +
                                    ''.join(
                                        f"<td style='border: 1px solid #d0d7eb; padding: 8px; {get_background_color(row[col], vmin, vmax)}'>{row[col]:.2f}</td>"
                                        for col in tickers
                                    ) +
                                    "</tr>"
                                    for _, row in correlation_matrix_df.iterrows()
                                )}
                            </tbody>
                        </table>
                    </div>
                    """
                    # ---------- 1. static CSS (original look + overlay skeleton) ----------------
                    base_css = """
                    /* --- ORIGINAL APPEARANCE (unchanged) --- */
                    .correlation-table-container{
                    width:100%;max-width:1200px;overflow-x:auto;-webkit-overflow-scrolling:touch;
                    box-shadow:0 4px 12px rgba(0,0,0,.1);border-radius:12px;background:#fff;
                    border:1px solid #0a4daa;margin-bottom:10px;
                    }
                    .correlation-table{
                    border-collapse:separate;border-spacing:0;width:100%;font:1em sans-serif;
                    box-shadow:0 4px 12px rgba(0,0,0,.1);border-radius:12px;table-layout:auto;
                    }
                    .correlation-table tbody tr:nth-of-type(even){background:#f9f9f9;}
                    .correlation-table thead th{background:#427ecd;}
                    .correlation-table tbody td{background:#fff;color:#000;}
                    .correlation-table tbody tr:last-of-type td{border-bottom:none;}
                    .correlation-table th:first-child,
                    .correlation-table td:first-child{position:sticky;left:0;background:#fff;z-index:2;}
                    .correlation-table thead th:first-child{z-index:3;}

                    /* --- CROSS-HAIR OVERLAY skeleton --- */
                    .correlation-table th,
                    .correlation-table td{
                    position:relative;                /* anchor for ::after overlay */
                    }
                    .correlation-table th::after,
                    .correlation-table td::after{
                    content:"";
                    position:absolute;inset:0;
                    background:rgba(255,249,209,1); /* soft yellow */
                    transition: opacity .18s ease-out, transform .18s ease-out;

                    opacity:0;                        /* hidden by default */
                    pointer-events:none;              /* don’t block clicks/hover */
                    mix-blend-mode:multiply;
                    border-radius:inherit;
                    }

                    /* row-label highlight (first column) */
                    .correlation-table tbody tr:hover td:first-child::after{
                    opacity:1;
                    transform: scale(1);            /* subtle zoom */
                    transition: opacity .18s ease-out, transform .18s ease-out;
                    }
                    """

                    # ---------- 2. per-column header overlay rules -----------------
                    col_css_parts = []
                    #  → start at 2 (skip 'Asset') and go up to n+1
                    for nth in range(2, len(tickers) + 2):          # len(tickers) == n
                        col_css_parts.append(f"""
                    /* ticker column {nth-1} (header only) */
                    .correlation-table:has(tbody td:nth-child({nth}):hover) thead th:nth-child({nth})::after,
                    .correlation-table thead th:nth-child({nth}):hover::after {{
                        opacity: 1;
                    }}""")


                    full_css = f"<style>{base_css}{''.join(col_css_parts)}</style>"

                    # ---------- 3. emit CSS once, then the table --------------------------------
                    st.markdown(full_css, unsafe_allow_html=True)
                    st.markdown(html_table, unsafe_allow_html=True)



























            def get_download_link(data, filename, mime, label, disabled=False):
                """Generate a styled HTML download link. If disabled, show same visual style but block download."""
                if disabled:
                    # Same style but not clickable
                    return f'''
                        <div style="
                            display: inline-block;
                            color: white;
                            background: linear-gradient(90deg, #107c41 0%, #1fa960 100%);
                            text-decoration: none;
                            border: 2px solid #107c41;
                            border-radius: 8px;
                            padding: 10px 20px;
                            margin-bottom: 20px;
                            font-size: 16px;
                            font-weight: 500;
                            text-align: center;
                            cursor: not-allowed;
                        ">
                            {label}
                        </div>
                    '''
                else:
                    if isinstance(data, bytes):
                        b64 = base64.b64encode(data).decode()
                    else:
                        b64 = base64.b64encode(data.encode()).decode()

                    return f'''
                        <a href="data:{mime};base64,{b64}" download="{filename}" style="
                            display: inline-block;
                            color: white;
                            background: linear-gradient(90deg, #107c41 0%, #1fa960 100%);
                            text-decoration: none;
                            border: 2px solid #107c41;
                            border-radius: 8px;
                            padding: 10px 20px;
                            margin-bottom: 20px;
                            font-size: 16px;
                            font-weight: 500;
                            text-align: center;
                            cursor: pointer;
                        ">
                            {label}
                        </a>
                    '''

            @st.fragment
            def download_section(simulation_data=simulation_data, product_id=None):
                st.markdown('<h2 style="margin-top: 80px;">Portfolio Data Download</h2>', unsafe_allow_html=True)

                if is_pro(product_id):
                    # --- Excel ---
                    portfolio_excel = prepare_portfolio_download(
                        portfolio_returns_df=simulation_data["portfolio_returns_df"],
                        cumulative_returns_df=simulation_data.get("cumulative_returns_df"),
                        max_sharpe_allocation_df=simulation_data.get("max_sharpe_allocation_df"),
                        max_sharpe_summary_df=simulation_data.get("max_sharpe_summary_df"),
                        max_return_allocation_df=simulation_data.get("max_return_allocation_df"),
                        max_return_summary_df=simulation_data.get("max_return_summary_df"),
                        min_volatility_allocation_df=simulation_data.get("min_volatility_allocation_df"),
                        min_volatility_summary_df=simulation_data.get("min_volatility_summary_df"),
                        custom_allocation_df=simulation_data.get("custom_allocation_df"),
                        custom_summary_df=simulation_data.get("custom_summary_df"),
                        dividend_allocation_df=simulation_data.get("dividend_allocation_df"),
                        dividend_summary_df=simulation_data.get("dividend_summary_df"),
                        returns=simulation_data["returns"],
                        tickers=simulation_data["tickers"],
                        covariance_matrix=simulation_data["returns"].cov()
                            if not simulation_data["returns"].empty else None,
                        correlation_matrix=simulation_data["returns"][simulation_data["tickers"]].corr()
                            if simulation_data["tickers"]
                            and not simulation_data["returns"].empty
                            and all(s in simulation_data["returns"].columns for s in simulation_data["tickers"])
                            else None
                    )

                    # --- CSV ---
                    efficient_frontier_csv = prepare_efficient_frontier_download(
                        efficient_results=simulation_data["efficient_frontier_data"],
                        tickers=simulation_data["tickers"]
                    )

                    st.markdown(get_download_link(
                        data=portfolio_excel,
                        filename="portfolio_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        label="Download Portfolio Data (Excel)"
                    ), unsafe_allow_html=True)

                    st.markdown(get_download_link(
                        data=efficient_frontier_csv,
                        filename="efficient_frontier_data.csv",
                        mime="text/csv",
                        label="Download Efficient Frontier Data (CSV)"
                    ), unsafe_allow_html=True)

                else:
                    # Show visually identical buttons but non-functional
                    st.markdown(get_download_link(
                        data="",
                        filename="portfolio_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        label="Download Portfolio Data (Excel)",
                        disabled=True
                    ), unsafe_allow_html=True)

                    st.markdown(get_download_link(
                        data="",
                        filename="efficient_frontier_data.csv",
                        mime="text/csv",
                        label="Download Efficient Frontier Data (CSV)",
                        disabled=True
                    ), unsafe_allow_html=True)

                    # Lock & Upgrade CTA below
                    st.markdown(
                        """
                        <div style="margin-top: 10px; font-size: 15px;">
                            🔒
                            <a href="https://peakportfolio.ai/#Pricing" target="_blank" style="
                                text-decoration: none;
                                color: #0a4daa;
                                font-weight: 600;
                                margin-left: 8px;
                                font-size: 16px;
                            ">
                                Upgrade to Pro
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ▲▲——— call the fragment so it renders ———▲▲
            download_section(simulation_data=simulation_data, product_id=product_id)



 





















            st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

            with st.expander("User Agreement & Liability Waiver"):
                st.markdown("""
            **Disclaimer for Peak Portfolio**

            **1. No Investment Advice**  
            PeakPortfolio.ai is an educational and informational tool designed to assist users in analyzing, constructing, and optimizing investment portfolios. The Platform does not provide personalized financial, investment, tax, or legal advice. Any content, simulations, or AI-driven insights generated by the Platform are for informational purposes only and should not be construed as financial advice, recommendations, or endorsements. Users should conduct independent research and consult a licensed financial professional before making any investment decisions.

            **2. Risks of Investing**  
            Investing in financial markets carries inherent risks, including but not limited to capital loss, market volatility, liquidity risks, geopolitical risks, currency fluctuations, and economic downturns. Peak Portfolio does not guarantee profits, returns, or the accuracy of its risk-reward projections. Past performance of any portfolio or asset is not indicative of future results. Users assume full responsibility for their investment decisions and agree that Peak Portfolio is not liable for any financial losses, opportunity costs, or adverse market events.

            **3. AI & Algorithmic Limitations**  
            Peak Portfolio leverages AI-powered analytics, statistical models, and mathematical optimization techniques such as Mean-Variance Optimization (MVO) and the Modern Portfolio Theory (MPT). However, these models have limitations:

            AI-generated insights may be incomplete, outdated, or incorrect due to evolving market conditions.  
            Portfolio recommendations are based on historical data and may not account for unforeseen economic events or black swan events.  
            AI-driven adjustments do not replace professional financial advice or human discretion.  
            The Platform does not guarantee that its AI-powered portfolio recommendations will outperform other investment strategies.

            Users acknowledge that AI-driven simulations and insights should be used as a supplementary tool and not as a definitive investment strategy.

            **4. Data Accuracy & Third-Party Sources**  
            The Platform aggregates financial data from third-party sources such as Yahoo Finance and market APIs. While we strive to ensure accuracy and timeliness, Peak Portfolio does not guarantee the completeness, reliability, or accuracy of this data. Errors, omissions, or delays in financial data may impact portfolio outcomes. Users should verify critical information before making investment decisions.

            **5. Regulatory Compliance & Jurisdiction**  
            Peak Portfolio does not operate as a registered financial advisor, broker-dealer, or fiduciary. The Platform does not execute trades, manage investments, or provide discretionary investment services. Users must ensure that their use of the Platform complies with applicable financial laws and regulations in their jurisdiction. Certain features, such as portfolio simulations and AI-driven insights, may be restricted in specific countries due to regulatory limitations. Users acknowledge their responsibility to comply with all legal and tax obligations when using the Platform.

            **6. No Fiduciary Duty**  
            Use of the Platform does not establish a fiduciary relationship between Peak Portfolio and its users. The Platform provides generalized insights and does not act in the best interests of any specific investor. Users remain solely responsible for assessing the suitability of any portfolio allocation or investment strategy.

            **7. No Liability for Losses**  
            To the fullest extent permitted by law, Peak Portfolio, its founders, employees, partners, and affiliates disclaim any liability for financial losses, damages, claims, or liabilities arising from:

            Investment decisions based on the Platform's outputs.  
            Market downturns, portfolio underperformance, or unexpected financial events.  
            Technical failures, data errors, AI misinterpretations, or third-party API outages.  
            Regulatory or tax consequences related to portfolio allocations.

            Users waive any claims against Peak Portfolio for direct, indirect, incidental, punitive, or consequential damages, including but not limited to financial losses and lost investment opportunities.

            **8. User Responsibility & Acknowledgment**  
            By using the Platform, users acknowledge and agree to:

            Use the Platform at their own risk.  
            Accept that AI-generated outputs are for informational purposes only and do not constitute financial advice.  
            Conduct independent research before making investment decisions.  
            Understand that financial markets are unpredictable, and no tool, AI model, or optimization algorithm can guarantee future performance.

            **9. Changes & Amendments**  
            Peak Portfolio reserves the right to update, modify, or discontinue features of the Platform without prior notice. We may also revise this disclaimer at any time. Continued use of the Platform constitutes acceptance of any changes.

            **10. Contact & Support**  
            For questions regarding this disclaimer, please contact us at [support@peakportfolio.ai](mailto:support@peakportfolio.ai).
            """)


        else:
            # Re-display the particle effect and subtitle if the simulation is not run
            with subtitle_placeholder.container():
                st.markdown(
                    """
                    <div style="text-align: center; font-weight: 500; font-size: 1.2em; color: #6E7B8B; margin: 15px 0;">
                        Adjust portfolio parameters in the sidebar and click <strong>Create My Portfolio</strong> to see results.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Only render particles if simulation has not started
            with particle_placeholder.container():
                render_particles()  
