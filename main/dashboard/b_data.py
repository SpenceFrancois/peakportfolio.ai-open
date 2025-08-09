import pandas as pd
from .tiingo_client import client
from tiingo import TiingoClient
import streamlit as st
from datetime import datetime, timedelta




@st.cache_data(show_spinner=False)
def fetch_and_resample_data(tickers, start_date, end_date):
    """
    Tiingo version: fetches stock data, resamples it to monthly returns,
    and provides company names, raw Tiingo assetType strings,
    and dynamically calculated dividend yields.
    """
    # ----- one‑time bulk asset‑type map -----
    if not hasattr(fetch_and_resample_data, "_asset_type_map"):
        bulk = client.list_tickers()  # supported_tickers.csv from Tiingo
        fetch_and_resample_data._asset_type_map = {
            row["ticker"]: row["assetType"] for row in bulk
        }

    returns = pd.DataFrame()
    ticker_info = []

    for ticker in tickers:
        try:
            # ---- Monthly price ----
            price_data = client.get_ticker_price(
                ticker,
                startDate=start_date,
                endDate=end_date,
                frequency='monthly'
            )
            if not price_data:
                raise ValueError(f"No price data for {ticker}")

            df = pd.DataFrame(price_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df[['adjClose']].rename(columns={'adjClose': ticker})
            returns = df if returns.empty else returns.join(df, how='outer')

            # ---- Metadata ----
            meta = client.get_ticker_metadata(ticker)
            company_name = meta.get("name", f"Unknown ({ticker})")
            asset_type = fetch_and_resample_data._asset_type_map.get(
                ticker, "Unknown"
            )

        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue





        # ---- Compute trailing-12M dividend yield (split-safe) ----
        try:
            # trailing window for divs
            # Note: This clamp (today = datetime.today().date) applies only when end_date exceeds today’s date.
            today = datetime.today().date()
            window_end = min(datetime.fromisoformat(end_date).date(), today)
            window_start = window_end - timedelta(days=365)

            # 1) pull daily cash-dividends inside the window
            raw_divs = client.get_ticker_price(
                ticker,
                startDate=window_start.isoformat(),
                endDate=window_end.isoformat(),
                frequency="daily",
                fmt="json",
                columns=["divCash"],
            )
            divs = pd.DataFrame(raw_divs)
            if divs.empty:
                raise ValueError("no dividends returned")

            # 2) normalise, dedupe, filter positives
            divs = (divs.rename(columns={"date": "paymentDate", "divCash": "amount"})
                        .assign(paymentDate=lambda d: pd.to_datetime(d["paymentDate"]).dt.date))
            divs = (
                divs[divs["amount"] > 0]
                .query(" @window_start <= paymentDate <= @window_end")
                .groupby("paymentDate", as_index=False)["amount"]
                .sum()
            )
            if divs.empty:
                raise ValueError("no dividends in 365-day window")

            # 3) pull ALL splits from earliest dividend to window_end
            first_div = divs["paymentDate"].min()
            raw_splits = client.get_ticker_price(
                ticker,
                startDate=first_div.isoformat(),   # <- ensures pre-window splits included
                endDate=window_end.isoformat(),
                frequency="daily",
                fmt="json",
                columns=["splitFactor"],
            )
            splits_df = (pd.DataFrame(raw_splits)
                        .rename(columns={"date": "splitDate"})
                        .assign(splitDate=lambda s: pd.to_datetime(s["splitDate"]).dt.date))
            splits_df = splits_df[
                splits_df["splitFactor"].notna() & (splits_df["splitFactor"] != 1.0)
            ]

            # 4) helper: cumulative forward split factor
            def cum_split_factor(div_date):
                # multiply every split that happens *after* the dividend but *before* end_date
                relevant = splits_df[splits_df["splitDate"] > div_date]
                return relevant["splitFactor"].prod() if not relevant.empty else 1.0

            # 5) forward-adjust dividends
            divs["adjusted_amount"] = divs.apply(
                lambda r: r["amount"] / cum_split_factor(r["paymentDate"]), axis=1
            )

            # 6) annualised cash payout
            annual_cash = divs["adjusted_amount"].sum()

            # 7) denominator: most-recent **raw close** (split-adj, not dividend-adj)
            price_snap = client.get_ticker_price(
                ticker,
                startDate=(window_end - timedelta(days=5)).isoformat(),  # 5-day cushion
                endDate=window_end.isoformat(),
                frequency="daily",
                fmt="json",
                columns=["close"],
            )
            price_df = pd.DataFrame(price_snap)
            price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
            price_df.set_index("date", inplace=True)
            if price_df["close"].dropna().empty:
                raise ValueError("missing price")
            price = price_df["close"].iloc[-1]

            # 8) trailing-12M dividend yield
            dividend_yield = round(annual_cash / price, 4)

        except Exception as err:
            print(f"Warning computing dividends for {ticker}: {err}")
            dividend_yield = 0.0



            #test ticker for debugging
        if ticker == "SCHD, KO, PEP, SPY":
            print("Dividend rows:\n", divs)
            print("Split rows:\n", splits_df)
            print("Price data:\n", price_df.tail())
            print(f"Sum of adjusted dividends: {divs['adjusted_amount'].sum():.4f}")



        ticker_info.append({
            "ticker": ticker,
            "name": company_name,
            "type": asset_type,        # raw Tiingo label
            "dividend_yield": round(dividend_yield, 4)
        })

    # ---- Monthly returns ----
    returns = returns.sort_index().pct_change().dropna()
    return returns, ticker_info



@st.cache_data(show_spinner=False)
def fetch_benchmark_data(benchmark, start_date, end_date):
    """
    Fetches benchmark data and calculates monthly returns, along with its long name using Tiingo.

    Parameters:
        benchmark (str): Benchmark asset ticker.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        tuple: (DataFrame of monthly returns, benchmark long name)
    """
    if not benchmark:
        return None, None

    try:
        # 1. Fetch monthly adjusted prices
        price_data = client.get_ticker_price(
            benchmark,
            startDate=start_date,
            endDate=end_date,
            frequency='monthly'
        )
        df = pd.DataFrame(price_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df[['adjClose']].rename(columns={'adjClose': benchmark})

        # 2. Calculate monthly returns
        returns = df.pct_change().dropna()

        # 3. Get benchmark name
        metadata = client.get_ticker_metadata(benchmark)
        benchmark_name = metadata.get("name", f"Unknown ({benchmark})")

        return returns, benchmark_name

    except Exception as e:
        print(f"Error fetching benchmark data for {benchmark}: {e}")
        return None, None