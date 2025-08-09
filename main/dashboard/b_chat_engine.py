# b_nlp_engine.py

import logging
import tiktoken
import os
from openai import OpenAI
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz  # Use rapidfuzz for efficient fuzzy matching
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Instantiate the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Counts the number of tokens in a given text for a specified model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logging.error(f"Error counting tokens: {e}")
        return 0

def query_gpt(user_question: str, prepared_data: str, recent_events_summary: str, model: str = "gpt-4o", max_tokens: int = 8000, temperature: float = 0.90) -> str:
    """
    Constructs a context-rich prompt dynamically and queries GPT with it.
    """
    try:
        # Dynamically build the prompt
        prompt = f"""
        You are a knowledgeable financial advisor with expertise in risk management and asset allocation, catering to both retail traders and investment managers. Your mission is to provide clear, actionable, and data-driven insights that are accessible to all users, regardless of their financial expertise.

        **simulation Data Summary:**
        {prepared_data}
        
        **Recent Market Events:**
        {recent_events_summary}

        **User Question:**
        {user_question}

        **Response Guidelines:**
        You are a highly knowledgeable financial advisor and quantitative analyst specializing in portfolio management and investment strategies. Your task is to generate **clear, actionable, and insightful responses** based on the provided context and user question. The goal is to offer advice or explanations in a manner that is easy to understand, accurate, and aligned with the user's intent.

        **Context Provided**:
        1. **Portfolio Data**: Summarized information about the portfolio, including metrics such as expected return, volatility, risk metrics, and allocation breakdowns.
        2. **Recent Market Events**: Key market trends or news relevant to the portfolio or broader financial landscape.
        3. **User Question**: The specific query or concern provided by the user.

        **Response Guidelines**:
        1. **Align with User Goals**:
        - Address the user's intent directly, ensuring the response aligns with their financial objectives, risk tolerance, and investment horizon.
        - If user intent is unclear, provide general insights that encourage thoughtful decision-making or further exploration.

        2. **Provide Actionable and Analytical Insights**:
        - Use the provided portfolio data and market context to deliver **strategic and data-driven insights**.
        - Explain risk-return dynamics, potential adjustments, and alternative strategies in simulationple terms.
        - If applicable, suggest actions that balance short-term opportunities with long-term goals.

        3. **Cite Sources Explicitly**:
        - Reference relevant market data, news articles, or reports to support your recommendations.
        - Reference all market data sources clearly, including news articles, market reports, and any other credible sources.
        - Generate responses that explicitly cite sources using the name of the source as a clickable link. For example, use 'Bloomberg Article on Market Trends' as the hyperlink text instead of the name followed by a 'read more' link. Ensure citations are concise, formatted consistently, and include brief summaries of their relevance to the topic. Key points from the sources should be summarized clearly to provide context and enhance understanding.

        4. **Adapt to Question Type**:
        - **For specific questions**: Provide a concise and focused response based on the question.
        - **For general questions**: Offer a high-level overview and include actionable suggestions.
        - **For comparisons**: Clearly outline the strengths, weaknesses, and trade-offs between options.

        5. **Ensure Accessibility and Clarity**:
        - Avoid overly technical jargon unless necessary, and define terms simulationply if used.
        - Use structured formats (e.g., bullet points, sections) to improve readability.
        - Avoid repeating information available in the interface or other outputs.

        6. **Maintain Credibility and Integrity**:
        - Avoid making assumptions not supported by the provided context.
        - Emphasize that all advice is informational and users should consult a financial advisor for personalized guidance.
        - Ensure neutrality and avoid biases toward specific strategies or assets.

        **Format the Response**:
        - **Start** with a brief overview or answer to the user's question.
        - **Follow** with actionable insights or explanations, segmented for clarity (e.g., portfolio adjustments, risk insights).
        - Where applicable, include **source citations** with links to credible articles, data, or reports.
        - **Conclude** with a high-level recommendation or encouragement for further exploration.

        **Important Notes**:
        - Do not include specifics unless explicitly provided in the user question or context.
        - Avoid unnecessary details or unrelated tangents to maintain focus and relevance.
        - Always cite sources clearly and succinctly where applicable.
        """

        # Query GPT
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"GPT query failed: {e}")
        return f"Error: {e}"

def fetch_recent_events(stock_list: list, max_articles: int = 8) -> dict:
    """
    Fetches recent news for a given list of tickers using yfinance, returning a summary and data dictionary.
    """
    news_data = {}
    for stock in stock_list:
        ticker = yf.Ticker(stock)
        try:
            news = ticker.news
            formatted_news = []
            for article in news[:max_articles]:
                publish_time = article.get("providerPublishTime", None)
                publish_date = (
                    datetime.utcfromtimestamp(publish_time).strftime("%Y-%m-%d %H:%M:%S UTC")
                    if publish_time else "Unknown date"
                )
                formatted_news.append({
                    "title": article.get("title", "No title"),
                    "date": publish_date,
                    "source": article.get("publisher", "Unknown source"),
                    "url": article.get("link", "#")
                })
            news_data[stock] = formatted_news
        except Exception as e:
            logging.error(f"Failed to fetch news for {stock}: {e}")
            news_data[stock] = [{
                "title": f"Failed to fetch news: {str(e)}",
                "date": "N/A",
                "source": "N/A",
                "url": "#"
            }]

    # Summarize news with proper Markdown formatting and clickable links
    news_summary = "\n".join(
        [
            f"**{stock}**:\n" + "\n".join(
                [f"- [{event['date']}] {event['title']} ({event['source']}) [Read More]({event['url']})" for event in events]
            )
            for stock, events in news_data.items()
        ]
    )
    return {"news_data": news_data, "news_summary": news_summary}

def normalize_term(user_term: str, alias_map: dict, threshold: int = 60) -> str:
    """
    Normalize a user-provided term to a canonical term based on an alias mapping with fuzzy matching.
    All terms are treated with equal priority.
    """
    user_term = user_term.lower()

    # Build a dictionary of all possible terms and their canonical counterparts
    all_terms = {canonical_term: canonical_term for canonical_term in alias_map.keys()}
    for canonical_term, aliases in alias_map.items():
        for alias in aliases:
            all_terms[alias] = canonical_term

    # Fuzzy match the user term to all possible terms
    match, score = process.extractOne(user_term, all_terms.keys(), scorer=fuzz.ratio)
    if score >= threshold:
        return all_terms[match]  # Return the mapped canonical term

    return user_term  # If no match meets the threshold, return the term as-is

























def prepare_simulation_data_for_prompt(simulation_data: dict, max_tokens: int = 10000) -> dict:
    """
    Prepares comprehensive simulation data for the GPT prompt with alias handling for ambiguous terms.
    - ALL Efficient Frontier portfolios
    - Max Sharpe and Max Return Portfolios
    - Custom Portfolio Data (if available)
    - Single Stock Data
    - Full Correlation and Covariance Matrices
    """
    try:
        ALIAS_MAP = {
            # max-Sharpe / “Personalized”
            "max_sharpe_portfolio": [
                "personalized portfolio", "balanced portfolio", "risk adjusted portfolio",
                "sharpe optimized portfolio", "maximum sharpe ratio portfolio",
                "highest sharpe ratio", "sharpe ratio king", "risk reward portfolio",
                "optimized sharpe portfolio", "best risk-adjusted return",
                "best reward to risk", "peakportfolio"
            ],

            # max-return / Aggressive
            "max_return_portfolio": [
                "highest return portfolio", "maximized return portfolio", "max return",
                "return maximizing portfolio", "profit maximized", "most profitable portfolio",
                "maximum profit portfolio", "best return", "top return portfolio",
                "highest gain portfolio", "return leader", "high risk portfolio"
            ],

            # custom
            "custom_portfolio_data": [
                "custom portfolio", "user-defined portfolio", "bespoke portfolio",
                "tailored portfolio", "portfolio I made", "my portfolio",
                "individual portfolio", "unique portfolio"
            ],

            # efficient-frontier set
            "efficient_frontier_data": [
                "efficient frontier portfolios", "optimal portfolios", "frontier portfolios",
                "pareto optimal portfolios", "best portfolios", "portfolio frontier",
                "set of best portfolios", "efficient curve"
            ],

            # matrices & single-stock data (unchanged)
            "correlation_matrix": [
                "correlation data", "correlation matrix", "asset correlations",
                "relationship matrix"
            ],
            "covariance_matrix": [
                "covariance data", "covariance matrix", "asset covariances",
                "investment variances"
            ],
            "single_stock_portfolio_data": [
                "individual stock metrics", "single stock data", "stock breakdown"
            ],

            # min-vol
            "min_volatility_portfolio": [
                "low volatility portfolio", "minimum volatility portfolio",
                "least volatile portfolio", "risk minimized portfolio"
            ],

            # dividend
            "dividend_portfolio": [
                "highest dividend yield portfolio", "dividend focused portfolio",
                "maximum dividend portfolio", "income generating portfolio"
            ],
        }









        combined_summary = "### Portfolio simulation Data\n\n"

        # Retrieve simulation_data from session state
        simulation_data = st.session_state.simulation_data

        ai_refinement_option = st.session_state.get("ai_refinement_option", "Yes")

        # --- Extract Basic Data ---
        tickers = simulation_data.get("tickers", [])
        returns_df = simulation_data.get("returns", pd.DataFrame())

        # --- Low Volatility Portfolio Data (Min Volatility) ---
        combined_summary += "### Conservative Portfolio\n"


        # Helper for safely appending metric blocks
        def _append_metrics(title: str,
                            exp_ret, vol, sharpe, sortino, dyld, weights):
            nonlocal combined_summary
            combined_summary += f"### {title}\n"
            if all(v is not None for v in
                [exp_ret, vol, sharpe, sortino, dyld, weights]):
                combined_summary += (
                    f"- **Expected Return:** {exp_ret:.4f}\n"
                    f"- **Volatility:** {vol:.4f}\n"
                    f"- **Sharpe Ratio:** {sharpe:.4f}\n"
                    f"- **Sortino Ratio:** {sortino:.4f}\n"
                    f"- **Dividend Yield:** {dyld:.4f}\n"
                    "- **Allocations:**\n"
                )
                for stock, w in zip(tickers, weights):
                    combined_summary += f"  - {stock}: {w * 100:.2f}%\n"
                combined_summary += "\n"
            else:
                combined_summary += f"No {title.lower()} data available.\n\n"

        # ───────────────────────────────────────────────────────── Conservative (no AI)
        _append_metrics(
            "Conservative Portfolio",
            simulation_data.get("min_volatility_return"),
            simulation_data.get("min_volatility_volatility"),
            simulation_data.get("min_volatility_ratio"),
            simulation_data.get("min_volatility_sortino"),
            simulation_data.get("min_volatility_yield"),
            simulation_data.get("min_volatility_weights", [])
        )

        # ─────────────────────────────────────────────────────── Personalized (AI stays)
        ai_refinement_option = st.session_state.get("ai_refinement_option", "Yes")
        if ai_refinement_option == "Yes" and simulation_data.get("ai_refined_weights"):
            metrics     = simulation_data["ai_portfolio_metrics"]["balanced"]
            weights     = simulation_data["ai_refined_weights"]["balanced"]
            exp_ret     = metrics["expected_return"]
            vol         = metrics["volatility"]
            sharpe      = metrics["sharpe_ratio"]
            sortino     = metrics["sortino_ratio"]
            dyld        = metrics["dividend_yield"]
        else:
            exp_ret     = simulation_data.get("max_sharpe_return")
            vol         = simulation_data.get("max_sharpe_volatility")
            sharpe      = simulation_data.get("max_sharpe_ratio")
            sortino     = simulation_data.get("max_sharpe_sortino")
            dyld        = simulation_data.get("max_sharpe_yield")
            weights     = simulation_data.get("max_sharpe_weights", [])

        _append_metrics("Personalized Portfolio",
                        exp_ret, vol, sharpe, sortino, dyld, weights)

        # ───────────────────────────────────────────────────────── Aggressive (no AI)
        _append_metrics(
            "Aggressive Portfolio",
            simulation_data.get("max_return_return"),
            simulation_data.get("max_return_volatility"),
            simulation_data.get("max_return_ratio"),
            simulation_data.get("max_return_sortino"),
            simulation_data.get("max_return_yield"),
            simulation_data.get("max_return_weights", [])
        )

        # ───────────────────────────────────────────────────── Dividend-Focused
        dividend_portfolio = simulation_data.get("dividend_portfolio_data")
        if dividend_portfolio:
            d_alloc_df = simulation_data.get("dividend_allocation_df", pd.DataFrame())
            _append_metrics(
                "Dividend-Focused Portfolio",
                dividend_portfolio.get("d_expected_return"),
                dividend_portfolio.get("d_volatility"),
                dividend_portfolio.get("d_sharpe_ratio"),
                dividend_portfolio.get("d_sortino_ratio"),
                dividend_portfolio.get("d_dividend_yield"),
                d_alloc_df["% Allocation"].tolist() if not d_alloc_df.empty else []
            )
        else:
            combined_summary += "### Dividend-Focused Portfolio\nNo dividend-focused portfolio data available.\n\n"

        # ───────────────────────────────────────────────────── Custom
        custom_portfolio = simulation_data.get("custom_portfolio_data")
        if custom_portfolio:
            c_alloc_df = simulation_data.get("custom_allocation_df", pd.DataFrame())
            _append_metrics(
                "Custom Portfolio",
                custom_portfolio.get("c_expected_return"),
                custom_portfolio.get("c_portfolio_volatility"),
                custom_portfolio.get("c_sharpe_ratio"),
                custom_portfolio.get("c_sortino_ratio"),
                custom_portfolio.get("c_dividend_yield"),
                c_alloc_df["% Allocation"].tolist() if not c_alloc_df.empty else []
            )
        else:
            combined_summary += "### Custom Portfolio\nNo custom portfolio data available.\n\n"

        # ─────────────────────────────────────────────── Token-budget safety valve
        # crude slice: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(combined_summary) > max_chars:
            combined_summary = combined_summary[:max_chars]





        # --- Correlation and Covariance Matrices ---
        combined_summary += "### Correlation and Covariance Matrices\n"
        if isinstance(returns_df, pd.DataFrame) and not returns_df.empty:
            corr_matrix = returns_df[tickers].corr().to_markdown()
            cov_matrix = returns_df[tickers].cov().to_markdown()
            combined_summary += f"- **Correlation Matrix:**\n{corr_matrix}\n\n"
            combined_summary += f"- **Covariance Matrix:**\n{cov_matrix}\n\n"
        else:
            combined_summary += "No returns data available.\n\n"

        # --- Single Stock Data ---
        combined_summary += "### Single Stock Metrics\n"
        single_stock_portfolio_data = simulation_data.get("single_stock_portfolio_data", pd.DataFrame())
        if isinstance(single_stock_portfolio_data, pd.DataFrame) and not single_stock_portfolio_data.empty:
            combined_summary += single_stock_portfolio_data.to_markdown(index=False) + "\n\n"
        else:
            combined_summary += "No single stock data available.\n\n"


    except Exception as e:
        logging.error(f"Error preparing simulation data for prompt: {e}")
        return {
            "summary": combined_summary,
            "alias_map": ALIAS_MAP,
            "tickers": tickers
        }

