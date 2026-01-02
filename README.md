# PeakPortfolio.ai – Open Source Edition

**PeakPortfolio.ai** is an AI-powered portfolio analysis and optimization platform that blends quantitative finance, market data pipelines, and AI-driven insights into a single, intuitive application. This repository contains the **open-source version** of the app — everything you need to run it locally, explore the algorithms, and extend the features for your own projects.

---

## 📖 Overview
PeakPortfolio connects portfolio optimization, asset-level data, prompt engineering, and context-aware AI that allows users to express investment goals in their own words and receive a personalized portfolio that is constructed, reviewed, and explained by AI agents putting the data into context and then shaping the portfolio around your intent.
Backend Transformation Workflow
Input & Baselines
Begins with user-supplied asset data and a natural-language investment brief.
A mean-variance optimization (MVO) engine computes three reference portfolios:
Minimum Volatility
Maximum Sharpe Ratio
Maximum Return
These serve as performance baselines.


Allocator Agent
Constructs a new portfolio based on the user context, asset list, and reference portfolios.
Applies all allocation constraints to produce a valid portfolio (ETF floors, crypto caps, position limits).
Proposes allocations, computes performance results, and turns the portfolio into a deliverable (JSON) for the Manager Agent.

Manager Agent
Serves as compliance and risk oversight.
Receives proposed allocations, user context, and computed portfolio metrics (expected return, volatility, Sharpe ratio, Sortino, dividend yield).
Assesses whether the portfolio is compliant and financially appropriate.
If violations occur, triggers a reallocation loop (up to three attempts).
If no compliant solution is found, defaults to the MVO-based balanced portfolio.

Explainer Agent
Delivers a client-ready narrative for each portfolio position.
Interprets the portfolio in light of the user’s objectives and risk profile
Explains rationale, expected contribution, and forward-looking outlook for each asset.


Finalization
Approved portfolio is re-evaluated through the quant engine.
Final output includes allocations, performance metrics, and human-grade explanations.
Delivered asynchronously to the user in under two minutes.


Core system components:
- **Backend Engineering** – APIs, data ingestion, risk and return calculations.
- **Quantitative Models** – Mean-variance optimization, Sortino and Sharpe analysis, dividend yield targeting, and max drawdown tracking.
- **AI Integration** – Market event summarization and strategy recommendations powered by OpenAI.
- **Frontend/UI** – Fully interactive dashboards built with Streamlit.
- **Authentication & Access Control** – Firebase-backed user management.
- **Cloud Deployment** – Production-ready for hosting and scaling.


Built over **8 months by Spencer Francois**, PeakPortfolio.ai demonstrates the integration of portfolio allocation, artificial intelligence, and software engineering into a coherent, easy-to-use system.

---

## 🚀 Features

- **Portfolio Optimization** – Mean-variance, dividend-focused, and AI-refined strategies.
- **Risk Metrics** – Sharpe ratio, Sortino ratio, downside deviation, and maximum drawdown calculations.
- **Custom Portfolio Analysis** – Define allocations, run backtests, and simulate performance.
- **Market Event Summaries** – Automatic news retrieval and summarization for portfolio assets.
- **Pro Mode (Disabled Here)** – Premium-only production features excluded from this release; local version still includes a full analytics environment for experimentation.

---

## 🛠 Installation

```bash

# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
# Create a .env file in the project root with:
TIINGO_API_KEY=your_tiingo_key
OPENAI_API_KEY=your_openai_key

#3. Run locally
streamlit run app.py
```

---

## 🌐 Live Version
A hosted version of PeakPortfolio.ai is available here:  
[https://app.peakportfolio.ai](https://app.peakportfolio.ai)

---


## 🙌 Acknowledgements

- [Tiingo](https://www.tiingo.com/) – Financial market data
- [Yahoo Finance](https://pypi.org/project/yfinance/) – Historical data and news
- [OpenAI](https://openai.com/) – AI-powered market summaries and analysis
- [Streamlit](https://streamlit.io/) – Application framework
- [Firebase](https://firebase.google.com/) – Authentication and user management

---

## 💡 Contributing

Contributions are welcome! Fork the repo, make your changes, and submit a pull request.

---

## 🧠 Author

**Creator:** Spencer Francois 
