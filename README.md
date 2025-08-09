# PeakPortfolio.ai – Open Source Edition

**PeakPortfolio.ai** is an AI-powered portfolio analysis and optimization platform that blends quantitative finance, market data pipelines, and AI-driven insights into a single, intuitive application. This repository contains the **open-source version** of the app — everything you need to run it locally, explore the algorithms, and extend the features for your own projects.

---

## 📖 Overview

Built over **8 months by a single developer**, PeakPortfolio.ai is a demonstration of end-to-end product creation in a specialized domain:

- **Backend Engineering** – APIs, data ingestion, risk and return calculations.
- **Quantitative Models** – Mean-variance optimization, Sortino and Sharpe analysis, dividend yield targeting, and max drawdown tracking.
- **AI Integration** – Market event summarization and strategy recommendations powered by OpenAI.
- **Frontend/UI** – Fully interactive dashboards built with Streamlit.
- **Authentication & Access Control** – Firebase-backed user management.
- **Cloud Deployment** – Production-ready for hosting and scaling.

This project shows how **financial computation, machine learning, and cloud application architecture** can be integrated into one cohesive system.

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
