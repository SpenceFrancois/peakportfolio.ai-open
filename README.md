# PeakPortfolio.ai – Open Source Edition

---

### What peakportfolio does

PeakPortfolio is an investment portfolio construction application that allows users to express investment goals, ideas, and their own analysis in natural language and receive a personalized portfolio that is constructed, reviewed, and explained through a structured, multi-stage workflow.

---

### The problem this is designed to solve

Modern portfolio construction draws on multiple forms of analysis and judgment, which are frequently executed through separate tools and workflows.

PeakPortfolio explores a unified approach to expressing, constructing, and evaluating portfolios in one structured system.

---

### How portfolio construction is approached

Portfolio construction begins with quantitative baselines derived from multiple MVO portfolios and single-asset portfolios. These baselines provide context for reconstructing portfolios around user defined intent expressed in natural language.

User intent, MVO portfolios, asset-level data, and explicit constraints are sent through a chain of three AI agents (Allocator, Manager, Explainer), each working on a specific component of portfolio construction, to produce portfolio allocations with performance metrics and explanations for each aspect of the portfolio.

---

### Human judgment and system support

PeakPortfolio is designed to support human portfolio decision-making, not to replace it. Users define the investment universe, objectives, constraints, and retain full discretion over portfolio decisions.

---

<br><br>

## Technical Overview
---
### 1. Input & Baselines

The process begins with:
- User-supplied asset tickers  
- A natural-language investment brief  
- Analysis start and end dates  
- Minimum and maximum portfolio weights  

A mean-variance optimization (MVO) engine then computes three reference portfolios:

- Minimum Volatility
- Maximum Sharpe Ratio  
- Maximum Return

These portfolios serve as quantitative performance baselines for downstream analysis and AI-driven portfolio construction.

<br>

### 2. Decision Structure & Separation of Responsibilities
Using a structure of multiple AI agents performing scoped tasks allows for efficient processing of both quantitative and qualitative context, including single-asset portfolios, MVO reference portfolios, and user intent expressed in natural language. 

Separating responsibilities across three agents reduces hallucination risk by limiting each agent’s scope to a specific function. This structure helps defend against hallucinated outputs, faulty formats, and invalid or arbitrary allocations. 

By constraining each stage of the process, portfolios can be analyzed, allocated, reviewed, and explained with greater intent, consistency, and control.

### Allocator Agent

Constructs a new portfolio based on the user context, asset list, and reference portfolios.  
Applies all allocation constraints to produce a valid portfolio (ETF floors, crypto caps, position limits).  
Proposes allocations, computes performance results, and turns the portfolio into a deliverable (JSON) for the Manager Agent.

---

### Manager Agent

Serves as compliance and risk oversight.  
Receives proposed allocations, user context, and computed portfolio metrics (expected return, volatility, Sharpe ratio, Sortino, dividend yield).  
Assesses whether the portfolio is compliant and financially appropriate.  
If violations occur, triggers a reallocation loop (up to three attempts).  
If no compliant solution is found, defaults to the MVO-based balanced portfolio.

---

### Explainer Agent

Delivers a client-ready narrative for each portfolio position.  
Interprets the portfolio in light of the user’s objectives and risk profile.  
Explains rationale, expected contribution, and forward-looking outlook for each asset.

---

### Finalization

Approved portfolio is re-evaluated through the quant engine.  
Final output includes allocations, performance metrics, and human-grade explanations.  
Delivered asynchronously to the user in under two minutes.

---
<br><br>

## Portfolio Outputs, Reporting & Data Export
---
### Portfolio Breakdown & Visualization

- **Pick A Portfolio**
  - Five interactive tabs displaying different portfolio breakdowns with:
    - Ticker  
    - Asset Name  
    - Asset Type  
    - Dollar Allocation  
    - Percent Allocation  

- **Allocation Snapshot**
  - Donut chart for top-weighted allocations  

- **Portfolio Summary**
  - Expected Return  
  - Sharpe Ratio  
  - Sortino Ratio  
  - Volatility  
  - Dividend Yield  
  - Max Drawdown  

- **Portfolio vs. Benchmark Comparison**
  - Performance graph comparing selected portfolios to the chosen benchmark  
  - Interactive crosshair enabled for detailed inspection  

---

### Reporting & Presentation Outputs

Every portfolio simulation automatically generates downloadable reports, including:
- Portfolio vs. Benchmark Comparison  
- Top Allocations  
- Portfolio Summary  

Reports support company logo integration for presentation-ready outputs.

---

### Data Analysis & Export

- **Risk-Reward Portfolio Table**
  - Displays sortable efficient portfolios with filters for:
    - Ticker  
    - Portfolio Allocation  
    - Dividend Yield  
    - Volatility  

- **Correlation Table**
  - Shows correlation between all portfolio assets  

- **Excel Data Export**
  - Users can download full simulation data in Excel format for independent analysis and manipulation, including:
    - Returns and cumulative returns  
    - Covariance and correlation matrices  
    - Efficient frontier data  
    - Portfolio-level returns  

---

## Core System Components

- **Backend Engineering** – APIs, data ingestion, risk and return calculations  
- **Quantitative Models** – Mean-variance optimization, Sortino and Sharpe analysis, dividend yield targeting, and max drawdown tracking  
- **AI Integration** – Market event summarization and strategy recommendations powered by OpenAI  
- **Frontend/UI** – Fully interactive dashboards built with Streamlit  
- **Authentication & Access Control** – Firebase-backed user management  
- **Cloud Deployment** – Production-ready for hosting and scaling  

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

##  Live Version
A hosted version of PeakPortfolio.ai is available here:  
[https://app.peakportfolio.ai](https://app.peakportfolio.ai)

---


##  Acknowledgements

- [Tiingo](https://www.tiingo.com/) – Financial market data
- [Yahoo Finance](https://pypi.org/project/yfinance/) – Historical data and news
- [OpenAI](https://openai.com/) – AI-powered market summaries and analysis
- [Streamlit](https://streamlit.io/) – Application framework
- [Firebase](https://firebase.google.com/) – Authentication and user management

---

##  Contributing

Contributions are welcome! Fork the repo, make your changes, and submit a pull request.

---

## 🧠 Author

**Creator:** Spencer Francois 
