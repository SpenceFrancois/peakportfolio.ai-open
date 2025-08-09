# peakportfolio.ai-open
Open source AI powered investment portfolio analysis and allocation platform.

PeakPortfolio.ai is an AI-powered portfolio analysis and optimization platform that combines quantitative finance, machine learning, and intuitive visualization to help investors make better decisions.
This repository contains the open-source version of the core application, allowing you to run it locally, explore the algorithms, and extend the platform.

🚀 Overview
PeakPortfolio.ai integrates:

Quantitative finance models – Efficient frontier optimization, Sharpe/Sortino ratios, max drawdown, dividend yield strategies, and custom portfolio analysis.

AI-driven insights – OpenAI integration for contextual portfolio commentary and market event analysis.

Live market data – Pulls from APIs like Tiingo and Yahoo Finance for up-to-date returns, volatility, and news.

Secure user authentication – Firebase authentication for account management and subscription handling.

Modern web app – Built with Streamlit for rapid UI development and rich, interactive dashboards.

This project was built end-to-end by a single developer over 8 months, covering:

Backend architecture and API integration

Quant modeling & algorithm design

Frontend/UI implementation

Deployment and cloud infrastructure

Data pipeline creation and caching strategies

📦 Features
Portfolio Optimization – Mean-variance, dividend-focused, and AI-refined strategies.

Risk Metrics – Sharpe, Sortino, downside deviation, and max drawdown calculations.

Custom Portfolios – Analyze your own allocations and simulate performance.

Market Event Summaries – News scraping and summarization.

Pro Mode (Disabled Here) – Certain production features are behind paywalls in the live app, but the open-source version provides a complete working environment for local use and experimentation.

🛠️ Installation
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/<your-username>/peakportfolio.ai-open.git
cd peakportfolio.ai-open
2. Create and activate a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Set up environment variables
Create a .env file in the root directory:

ini
Copy
Edit
TIINGO_API_KEY=your_tiingo_key
OPENAI_API_KEY=your_openai_key
FIREBASE_CREDENTIALS_PATH=path_to_your_firebase.json
Note:
API keys are not included in this repository. You must generate your own from the respective services.

5. Run locally
bash
Copy
Edit
streamlit run app.py
📄 License
This project is licensed under the [Custom License] – you may use, modify, and distribute, but commercial resale is prohibited. See LICENSE for details.

💡 Acknowledgements
Tiingo for market data

Yahoo Finance

Streamlit for the UI framework

OpenAI for language model integration

Firebase for authentication and database

🌐 Live Version
The full commercial version is available at PeakPortfolio.ai.

If you want, I can now also add a “Why this matters” section at the top that explains the 8 months, one mind context so a recruiter or investor instantly sees the magnitude of the solo build. That will make this readme not just instructional, but memorable.

Do you want me to add that part?
