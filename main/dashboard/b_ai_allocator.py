import streamlit as st
import os
import json
import pandas as pd             
import numpy as np
import threading
from openai import OpenAI
from dashboard.b_optim import calculate_ai_refined_portfolio_data
from datetime import datetime




# Instantiate the AI client for portfolio refinement
portfolio_refinement_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


_MAX_EVENTS = 100  # optional: keep only the latest N events

def _log(sd: dict, role: str, payload: dict | str, msg: str = "") -> None:
    """
    Append one event to simulation_data['ai_logs'].
    Event = {ts, role, msg, data}.  Trims buffer to _MAX_EVENTS.
    """
    buf = sd.setdefault("ai_logs", [])
    buf.append(
        {
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "role": role,
            "msg": msg,
            "data": payload,
        }
    )
    if len(buf) > _MAX_EVENTS:
        del buf[0 : len(buf) - _MAX_EVENTS]




def build_allocations(weights, tickers, company_names):
    return [
        {
            "ticker": ticker,
            "company": company,
            "weight": round(weight, 3)
        }
        for ticker, company, weight in zip(tickers, company_names, weights)
    ]


def get_portfolio_data():
    if "simulation_data" not in st.session_state:
        st.error("Simulation data not found. Run the portfolio optimization first.")
        return None

    simulation_data = st.session_state["simulation_data"]

    tickers = simulation_data.get("tickers", [])
    company_names = simulation_data.get("company_names", [])

    # ─────── Build allocations with rounded weights (handled inside build_allocations) ───────
    conservative_allocations = build_allocations(
        simulation_data["min_volatility_weights"],
        tickers,
        company_names,
    )
    balanced_allocations = build_allocations(
        simulation_data["max_sharpe_weights"],
        tickers,
        company_names,
    )
    aggressive_allocations = build_allocations(
        simulation_data["max_return_weights"],
        tickers,
        company_names,
    )

    # ─────── Helper to round metric values ───────
    def round_metric(value):
        return round(value, 3) if isinstance(value, (int, float)) else value

    # ─────── Assemble portfolio data with rounded metrics ───────
    portfolio_data = {
        "conservative": {
            "allocations": conservative_allocations,
            "metrics": {
                "return":         round_metric(simulation_data["min_volatility_return"]),
                "volatility":     round_metric(simulation_data["min_volatility_volatility"]),
                "sharpe_ratio":   round_metric(simulation_data["min_volatility_ratio"]),
                "sortino_ratio":  round_metric(simulation_data["min_volatility_sortino"]),
                "dividend_yield": round_metric(simulation_data["min_volatility_yield"]),
                "max_drawdown":   round_metric(simulation_data.get("min_volatility_max_drawdown", 0)),
            },
        },
        "balanced": {
            "allocations": balanced_allocations,
            "metrics": {
                "return":         round_metric(simulation_data["max_sharpe_return"]),
                "volatility":     round_metric(simulation_data["max_sharpe_volatility"]),
                "sharpe_ratio":   round_metric(simulation_data["max_sharpe_ratio"]),
                "sortino_ratio":  round_metric(simulation_data["max_sharpe_sortino"]),
                "dividend_yield": round_metric(simulation_data["max_sharpe_yield"]),
                "max_drawdown":   round_metric(simulation_data.get("max_sharpe_max_drawdown", 0)),
            },
        },
        "aggressive": {
            "allocations": aggressive_allocations,
            "metrics": {
                "return":         round_metric(simulation_data["max_return_return"]),
                "volatility":     round_metric(simulation_data["max_return_volatility"]),
                "sharpe_ratio":   round_metric(simulation_data["max_return_ratio"]),
                "sortino_ratio":  round_metric(simulation_data["max_return_sortino"]),
                "dividend_yield": round_metric(simulation_data["max_return_yield"]),
                "max_drawdown":   round_metric(simulation_data.get("max_return_max_drawdown", 0)),
            },
        },
    }

    # ─────── Add single-stock data (rounded) ───────
    single_stock_df = simulation_data.get("computed_single_stock_data")
    if single_stock_df is not None and not single_stock_df.empty:
        single_stock_df = single_stock_df.round(3)  # Round all floats in DataFrame
        portfolio_data["single_stock_portfolios"] = json.loads(
            single_stock_df.to_json(orient="records")
        )

    return portfolio_data

# ------------------------------------------------------------------
# 1. Thin wrapper for OpenAI chat completions
# ------------------------------------------------------------------
def _chat(prompt: str) -> str:
    response = portfolio_refinement_client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "system", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

# ------------------------------------------------------------------
# 2. JSON loader that never crashes the pipeline
# ------------------------------------------------------------------
def _safe_json(text: str, fallback):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback

# ------------------------------------------------------------------
# 3. Prompt builders – each agent gets full context and rules
# ------------------------------------------------------------------

def _validate_explainer_output(
    explanations: list[dict],
    tickers: list[str],
    weights: list[float],
    tolerance: float = 1e-5,
) -> bool:
    """
    Ensures the explainer output matches the tickers and weights provided by the Allocator.
    """
    if len(explanations) != len(tickers):
        return False

    for i, (exp, expected_ticker, expected_weight) in enumerate(zip(explanations, tickers, weights)):
        if exp.get("ticker") != expected_ticker:
            return False
        if not isinstance(exp.get("weight"), (float, int)):
            return False
        if abs(exp["weight"] - expected_weight) > tolerance:
            return False

    return True




def _build_allocator_prompt(
    ai_context_input: str,
    portfolio_data: dict,
    ticker_list: list[str],
    manager_feedback: str | None = None,
    previous_weights: list[float] | None = None,
):
    feedback_block = (
        f"\n[MANAGER FEEDBACK]\n{manager_feedback}\n" if manager_feedback else ""
    )

    previous_block = (
        f"\n[PREVIOUS PROPOSAL]\nLast attempted weights (same order as TICKERS):\n{previous_weights}\n"
        if previous_weights else ""
    )

    return f"""
{feedback_block}{previous_block}
[ROLE DESCRIPTION]  
You are the **Allocator Agent**, acting as a disciplined portfolio manager.
Your only task is to determine and output a JSON list of asset weights.
Every allocation directly impacts clients' financial outcomes.
Always make precise allocations focused on strictly satisfying the hard rules and the spirit of the client's context.
Do **not** explain anything; your output must be only the final allocation.

[UNIVERSAL RULES]
1. Weights must sum to 1.
2. Do NOT introduce new tickers.
3. If a rule ever conflicts with ai_context_input, the context wins—unless it violates Rule 1 or Rule 2.

[CAPS]
• ETFs must be ≥ 60 % unless the user context explicitly overrides.  
• Any single ticker must be ≤ 10 % (≤ 8 % if the user is low-risk).  
• Crypto must be ≤ 5 % unless the user asks for more (hard cap 10 %).
•  Never give completely equal weights (e.g. 25%, 25%, 25%, 25%) unless the user explicitly asks for it.

[WORKFLOW]
1. Parse constraints from [USER CONTEXT]: separate hard caps (must be followed) from soft preferences (optimize toward).
2. Enforce [UNIVERSAL RULES]
3. Review MVO portfolios: identify the closest starting point based on client’s risk preferences and constraints.
4. Review single-stock portfolios for data that matches the user context.
5. Refine your proposal using the previous attempt (if available) and manager feedback.
6. Validate compliance: check total = 1.0 and individual caps.
7. Output only strict JSON, no extra text.

[TICKERS – ORDER IS FIXED]
{json.dumps(ticker_list)}

[USER CONTEXT]
{ai_context_input}

[MVO REFERENCE PORTFOLIOS]
{json.dumps(portfolio_data, indent=2)}

[OUTPUT FORMAT – STRICT JSON ONLY]
{{"weights": [0.00, ...]}}
"""


def _build_manager_prompt(
    proposed_weights: list[float],
    ai_context_input: str,
    portfolio_data: dict,
    computed_metrics: dict,
):
    return f"""
[ROLE DESCRIPTION]  
You are the **Manager Agent**, responsible for reviewing portfolio allocations to ensure they broadly reflect the user's intent and are directionally appropriate.  
Your goal is to **support progress, not block it**.  

- You must **approve** any allocation that is **reasonable**, even if it doesn't perfectly meet all numeric targets.  
- You must **not reject** a portfolio just because it doesn't hit an ambitious or unrealistic user request (e.g. 10% yield, 1000% return).  
- Only reject allocations if they are clearly broken, infeasible, or ignore the user's request entirely.  
- The user’s request is **context**, not gospel — assume it may be messy, contradictory, or emotionally charged.  
- If the proposal reflects a good-faith effort to balance competing priorities, **approve it**.  
- You may offer a recommendation, but do **not block approval** unless the allocation is fundamentally wrong (e.g. all gold, zero equity for a growth-focused user).  
- Never enforce hard numeric targets unless clearly specified as non-negotiable **and** feasible.  
- Do not nitpick small shortfalls (e.g. 2.4% yield when user asked for 3%). That is acceptable.  

Your bias should be toward approval. Rejections should be rare and only for serious misalignments.

[INPUT]
proposed_weights = {proposed_weights}
ai_context_input = {ai_context_input}
computed_metrics = {computed_metrics}
mvo_reference = {json.dumps(portfolio_data, indent=2)}

[OUTPUT – STRICT JSON ONLY]
{{
  "approved": true | false,
  "violations": ["..."], 
  "recommendation": "single sentence if not approved"
}}
"""




def _build_explainer_prompt(
    final_weights: list[float],
    ticker_list: list[str],
    company_name_list: list[str],
    ai_context_input: str,
):
    combined_payload = [
        {"ticker": t, "company_name": n, "weight": w}
        for t, n, w in zip(ticker_list, company_name_list, final_weights)
    ]
    return f"""
[ROLE DESCRIPTION] You are the **Elite Private Wealth Management Advisor (Explainer Agent)**. Your core function is to translate complex portfolio allocations into clear, concise, and compelling narratives for sophisticated clients. You don't just explain; you educate, reassure, and empower. Your communication should reflect the highest standards of financial acumen, client empathy, and strategic insight, comparable to the most trusted advisors in the private wealth sector.

[EXPLANATION RULES]
1. **Client-Centric Communication:** Always frame explanations from the client's perspective, addressing their goals, risk tolerance, and specific financial situation as outlined in the [USER CONTEXT]. Tailor the language to be easily digestible while maintaining professional rigor.
2. **Holistic Justification:** Provide an in-depth, multi-faceted justification for **each** weight in the portfolio. This must go beyond merely mirroring the weight; it should articulate the strategic rationale, expected contribution to the overall portfolio objectives, and alignment with market conditions and the client's financial plan.
3. **Forward-Looking Insights:** Include a concise, forward-looking perspective for each holding. This should briefly touch upon the expected future performance drivers, potential risks, and how the asset is positioned for long-term growth or stability within the portfolio.
4. **Professional & Empathetic Tone:** Maintain a tone that is authoritative yet approachable, professional yet genuinely friendly and empathetic. Avoid jargon where simpler terms suffice, but do not shy away from sophisticated financial language when precision is paramount.
5. **Clarity and Conciseness:** While in-depth, explanations must remain clear and concise. Avoid unnecessary verbosity. Every sentence should add value.
6. **Actionable Understanding:** The goal is for the client to not only understand *what* they own but *why* they own it and *how* it contributes to their financial well-being.
7. **Adherence to Fiduciary Standards:** Implicitly uphold principles of transparency, honesty, and acting in the client's best interest.

[FINAL PORTFOLIO]
{json.dumps(combined_payload, indent=2)}

[USER CONTEXT]
{ai_context_input}

[OUTPUT – STRICT JSON ONLY]
{{
  "explanations": [
    {{"ticker":"...", "weight":0.00, "explanation":"..."}},
    ...
  ]
}}
"""







# ------------------------------------------------------------------
# 4. Worker – mirrors legacy behaviour exactly
# ------------------------------------------------------------------
def _pipeline_worker(
    ai_context_input: str,
    portfolio_data: dict,
    simulation_data: dict,
):
    ticker_list       = simulation_data["tickers"]
    company_name_list = simulation_data["company_names"]
    manager_feedback: str | None = None
    previous_weights: list[float] | None = None


    # ── Allocator / Manager loop (max 3 tries) ───────────────────────────
    # ── Allocator / Manager loop (max 3 tries) ───────────────────────────
    for _ in range(3):
        # ---------- Allocator turn ----------
        allocator_prompt = _build_allocator_prompt(
            ai_context_input,
            portfolio_data,
            ticker_list,
            manager_feedback,
            previous_weights,  # ← new
        )



        allocator_raw = _chat(allocator_prompt)

        allocator_json = _safe_json(allocator_raw, {})
        proposed_weights = allocator_json.get("weights")
        previous_weights = proposed_weights  # ← store for next iteration

        if not proposed_weights or len(proposed_weights) != len(ticker_list):
            manager_feedback = "Weights missing or wrong length – JSON please."
            continue

        # ---------- metrics for Manager ----------
        loop_metrics = calculate_ai_refined_portfolio_data(
            ai_weights      = {"balanced": proposed_weights},
            mean_returns    = simulation_data["returns"].mean(),
            cov_matrix      = simulation_data["returns"].cov(),
            annual_rfr      = simulation_data["Annual Risk Free Rate"],
            returns_df      = simulation_data["returns"],
            dividend_yields = np.array(simulation_data["dividend_yields"]),

        )

        # ---------- Manager turn ----------
        manager_prompt = _build_manager_prompt(
            proposed_weights,
            ai_context_input,
            portfolio_data,
            loop_metrics,
        )


        manager_raw = _chat(manager_prompt)
    
        manager_json = _safe_json(
            manager_raw,
            {
                "approved": False,
                "violations": ["parse fail"],
                "recommendation": "Return valid JSON",
            },
        )

        if manager_json.get("approved"):
            break

        manager_feedback = manager_json.get("recommendation") or "Fix violations."

    else:
        # ── Manager never approved → fallback to raw MVO 'balanced' ─────────
        proposed_weights = [
            a["weight"] for a in portfolio_data["balanced"]["allocations"]
        ]

    # ── Explainer (runs for both approved and fallback paths) ──────────────
    explainer_json = _safe_json(
        _chat(_build_explainer_prompt(
            proposed_weights,  
            ticker_list, 
            company_name_list,
            ai_context_input,     
        )),
        {},                      # fallback if JSON parse fails
    )
    _log(simulation_data, "Explainer", {}, "client facig agent explanations completed...")
    explainer_output = explainer_json.get("explanations", [])

    if _validate_explainer_output(explainer_output, ticker_list, proposed_weights):
        final_explanations = explainer_output
    else:
        simulation_data["ai_error"] = (
            "Explainer output mismatch – fallback triggered."
        )
        final_explanations = [
            {
                "ticker": t,
                "weight": w,
                "explanation": "No explanation – fallback.",
            }
            for t, w in zip(ticker_list, proposed_weights)
        ]

    # ── Build DataFrame (legacy format) ────────────────────────────────────
    account_size = simulation_data["account_size"]
    ticker_info  = simulation_data.get("ticker_info", [{}] * len(ticker_list))
    refined_df   = pd.DataFrame({
        "Ticker":        ticker_list,
        "Asset Name":    company_name_list,
        "Asset Type":    [info.get("type", "Unknown") for info in ticker_info],
        "$ Allocation":  [f"{round(w * account_size):,}" for w in proposed_weights],
        "% Allocation":  proposed_weights,
    })

    # ── Optional metrics update ────────────────────────────────────────────
    if simulation_data.get("dividend_yields") is not None:
        final_metrics = calculate_ai_refined_portfolio_data(
            ai_weights      = {"balanced": proposed_weights},
            mean_returns    = simulation_data["returns"].mean(),
            cov_matrix      = simulation_data["returns"].cov(),
            annual_rfr      = simulation_data["Annual Risk Free Rate"],
            returns_df      = simulation_data["returns"],
            dividend_yields = np.array(simulation_data["dividend_yields"]),
        )
        simulation_data["ai_portfolio_metrics"] = final_metrics

    # ── Persist session-state keys ─────────────────────────────────────────
    simulation_data["ai_refined_weights"]       = {"balanced": proposed_weights}
    simulation_data["ai_refined_explanations"]  = {"balanced": final_explanations}
    simulation_data["ai_refined_portfolios"]    = {"balanced": refined_df}


# ------------------------------------------------------------------
# 5. Public Streamlit entry point – still non-blocking
# ------------------------------------------------------------------
def ai_portfolio_refinement() -> bool:
    portfolio_data = get_portfolio_data()
    if not portfolio_data:
        return False

    simulation_data  = st.session_state["simulation_data"]
    ai_context_input = simulation_data.get("ai_context_input", "")

    simulation_data.setdefault("ai_thread_started", False)
    simulation_data.setdefault("ai_done",           False)
    simulation_data.setdefault("ai_error",          None)

    if not simulation_data["ai_thread_started"]:
        simulation_data["ai_thread_started"] = True

        def _run_safe():
            try:
                _pipeline_worker(ai_context_input, portfolio_data, simulation_data)
            except Exception as exc:
                simulation_data["ai_error"] = str(exc)
            finally:
                simulation_data["ai_done"] = True
                simulation_data["ai_force_rerun"] = True 

        threading.Thread(target=_run_safe, daemon=True).start()

    return simulation_data.get("ai_done", False)

