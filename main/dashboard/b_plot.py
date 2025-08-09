import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from .b_backtest import extend_end_date
import pandas as pd
from typing import Dict
from matplotlib.dates import date2num, num2date
from datetime import timedelta



ai_refinement_option = st.session_state.get("ai_refinement_option", "Yes")

def plot_efficient_frontier(
    efficient_results,
    max_sharpe_idx,
    max_return_idx,
    min_volatility_idx,
    tickers,
    single_stock_portfolio_data,
    annual_rfr,
    custom_portfolio_data=None,
    dividend_portfolio_data=None  
):
    """
    Plot the Efficient Frontier with max Sharpe, max Return, min Volatility,
    optional Custom Portfolio, and optional Dividend-Focused Portfolio.

    Args:
        efficient_results (dict): Contains keys "volatilities", "returns", etc.
        max_sharpe_idx (int): Index of the max Sharpe ratio portfolio.
        max_return_idx (int): Index of the max return portfolio.
        min_volatility_idx (int): Index of the min volatility portfolio.
        tickers (list): List of stock tickers for single-stock data.
        single_stock_portfolio_data (pd.DataFrame): DataFrame with 'Stock', 'Return', 'Volatility'.
        annual_rfr (float): Annual risk-free rate.
        custom_portfolio_data (dict, optional): Custom portfolio metrics.
        dividend_portfolio_data (dict, optional): Dividend-focused portfolio metrics 
            (e.g., from dividend_focused_portfolio()).

    Returns:
        matplotlib.figure.Figure: Figure containing the Efficient Frontier plot.
    """
    ef_volatilities = efficient_results["volatilities"]
    ef_returns = efficient_results["returns"]

    if not ef_volatilities or not ef_returns:
        raise ValueError("Efficient frontier data is missing or incomplete.")

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Aesthetics for the figure
    fig.subplots_adjust(left=0.1, right=1.3, top=1.3, bottom=0.1)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1)

    # Plot the efficient frontier line
    ax.plot(
        ef_volatilities,
        ef_returns,
        color="blue",
        linewidth=2,
        linestyle="--",
        alpha=1,
        label="Eff. Frontier",
        zorder=1
    )

    simulation_data = st.session_state.simulation_data
    metrics_dict = simulation_data.get("ai_portfolio_metrics")

    if ai_refinement_option == "Yes" and metrics_dict:
        # --- Balanced uses AI-refined values ---
        balanced_vol = metrics_dict["balanced"]["volatility"]
        balanced_ret = metrics_dict["balanced"]["expected_return"]

        # --- Aggressive & Conservative fall back to MVO points ---
        aggressive_vol, aggressive_ret   = ef_volatilities[max_return_idx],     ef_returns[max_return_idx]
        conservative_vol, conservative_ret = ef_volatilities[min_volatility_idx], ef_returns[min_volatility_idx]
    else:
        # Pure MVO for all three
        balanced_vol, balanced_ret       = ef_volatilities[max_sharpe_idx],     ef_returns[max_sharpe_idx]
        aggressive_vol, aggressive_ret   = ef_volatilities[max_return_idx],     ef_returns[max_return_idx]
        conservative_vol, conservative_ret = ef_volatilities[min_volatility_idx], ef_returns[min_volatility_idx]


    # Consistent marker size for special portfolios
    fixed_marker_size = 800

    # Plot special portfolio points
    # Balanced (max Sharpe)
    ax.scatter(
        balanced_vol,
        balanced_ret,
        color='#f5cd05',
        s=fixed_marker_size,
        marker='*',
        label="Balanced Portfolio",
        zorder=6
    )

    # Aggressive (max Return)
    ax.scatter(
        aggressive_vol,
        aggressive_ret,
        color='#8321ff',
        s=fixed_marker_size,
        marker='*',
        label="Aggressive Portfolio",
        zorder=5
    )

    # Conservative (min Volatility)
    ax.scatter(
        conservative_vol,
        conservative_ret,
        color='#18af28',
        s=fixed_marker_size,
        marker='*',
        label="Conservative Portfolio",
        zorder=5
    )

    # Compute dynamic increments based on the overall efficient frontier range.
    x_range = max(ef_volatilities) - min(ef_volatilities)
    y_range = max(ef_returns) - min(ef_returns)

    # Define multipliers for the connector line and label offsets.
    line_offset_x_factor = 0.02  # connector length relative to x_range
    line_offset_y_factor = 0.02  # connector length relative to y_range
    label_offset_x_factor = 0.005  # additional offset for label relative to x_range
    label_offset_y_factor = 0.005  # additional offset for label relative to y_range

    # Plot single-stock data with dynamic connector lines and label positions (no bbox)
    colors = plt.cm.tab10.colors
    for i, (_, row) in enumerate(single_stock_portfolio_data.iterrows()):
        stock_color = colors[i % len(colors)]
        # Plot the single stock data point.
        ax.scatter(
            row['Volatility'],
            row['Return'],
            color=stock_color,
            s=90,
            marker='o',
            zorder=3
        )

        # Calculate dynamic offsets for the connector line.
        line_offset_x = line_offset_x_factor * x_range
        line_offset_y = -line_offset_y_factor * y_range  # negative for downward slope
        line_end_x = row['Volatility'] + line_offset_x
        line_end_y = row['Return'] + line_offset_y

        # Draw the connector line.
        ax.plot(
            [row['Volatility'], line_end_x],
            [row['Return'], line_end_y],
            color=stock_color,
            linewidth=1.5,
            zorder=2
        )

        # Calculate label position using additional dynamic offset.
        label_x = line_end_x + label_offset_x_factor * x_range
        label_y = line_end_y + label_offset_y_factor * y_range

        # Place the label text without a bbox.
        ax.text(
            label_x,
            label_y,
            row['Stock'],
            fontsize=16,
            fontweight='normal',
            ha='left',
            va='top',
            zorder=4,
            color=stock_color
        )

    # Plot the custom portfolio (if provided)
    if custom_portfolio_data:
        ax.scatter(
            custom_portfolio_data["c_portfolio_volatility"],
            custom_portfolio_data["c_expected_return"],
            color='red',
            s=fixed_marker_size,
            marker='*',
            label="Custom Portfolio",
            zorder=7
        )

    # Plot the dividend-focused portfolio (if provided)
    if dividend_portfolio_data:
        ax.scatter(
            dividend_portfolio_data["d_volatility"],
            dividend_portfolio_data["d_expected_return"],
            color='#f697ff',  # A distinct color for Dividend-Focused
            s=fixed_marker_size,
            marker='*',
            label="Dividend-Focused Portfolio",
            zorder=7
        )

    # Add grid and axis labels with bounding boxes (inverted colors)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel(
        "Volatility",
        fontsize=25,
        labelpad=15,
        color="#0a4daa",  # Inverted text color
        fontweight='600',
        bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0')
    )
    ax.set_ylabel(
        "Return",
        fontsize=25,
        labelpad=5,
        color="#0a4daa",  # Inverted text color
        fontweight='600',
        bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0')
    )
    ax.tick_params(axis='both', labelsize=20)

    # Bold the tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontweight('500')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('500')
    
    # Adjust the border (axes spines): only show left and bottom borders.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor('#0a4daa')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_edgecolor('#0a4daa')
    ax.spines['bottom'].set_linewidth(1)
    
    # Format x and y axis tick labels as percentages.
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))

    # Capture current x, y limits.
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()


    # Compute the original MVO max Sharpe values
    mvo_balanced_vol = ef_volatilities[max_sharpe_idx]
    mvo_balanced_ret = ef_returns[max_sharpe_idx]

    # Capital Market Line (from balanced/max Sharpe portfolio)
    tangent_vol = mvo_balanced_vol
    tangent_ret = mvo_balanced_ret
    slope = (tangent_ret - annual_rfr) / tangent_vol if tangent_vol != 0 else 0
    extended_x_max = max(x_lim[1], max(ef_volatilities) * 1.2)
    line_x = [0, extended_x_max]
    line_y = [annual_rfr + slope * x for x in line_x]

    ax.plot(
        line_x,
        line_y,
        color='grey',
        linewidth=2,
        linestyle=':',
        alpha=1,
        label="Tangent Line"
    )

    # Restore axis limits.
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return fig




def plot_min_volatility_allocation(tickers, min_volatility_weights):
    """
    Plot the Minimum Volatility Portfolio allocation as a donut chart.

    This version displays the top 5 allocations and groups the Other 
    allocations into a "Other" slice so that the overall allocation percentages 
    remain intact.

    Parameters:
        tickers (list): List of stock tickers.
        min_volatility_weights (list): Weights for the minimum volatility portfolio.

    Returns:
        matplotlib.figure.Figure: Donut chart visualization for the minimum volatility portfolio.
    """


    from matplotlib.patches import Patch

    # Filter out assets with allocations <= 0.001
    filtered_tickers = []
    filtered_weights = []
    for ticker, weight in zip(tickers, min_volatility_weights):
        if weight > 0.001:
            filtered_tickers.append(ticker)
            filtered_weights.append(weight)

    if not filtered_tickers:
        raise ValueError("All assets have allocations below the threshold; nothing to plot.")

    # Combine and sort the tickers and weights in descending order by weight
    allocation = list(zip(filtered_tickers, filtered_weights))
    allocation_sorted = sorted(allocation, key=lambda x: x[1], reverse=True)

    # Group Other if more than 5 assets are available
    if len(allocation_sorted) > 5:
        top5 = allocation_sorted[:5]
        others = allocation_sorted[5:]
        others_weight = sum(w for _, w in others)
        new_allocation = top5 + [("Other", others_weight)]
    else:
        new_allocation = allocation_sorted

    # Unzip the allocation into tickers and weights
    final_tickers, final_weights = zip(*new_allocation)
    final_tickers = list(final_tickers)
    final_weights = list(final_weights)

    # Calculate percentages using the total of the original filtered weights
    total = sum(filtered_weights)
    percentages = [w / total for w in final_weights]

    # Define rainbow color palette and adjust colors
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(final_tickers)))
    def adjust_color(color, factor=0.85):
        return tuple([min(1, max(0, c * factor)) for c in color])
    adjusted_colors = [adjust_color(color) for color in colors]

    # Define explode values and create the donut chart
    explode = [0.05] * len(final_tickers)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    wedges, _ = ax.pie(
        percentages,
        labels=None,
        startangle=140,
        colors=adjusted_colors,
        explode=explode,
        wedgeprops=dict(width=0.7, edgecolor='white')
    )

    # Temporary adjustment to push Donut Chart Down 
    ax.set_title("   ", pad=65)



    # Add labels with leader lines
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={angle}"
        ax.annotate(
            f"{final_tickers[i]}: {percentages[i]*100:.1f}%",
            xy=(x, y),
            xytext=(1.3 * np.sign(x), 1.3 * y),
            horizontalalignment=horizontalalignment,
            arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle, color='#0a4daa'),
            fontsize=27,
            color='#0a4daa'
        )

    # Create and position the legend
    legend_patches = [
        Patch(facecolor=adjusted_colors[i], edgecolor='white', label=final_tickers[i])
        for i in range(len(final_tickers))
    ]
    fig.legend(
        handles=legend_patches,
        labels=final_tickers,
        loc='lower center',
        ncol=4,
        fontsize=30,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05)
    )
    plt.subplots_adjust(bottom=0.3, top=1.4, left=0.05, right=0.95)
    return fig














def plot_max_sharpe_allocation(tickers, weights_sharpe):
    """
    Plot the Maximum Sharpe Ratio Portfolio allocation as a donut chart.

    This version displays the top 5 allocations and groups the Other 
    allocations into a "Other" slice so that the overall allocation percentages 
    remain intact.

    Parameters:
        tickers (list): List of stock tickers.
        weights_sharpe (list): Weights for the Maximum Sharpe Ratio Portfolio.

    Returns:
        matplotlib.figure.Figure: Donut chart visualization for the Sharpe Ratio Portfolio.
    """

    from matplotlib.patches import Patch

    # Filter out tickers with allocations <= 0.001
    filtered_tickers = []
    filtered_weights_sharpe = []
    for stock, sharpe in zip(tickers, weights_sharpe):
        if sharpe > 0.001:
            filtered_tickers.append(stock)
            filtered_weights_sharpe.append(sharpe)

    if not filtered_tickers:
        raise ValueError("All assets have allocations below the threshold; nothing to plot.")

    # Combine and sort in descending order by weight
    allocation = list(zip(filtered_tickers, filtered_weights_sharpe))
    allocation_sorted = sorted(allocation, key=lambda x: x[1], reverse=True)

    # Group Other if more than 5 assets are available
    if len(allocation_sorted) > 5:
        top5 = allocation_sorted[:5]
        others = allocation_sorted[5:]
        others_weight = sum(w for _, w in others)
        new_allocation = top5 + [("Other", others_weight)]
    else:
        new_allocation = allocation_sorted

    final_tickers, final_weights_sharpe = zip(*new_allocation)
    final_tickers = list(final_tickers)
    final_weights_sharpe = list(final_weights_sharpe)

    total = sum(filtered_weights_sharpe)
    percentages = [w / total for w in final_weights_sharpe]

    # Define rainbow color palette and adjust colors
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(final_tickers)))
    def adjust_color(color, factor=0.85):
        return tuple([min(1, max(0, c * factor)) for c in color])
    adjusted_colors = [adjust_color(color) for color in colors]

    explode = [0.05] * len(final_tickers)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    wedges, _ = ax.pie(
        percentages,
        labels=None,
        startangle=140,
        colors=adjusted_colors,
        explode=explode,
        wedgeprops=dict(width=0.7, edgecolor='white')
    )



    # Temporary adjustment to push Donut Chart Down 
    ax.set_title("   ", pad=65)

    # Add labels with leader lines
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={angle}"
        ax.annotate(
            f"{final_tickers[i]}: {percentages[i]*100:.1f}%",
            xy=(x, y),
            xytext=(1.3 * np.sign(x), 1.3 * y),
            horizontalalignment=horizontalalignment,
            arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle, color='#0a4daa'),
            fontsize=27,
            color='#0a4daa'
        )

    # Create and position the legend
    legend_patches = [
        Patch(facecolor=adjusted_colors[i], edgecolor='white', label=final_tickers[i])
        for i in range(len(final_tickers))
    ]
    fig.legend(
        handles=legend_patches,
        labels=final_tickers,
        loc='lower center',
        ncol=4,
        fontsize=30,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05)
    )
    plt.subplots_adjust(bottom=0.3, top=1.4, left=0.05, right=0.95)
    return fig









def plot_max_return_allocation(tickers, weights_return):
    """
    Plot the Maximum Expected Return Portfolio allocation as a donut chart.

    This version displays the top 5 allocations and groups the Other 
    allocations into a "Other" slice so that the overall allocation percentages 
    remain intact.

    Parameters:
        tickers (list): List of stock tickers.
        weights_return (list): Weights for the Maximum Expected Return Portfolio.

    Returns:
        matplotlib.figure.Figure: Donut chart visualization for the Expected Return Portfolio.
    """

    from matplotlib.patches import Patch

    filtered_tickers = []
    filtered_weights_return = []
    for stock, ret in zip(tickers, weights_return):
        if ret > 0.001:
            filtered_tickers.append(stock)
            filtered_weights_return.append(ret)

    if not filtered_tickers:
        raise ValueError("All assets have allocations below the threshold; nothing to plot.")

    allocation = list(zip(filtered_tickers, filtered_weights_return))
    allocation_sorted = sorted(allocation, key=lambda x: x[1], reverse=True)
    
    if len(allocation_sorted) > 5:
        top5 = allocation_sorted[:5]
        others = allocation_sorted[5:]
        others_weight = sum(w for _, w in others)
        new_allocation = top5 + [("Other", others_weight)]
    else:
        new_allocation = allocation_sorted

    final_tickers, final_weights_return = zip(*new_allocation)
    final_tickers = list(final_tickers)
    final_weights_return = list(final_weights_return)

    total = sum(filtered_weights_return)
    percentages = [w / total for w in final_weights_return]

    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(final_tickers)))
    def adjust_color(color, factor=0.85):
        return tuple([min(1, max(0, c * factor)) for c in color])
    adjusted_colors = [adjust_color(color) for color in colors]

    explode = [0.05] * len(final_tickers)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    wedges, _ = ax.pie(
        percentages,
        labels=None,
        startangle=140,
        colors=adjusted_colors,
        explode=explode,
        wedgeprops=dict(width=0.7, edgecolor='white')
    )
    # Temporary adjustment to push Donut Chart Down 
    ax.set_title("   ", pad=65)


    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={angle}"
        ax.annotate(
            f"{final_tickers[i]}: {percentages[i]*100:.1f}%",
            xy=(x, y),
            xytext=(1.3 * np.sign(x), 1.3 * y),
            horizontalalignment=horizontalalignment,
            arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle, color='#0a4daa'),
            fontsize=27,
            color='#0a4daa'
        )

    legend_patches = [
        Patch(facecolor=adjusted_colors[i], edgecolor='white', label=final_tickers[i])
        for i in range(len(final_tickers))
    ]
    fig.legend(
        handles=legend_patches,
        labels=final_tickers,
        loc='lower center',
        ncol=4,
        fontsize=30,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05)
    )
    plt.subplots_adjust(bottom=0.3, top=1.4, left=0.05, right=0.95)
    return fig






def plot_dividend_allocation(tickers, weights_dividend):
    """
    Plot the Dividend-Focused Portfolio allocation as a donut chart.

    This version displays the top 5 allocations and groups the Other 
    allocations into a "Other" slice so that the overall allocation percentages 
    remain intact.

    Parameters:
        tickers (list): List of stock tickers.
        weights_dividend (list or array): Weights for the Dividend-Focused Portfolio.

    Returns:
        matplotlib.figure.Figure: Donut chart visualization for the Dividend-Focused Portfolio.
    """
  
    from matplotlib.patches import Patch

    filtered_tickers = []
    filtered_weights = []
    for stock, weight in zip(tickers, weights_dividend):
        if weight > 0.001:
            filtered_tickers.append(stock)
            filtered_weights.append(weight)

    if not filtered_tickers:
        raise ValueError("All assets have allocations below the threshold; nothing to plot.")

    allocation = list(zip(filtered_tickers, filtered_weights))
    allocation_sorted = sorted(allocation, key=lambda x: x[1], reverse=True)
    
    if len(allocation_sorted) > 5:
        top5 = allocation_sorted[:5]
        others = allocation_sorted[5:]
        others_weight = sum(w for _, w in others)
        new_allocation = top5 + [("Other", others_weight)]
    else:
        new_allocation = allocation_sorted

    final_tickers, final_weights = zip(*new_allocation)
    final_tickers = list(final_tickers)
    final_weights = list(final_weights)

    total = sum(filtered_weights)
    percentages = [w / total for w in final_weights]

    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(final_tickers)))
    def adjust_color(color, factor=0.85):
        return tuple([min(1, max(0, c * factor)) for c in color])
    adjusted_colors = [adjust_color(color) for color in colors]

    explode = [0.05] * len(final_tickers)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    wedges, _ = ax.pie(
        percentages,
        labels=None,
        startangle=140,
        colors=adjusted_colors,
        explode=explode,
        wedgeprops=dict(width=0.7, edgecolor='white')
    )

    # Temporary adjustment to push Donut Chart Down 
    ax.set_title("   ", pad=65)

    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={angle}"
        ax.annotate(
            f"{final_tickers[i]}: {percentages[i]*100:.1f}%",
            xy=(x, y),
            xytext=(1.3 * np.sign(x), 1.3 * y),
            horizontalalignment=horizontalalignment,
            arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle, color='#0a4daa'),
            fontsize=27,
            color='#0a4daa'
        )

    legend_patches = [
        Patch(facecolor=adjusted_colors[i], edgecolor='white', label=final_tickers[i])
        for i in range(len(final_tickers))
    ]
    fig.legend(
        handles=legend_patches,
        labels=final_tickers,
        loc='lower center',
        ncol=4,
        fontsize=30,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05)
    )
    plt.subplots_adjust(bottom=0.3, top=1.4, left=0.05, right=0.95)
    return fig






def plot_custom_portfolio_allocation(tickers, custom_asset_weights):
    """
    Plot the Custom Portfolio allocation as a donut chart.

    This version displays the top 5 allocations and groups the Other 
    allocations into a "Other" slice so that the overall allocation percentages 
    remain intact.

    Parameters:
        custom_asset_weights (dict): Weights for the custom portfolio.
        tickers (list): List of stock tickers (used to retrieve company names if available).

    Returns:
        matplotlib.figure.Figure: Donut chart visualization for the custom portfolio.
    """
   
    from matplotlib.patches import Patch

    filtered_tickers = []
    filtered_weights = []
    filtered_company_names = []

    for ticker, weight in custom_asset_weights.items():
        if weight > 0.001:
            filtered_tickers.append(ticker)
            filtered_weights.append(weight)
            try:
                index = tickers.index(ticker)
                filtered_company_names.append(tickers[index])
            except ValueError:
                filtered_company_names.append(ticker)

    if not filtered_tickers:
        raise ValueError("All assets have allocations below the threshold; nothing to plot.")

    # Group the top 5 allocations and aggregate Other as "Other"
    allocation = list(zip(filtered_tickers, filtered_weights, filtered_company_names))
    allocation_sorted = sorted(allocation, key=lambda x: x[1], reverse=True)
    
    if len(allocation_sorted) > 5:
        top5 = allocation_sorted[:5]
        others = allocation_sorted[5:]
        others_weight = sum(x[1] for x in others)
        new_allocation = top5 + [("Other", others_weight, "Other")]
    else:
        new_allocation = allocation_sorted

    final_tickers, final_weights, final_company_names = zip(*new_allocation)
    final_tickers = list(final_tickers)
    final_weights = list(final_weights)
    final_company_names = list(final_company_names)

    total = sum(filtered_weights)
    percentages = [w / total for w in final_weights]

    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(final_tickers)))
    def adjust_color(color, factor=0.85):
        return tuple([min(1, max(0, c * factor)) for c in color])
    adjusted_colors = [adjust_color(color) for color in colors]

    explode = [0.05] * len(final_tickers)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    wedges, _ = ax.pie(
        percentages,
        labels=None,
        startangle=140,
        colors=adjusted_colors,
        explode=explode,
        wedgeprops=dict(width=0.7, edgecolor='white')

    )
    # Temporary adjustment to push Donut Chart Down 
    ax.set_title("   ", pad=65)


    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={angle}"
        ax.annotate(
            f"{final_company_names[i]}: {percentages[i]*100:.1f}%",
            xy=(x, y),
            xytext=(1.3 * np.sign(x), 1.3 * y),
            horizontalalignment=horizontalalignment,
            arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle, color='#0a4daa'),
            fontsize=27,
            color='#0a4daa'
        )

    legend_patches = [
        Patch(facecolor=adjusted_colors[i], edgecolor='white', label=final_company_names[i])
        for i in range(len(final_tickers))
    ]
    fig.legend(
        handles=legend_patches,
        labels=final_company_names,
        loc='lower center',
        ncol=4,
        fontsize=30,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05)
    )
    plt.subplots_adjust(bottom=0.3, top=1.4, left=0.05, right=0.95)
    return fig















# b_plot.py
from datetime import timedelta
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import pandas as pd

def plot_portfolio_vs_benchmark(
    port_cum:   pd.Series,
    bench_cum:  pd.Series,
    labels:     Dict[str, str],
    styles:     Dict[str, Dict[str, object]],
    figsize:    Tuple[int, int] = (10, 6),
    scale:      float = 1.0
) -> plt.Figure:
    """
    Draw cumulative-return lines with endpoint dots + % labels.

    styles = {
        "portfolio": {
            "line":      "#00aa00",
            "dot":       "#00aa00",   # dot/edge
            "box":       "#18af28",   # text-box fill
            "box_alpha": 0.9,
            "dot_size":  35
        },
        "benchmark": { … }
    }
    """

    # --- prepare axis range --------------------------------------------------
    extended_idx = extend_end_date(port_cum.index)

    fig, ax = plt.subplots(figsize=figsize)

    # --- lines ---------------------------------------------------------------
    for key, series in (("portfolio", port_cum), ("benchmark", bench_cum)):
        ax.plot(series.index, series.values,
                label=labels[key],
                linewidth=(2 if key=="portfolio" else 1.5) * scale,
                linestyle="--" if key=="benchmark" else "-",
                color=styles[key]["line"],
                alpha=0.5 if key=="benchmark" else 1.0,
                zorder=4 if key=="portfolio" else 3)

    # --- endpoint dots -------------------------------------------------------
    for key, series in (("portfolio", port_cum), ("benchmark", bench_cum)):
        ax.scatter(series.index[-1], series.values[-1],
                   color     = styles[key]["dot"],
                   edgecolor = styles[key]["dot"],
                   s         = styles[key]["dot_size"] * scale,
                   linewidth = 0.5 * scale,
                   zorder    = 5,
                   alpha     = 0.7)

    # --- % text boxes --------------------------------------------------------
    span   = date2num(extended_idx[-1]) - date2num(port_cum.index[0])
    offset = timedelta(days=span * 0.025)

    for key, series in (("portfolio", port_cum), ("benchmark", bench_cum)):
        pct = float(series.iloc[-1]) * 100            # <-- cast to float!
        ax.text(series.index[-1] + offset,
                series.values[-1],
                f"{pct:.2f}%",
                fontsize   = 11 * scale,
                color      = "white",
                fontweight = "700" if key=="portfolio" else "500",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor = styles[key]["box"],
                          alpha     = styles[key]["box_alpha"],
                          edgecolor = "none"))

    # --- cosmetics -----------------------------------------------------------
    ax.set_xlim(port_cum.index[0], extended_idx[-1])
    ax.set_ylabel("Portfolio Returns", fontsize=18*scale, color="#0a4daa")
    ax.legend(prop={"size":14*scale, "weight":600})
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5*scale)
    for spine in ax.spines.values():
        spine.set_color("#4d4d4d")
        spine.set_linewidth(0.3*scale)
    ax.tick_params(axis="both", labelsize=12*scale)

    return fig
