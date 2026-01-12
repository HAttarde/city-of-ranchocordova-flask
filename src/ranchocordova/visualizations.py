"""
Unified Visualization Module for Rancho Cordova Agents
======================================================

Handles visualizations for:
1. Energy Agent - Forecast & Consumption Comparison
2. Customer Service Agent - Call Volume & Reason Distribution

Simple, efficient, reusable code.
"""

import json
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ============================================================================
# CORE VISUALIZATION DETECTOR
# ============================================================================


def detect_visualization_need(query: str, agent_type: str) -> tuple:
    """
    Detect if query needs visualization and what type.

    Returns: (needs_viz: bool, viz_type: str)
    viz_type can be: 'forecast', 'comparison', 'trend', 'distribution'
    """
    query_lower = query.lower()

    # Common visualization keywords
    viz_keywords = ["chart", "graph", "plot", "visual", "show", "display", "trend"]
    hourly_kw = ["hourly", "hour", "pattern", "daily pattern", "time of day"]
    peak_kw = ["peak", "off-peak", "time of use", "tou"]
    cost_kw = ["cost", "bill", "price", "rate", "pay", "charge"]
    benchmark_kw = ["benchmark", "compare utility", "smud vs", "pge", "average"]
    appliance_kw = ["appliance", "equipment", "what uses", "breakdown", "device"]
    renewable_kw = ["renewable", "solar", "power supply", "grid mix"]
    has_viz_keyword = any(kw in query_lower for kw in viz_keywords)

    if agent_type == "energy":
        # Energy-specific patterns
        forecast_kw = [
            "forecast",
            "predict",
            "future",
            "next",
            "upcoming",
            "weeks",
            "projection",
        ]
        comparison_kw = [
            "compare",
            "comparison",
            "difference",
            "vs",
            "versus",
            "by type",
            "residential commercial",
        ]

        if any(kw in query_lower for kw in forecast_kw):
            return (True, "forecast")
        elif any(kw in query_lower for kw in comparison_kw) or has_viz_keyword:
            return (True, "comparison")
        elif any(kw in query_lower for kw in hourly_kw):
            return (True, "hourly_pattern")
        elif any(kw in query_lower for kw in peak_kw):
            return (True, "peak_analysis")
        elif any(kw in query_lower for kw in cost_kw):
            return (True, "cost_analysis")
        elif any(kw in query_lower for kw in benchmark_kw):
            return (True, "benchmark")
        elif any(kw in query_lower for kw in appliance_kw):
            return (True, "appliance_breakdown")
        elif any(kw in query_lower for kw in renewable_kw):
            return (True, "renewable_mix")

    elif agent_type == "customer_service":
        # Customer service-specific patterns
        trend_kw = ["trend", "over time", "daily", "weekly", "volume", "calls per"]
        distribution_kw = [
            "breakdown",
            "distribution",
            "by reason",
            "types of",
            "category",
            "most common",
        ]

        if any(kw in query_lower for kw in trend_kw):
            return (True, "trend")
        elif any(kw in query_lower for kw in distribution_kw) or has_viz_keyword:
            return (True, "distribution")

    return (False, None)


# ============================================================================
# ENERGY VISUALIZATIONS
# ============================================================================


def generate_energy_forecast(df: pd.DataFrame, days: int = 14) -> dict:
    """
    Generate energy consumption forecast (Line Chart).

    Args:
        df: Energy dataframe with columns: CustomerID, AccountType, EnergyConsumption_kWh
        days: Number of days to forecast (default 14)

    Returns:
        {
            'title': str,
            'chart_type': 'line',
            'data': [{'date': str, 'consumption': float}, ...]
        }
    """
    # Calculate actual averages
    avg_residential = df[df["AccountType"] == "Residential"][
        "EnergyConsumption_kWh"
    ].mean()
    avg_commercial = df[df["AccountType"] == "Commercial"][
        "EnergyConsumption_kWh"
    ].mean()

    # Count customers
    res_count = len(df[df["AccountType"] == "Residential"])
    com_count = len(df[df["AccountType"] == "Commercial"])

    # City-wide baseline
    base_daily = (avg_residential * res_count) + (avg_commercial * com_count)

    # Generate forecast
    start_date = datetime.now() + timedelta(days=1)
    forecast_data = []

    np.random.seed(42)
    for i in range(days):
        date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        seasonal_factor = 1 + (i / days) * 0.1  # 10% seasonal increase
        noise = np.random.normal(0, base_daily * 0.05)
        value = base_daily * seasonal_factor + noise

        forecast_data.append({"date": date, "consumption": round(value, 2)})

    return {
        "title": f"{days}-Day Energy Consumption Forecast",
        "chart_type": "line",
        "data": forecast_data,
    }


def generate_energy_comparison(df: pd.DataFrame) -> dict:
    """
    Generate energy consumption comparison by account type (Bar Chart).

    Returns:
        {
            'title': str,
            'chart_type': 'bar',
            'data': [{'category': str, 'value': float, 'count': int}, ...]
        }
    """
    comparison = (
        df.groupby("AccountType")["EnergyConsumption_kWh"]
        .agg(["mean", "count"])
        .reset_index()
    )

    data = []
    for _, row in comparison.iterrows():
        data.append(
            {
                "category": row["AccountType"],
                "value": round(row["mean"], 2),
                "count": int(row["count"]),
            }
        )

    return {
        "title": "Average Energy Consumption by Account Type",
        "chart_type": "bar",
        "data": data,
    }


# ============================================================================
# CUSTOMER SERVICE VISUALIZATIONS
# ============================================================================


def generate_call_volume_trend(df: pd.DataFrame, days: int = 14) -> dict:
    """
    Generate call volume trend over time (Line Chart).

    Args:
        df: Customer service dataframe with DateTime column
        days: Number of recent days to show (default 14)

    Returns:
        {
            'title': str,
            'chart_type': 'line',
            'data': [{'date': str, 'calls': int}, ...]
        }
    """
    # Parse dates
    df["Date"] = pd.to_datetime(df["DateTime"]).dt.date

    # Count calls per day
    daily_calls = df.groupby("Date").size().reset_index(name="calls")
    daily_calls = daily_calls.sort_values("Date").tail(days)

    data = []
    for _, row in daily_calls.iterrows():
        data.append(
            {"date": row["Date"].strftime("%Y-%m-%d"), "calls": int(row["calls"])}
        )

    return {
        "title": f"Call Volume - Last {days} Days",
        "chart_type": "line",
        "data": data,
    }


def generate_call_reason_distribution(df: pd.DataFrame) -> dict:
    """
    Generate call reason distribution (Bar Chart).

    Returns:
        {
            'title': str,
            'chart_type': 'bar',
            'data': [{'category': str, 'value': int}, ...]
        }
    """
    reason_counts = df["Reason"].value_counts()

    data = []
    for reason, count in reason_counts.items():
        data.append({"category": reason, "value": int(count), "count": int(count)})

    return {"title": "Call Distribution by Reason", "chart_type": "bar", "data": data}


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================


# CORRECTED visualizations.py - generate_visualization() function
# ===============================================================


def generate_visualization(query: str, agent_type: str, df: pd.DataFrame) -> dict:
    """
    Main entry point - generates appropriate visualization based on query and agent.

    Args:
        query: User's query string
        agent_type: 'energy' or 'customer_service'
        df: Relevant dataframe (energy or customer service)

    Returns:
        Visualization dict or None if no visualization needed
        {
            'title': str,
            'chart_type': 'line' or 'bar' or 'pie',
            'data': [...]
        }
    """
    needs_viz, viz_type = detect_visualization_need(query, agent_type)

    if not needs_viz:
        return None

    # Route to appropriate visualization
    if agent_type == "energy":
        if viz_type == "forecast":
            return generate_energy_forecast(df, days=14)
        elif viz_type == "comparison":
            return generate_energy_comparison(df)

        # NEW: Advanced analytics visualizations (MOVED HERE - was in customer_service!)
        elif viz_type in [
            "hourly_pattern",
            "peak_analysis",
            # "cost_analysis",
            "benchmark",
            "appliance_breakdown",
            "renewable_mix",
        ]:
            try:
                from .energy_analytics import (
                    analyze_appliance_consumption,
                    analyze_hourly_pattern,
                    analyze_renewable_mix,
                    calculate_electricity_cost,
                    compare_with_benchmarks,
                    detect_peak_hours,
                )

                if viz_type == "hourly_pattern":
                    return analyze_hourly_pattern()
                elif viz_type == "peak_analysis":
                    return detect_peak_hours()
                # elif viz_type == "cost_analysis":
                #     result = calculate_electricity_cost()
                #     return result.get("visualization") if result else None
                elif viz_type == "benchmark":
                    avg_consumption = df["EnergyConsumption_kWh"].mean()
                    return compare_with_benchmarks(avg_consumption)
                elif viz_type == "appliance_breakdown":
                    return analyze_appliance_consumption()
                elif viz_type == "renewable_mix":
                    return analyze_renewable_mix()
            except ImportError:
                print("⚠️ energy_analytics module not available")
                return None

    elif agent_type == "customer_service":
        if viz_type == "trend":
            return generate_call_volume_trend(df, days=14)
        elif viz_type == "distribution":
            return generate_call_reason_distribution(df)

    return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_visualization_suggestions(agent_type: str) -> list:
    """Get sample queries that trigger visualizations for an agent."""

    if agent_type == "energy":
        return [
            "Show me energy forecast for next 2 weeks",
            "Compare residential vs commercial consumption",
            "Predict energy usage trends",
            "Chart energy consumption by account type",
        ]

    elif agent_type == "customer_service":
        return [
            "Show call volume trend over last 2 weeks",
            "What are the most common call reasons?",
            "Display call distribution by category",
            "Chart daily call volumes",
        ]

    return []


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# if __name__ == "__main__":
#     print("Unified Visualization Module")
#     print("=" * 60)

#     # Example: Energy visualization
#     print("\nEnergy Agent Examples:")
#     energy_queries = [
#         "forecast energy for next 2 weeks",
#         "compare residential and commercial",
#         "what is customer RC1001 consumption"
#     ]

#     for q in energy_queries:
#         needs_viz, viz_type = detect_visualization_need(q, "energy")
#         print(f"  '{q}' → {viz_type if needs_viz else 'No visualization'}")

#     # Example: Customer Service visualization
#     print("\nCustomer Service Agent Examples:")
#     cs_queries = [
#         "show call volume trend",
#         "what are common call reasons",
#         "who called today"
#     ]

#     for q in cs_queries:
#         needs_viz, viz_type = detect_visualization_need(q, "customer_service")
#         print(f"  '{q}' → {viz_type if needs_viz else 'No visualization'}")

#     print("\n" + "=" * 60)
#     print("✓ Module loaded successfully")
