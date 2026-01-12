"""
Advanced Energy Analytics
=========================

Provides analytics functions for new energy datasets:
- Hourly pattern analysis
- Cost calculations
- Benchmarking
- Rebate matching
- Equipment analysis
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import (
    get_annual_report,
    get_benchmarks,
    get_equipment_uec_dwelling,
    get_equipment_uec_utility,
    get_hourly_data,
    get_rebates,
    get_tou_rates,
)

# ============================================================================
# HOURLY CONSUMPTION ANALYTICS
# ============================================================================


def analyze_hourly_pattern(customer_id: Optional[str] = None, days: int = 7) -> Dict:
    """
    Analyze hourly consumption pattern.

    Returns visualization data for hourly usage line chart.
    """
    hourly_df = get_hourly_data()

    if hourly_df is None:
        return None

    # Filter by customer if specified
    if customer_id and "CustomerID" in hourly_df.columns:
        hourly_df = hourly_df[hourly_df["CustomerID"] == customer_id]

    # Get last N days
    hourly_df = hourly_df.tail(24 * days)

    # Calculate average by hour of day
    if "Hour" in hourly_df.columns:
        avg_by_hour = hourly_df.groupby("Hour")["Consumption_kWh"].mean()

        data = []
        for hour in range(24):
            data.append(
                {
                    "hour": f"{hour:02d}:00",
                    "consumption": round(avg_by_hour.get(hour, 0), 2),
                }
            )

        return {
            "title": f"Average Hourly Usage Pattern ({days} days)",
            "chart_type": "line",
            "data": data,
        }

    return None


def detect_peak_hours(customer_id: Optional[str] = None) -> Dict:
    """
    Detect peak usage hours and compare with TOU rates.

    Returns stacked bar chart data.
    """
    hourly_df = get_hourly_data()
    tou_df = get_tou_rates()

    if hourly_df is None:
        return None

    # Filter by customer if specified
    if customer_id and "CustomerID" in hourly_df.columns:
        hourly_df = hourly_df[hourly_df["CustomerID"] == customer_id]

    # Categorize hours into peak/off-peak
    # Assuming TOU rates have 'Period' column (Peak/Part-Peak/Off-Peak)
    if tou_df is not None and "Period" in tou_df.columns:
        # Join with TOU periods
        # Simplified: aggregate by period
        peak_usage = hourly_df[hourly_df["Hour"].isin(range(16, 21))][
            "Consumption_kWh"
        ].sum()
        offpeak_usage = hourly_df[~hourly_df["Hour"].isin(range(16, 21))][
            "Consumption_kWh"
        ].sum()

        data = [
            {"period": "Peak (4-9pm)", "consumption": round(peak_usage, 2)},
            {"period": "Off-Peak", "consumption": round(offpeak_usage, 2)},
        ]
    else:
        # Fallback: just show usage by time blocks
        morning = hourly_df[hourly_df["Hour"].isin(range(6, 12))][
            "Consumption_kWh"
        ].sum()
        afternoon = hourly_df[hourly_df["Hour"].isin(range(12, 18))][
            "Consumption_kWh"
        ].sum()
        evening = hourly_df[hourly_df["Hour"].isin(range(18, 24))][
            "Consumption_kWh"
        ].sum()
        night = hourly_df[hourly_df["Hour"].isin(range(0, 6))]["Consumption_kWh"].sum()

        data = [
            {"period": "Morning (6am-12pm)", "consumption": round(morning, 2)},
            {"period": "Afternoon (12pm-6pm)", "consumption": round(afternoon, 2)},
            {"period": "Evening (6pm-12am)", "consumption": round(evening, 2)},
            {"period": "Night (12am-6am)", "consumption": round(night, 2)},
        ]

    return {"title": "Usage by Time Period", "chart_type": "bar", "data": data}


# ============================================================================
# COST ANALYSIS
# ============================================================================


def calculate_electricity_cost(
    customer_id: Optional[str] = None, month: Optional[str] = None
) -> Dict:
    """
    Calculate electricity cost based on consumption and TOU rates.

    Returns cost breakdown visualization.
    """
    hourly_df = get_hourly_data()
    tou_df = get_tou_rates()

    if hourly_df is None or tou_df is None:
        return None

    # Filter by customer and month if specified
    if customer_id and "CustomerID" in hourly_df.columns:
        hourly_df = hourly_df[hourly_df["CustomerID"] == customer_id]

    # Simplified cost calculation
    # Assume average rate from TOU table
    if "Rate_per_kWh" in tou_df.columns:
        avg_rate = tou_df["Rate_per_kWh"].mean()
    else:
        avg_rate = 0.15  # Default $0.15/kWh

    total_consumption = hourly_df["Consumption_kWh"].sum()
    total_cost = total_consumption * avg_rate

    return {
        "total_consumption": round(total_consumption, 2),
        "total_cost": round(total_cost, 2),
        "average_rate": round(avg_rate, 4),
        "visualization": {
            "title": "Monthly Electricity Cost Breakdown",
            "chart_type": "bar",
            "data": [
                {"category": "Consumption (kWh)", "value": round(total_consumption, 2)},
                {"category": "Total Cost ($)", "value": round(total_cost, 2)},
            ],
        },
    }


# ============================================================================
# BENCHMARKING
# ============================================================================


def compare_with_benchmarks(
    consumption_kwh: float, home_type: str = "single_family"
) -> Dict:
    """
    Compare consumption with California benchmarks.

    Returns comparison bar chart.
    """
    benchmarks_df = get_benchmarks()

    if benchmarks_df is None:
        # Use default benchmarks
        benchmarks = {
            "Your Home": consumption_kwh,
            "SMUD Average": 700,
            "PG&E Average": 550,
            "SCE Average": 600,
            "CA State Average": 650,
        }
    else:
        # Extract from benchmarks DataFrame
        benchmarks = {"Your Home": consumption_kwh}
        for _, row in benchmarks_df.iterrows():
            if "Utility" in row and "Average_kWh" in row:
                benchmarks[row["Utility"]] = row["Average_kWh"]

    data = [
        {"utility": name, "consumption": round(value, 2), "count": 1}
        for name, value in benchmarks.items()
    ]

    return {
        "title": "Energy Usage Comparison Across Utilities",
        "chart_type": "bar",
        "data": data,
    }


# ============================================================================
# REBATE MATCHING
# ============================================================================


def find_rebate_opportunities(
    consumption_kwh: float, home_type: str = "residential"
) -> Dict:
    """
    Find applicable rebate programs.

    Returns list of rebate opportunities.
    """
    rebates_df = get_rebates()

    if rebates_df is None:
        return {"rebates": [], "message": "Rebate data not available"}

    # Filter applicable rebates
    applicable_rebates = []

    for _, row in rebates_df.iterrows():
        # Check eligibility (simplified)
        if (
            "Program_Type" in row
            and home_type.lower() in str(row["Program_Type"]).lower()
        ):
            applicable_rebates.append(
                {
                    "program": row.get("Program_Name", "Unknown"),
                    "amount": row.get("Rebate_Amount", 0),
                    "description": row.get("Description", ""),
                    "eligibility": row.get("Eligibility", ""),
                }
            )

    return {
        "rebates": applicable_rebates,
        "total_potential_savings": sum(r["amount"] for r in applicable_rebates),
    }


# ============================================================================
# APPLIANCE ANALYSIS
# ============================================================================


def analyze_appliance_consumption(dwelling_type: str = "single_family") -> Dict:
    """
    Analyze appliance-level energy consumption.

    Returns pie chart of appliance breakdown.
    """
    equipment_df = get_equipment_uec_dwelling()

    if equipment_df is None:
        # Use default estimates
        appliances = {
            "HVAC": 45,
            "Water Heater": 18,
            "Lighting": 12,
            "Refrigerator": 8,
            "Other": 17,
        }
    else:
        # Extract from UEC data
        dwelling_df = (
            equipment_df[equipment_df["Dwelling_Type"] == dwelling_type]
            if "Dwelling_Type" in equipment_df.columns
            else equipment_df
        )

        appliances = {}
        for _, row in dwelling_df.iterrows():
            if "Appliance" in row and "UEC_kWh" in row:
                appliances[row["Appliance"]] = row["UEC_kWh"]

    total = sum(appliances.values())
    data = [
        {
            "category": name,
            "value": round(consumption, 2),
            "percentage": round((consumption / total) * 100, 1) if total > 0 else 0,
        }
        for name, consumption in appliances.items()
    ]

    return {
        "title": f"Appliance Energy Breakdown - {dwelling_type.title()}",
        "chart_type": "pie",
        "data": data,
    }


# ============================================================================
# GRID ANALYSIS
# ============================================================================


def analyze_renewable_mix() -> Dict:
    """
    Analyze SMUD's renewable energy mix.

    Returns pie chart of power sources.
    """
    annual_df = get_annual_report()

    if annual_df is None:
        # Default renewable mix
        power_sources = {
            "Solar": 25,
            "Wind": 10,
            "Hydro": 15,
            "Battery": 5,
            "Natural Gas": 30,
            "Other": 15,
        }
    else:
        # Extract from annual report
        power_sources = {}
        for _, row in annual_df.iterrows():
            if "Power_Source" in row and "Percentage" in row:
                power_sources[row["Power_Source"]] = row["Percentage"]

    # Calculate renewable percentage
    renewable_sources = ["Solar", "Wind", "Hydro", "Hydroelectric", "Battery"]
    renewable_pct = sum(
        pct
        for source, pct in power_sources.items()
        if any(r in source for r in renewable_sources)
    )

    data = [
        {"category": name, "value": round(pct, 2)}
        for name, pct in power_sources.items()
    ]

    return {
        "title": f"SMUD Power Supply Mix ({renewable_pct:.1f}% Renewable)",
        "chart_type": "pie",
        "data": data,
        "renewable_percentage": round(renewable_pct, 1),
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_analytics_summary() -> Dict:
    """Get summary of available analytics"""
    from .data_loader import get_data_loader

    available = get_data_loader().get_available_datasets()

    analytics = {
        "hourly_pattern": available["hourly_consumption"],
        "cost_analysis": available["hourly_consumption"] and available["tou_rates"],
        "benchmarking": available["benchmarks"],
        "rebates": available["rebates"],
        "appliance_analysis": available["equipment_uec_dwelling"],
        "grid_analysis": available["annual_report"],
    }

    return analytics


# if __name__ == "__main__":
#     """Test analytics functions"""
#     print("Testing Energy Analytics")
#     print("=" * 60)

#     summary = get_analytics_summary()
#     print("\nAvailable Analytics:")
#     for analysis, available in summary.items():
#         status = "✓" if available else "✗"
#         print(f"  {status} {analysis}")

#     print("\n" + "=" * 60)
