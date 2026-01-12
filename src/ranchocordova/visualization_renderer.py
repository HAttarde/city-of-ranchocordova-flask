"""
Visualization Renderer - Converts JSON chart data to base64 images
==================================================================

This module takes the JSON data structures from visualizations.py
and renders them as actual PNG images encoded in base64.
"""

import base64
import io

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np


def render_visualization_to_base64(viz_data: dict) -> str:
    """
    Convert visualization JSON data to base64-encoded PNG image.

    Args:
        viz_data: Dict with 'chart_type', 'title', and 'data'

    Returns:
        Base64-encoded PNG string with data URI prefix
    """
    print(f"🎨 Renderer called with: {viz_data.keys() if viz_data else 'None'}")

    if not viz_data or "chart_type" not in viz_data:
        print("❌ No viz_data or missing chart_type")
        return None

    chart_type = viz_data["chart_type"]
    title = viz_data.get("title", "Visualization")
    data = viz_data.get("data", [])

    print(f"🎨 Rendering {chart_type} chart with {len(data)} data points")

    if not data:
        print("❌ No data to render")
        return None

    try:
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        ax.set_facecolor("white")

        if chart_type == "line":
            render_line_chart(ax, data, title)
        elif chart_type == "bar":
            render_bar_chart(ax, data, title)
        elif chart_type == "pie":
            render_pie_chart(ax, data, title)
        else:
            print(f"❌ Unknown chart type: {chart_type}")
            plt.close(fig)
            return None

        # Convert to base64
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buf.seek(0)
        image_bytes = buf.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        plt.close(fig)

        print(
            f"✅ Image rendered: {len(image_base64)} base64 chars, {len(image_bytes)} bytes"
        )

        return f"data:image/png;base64,{image_base64}"

    except Exception as e:
        print(f"❌ Rendering exception: {e}")
        import traceback

        traceback.print_exc()
        plt.close("all")
        return None


def render_line_chart(ax, data, title):
    """Render line chart for forecast or trend data."""
    if not data:
        return

    # Extract dates and values
    dates = [item.get("date", item.get("day", str(i))) for i, item in enumerate(data)]
    values = [
        item.get("consumption", item.get("value", item.get("calls", 0)))
        for item in data
    ]

    # Plot
    ax.plot(dates, values, marker="o", linewidth=2, markersize=6, color="#1f77b4")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=11)

    # Determine y-axis label based on data
    if "consumption" in data[0]:
        ax.set_ylabel("Energy Consumption (kWh)", fontsize=11)
    elif "calls" in data[0]:
        ax.set_ylabel("Number of Calls", fontsize=11)
    else:
        ax.set_ylabel("Value", fontsize=11)

    ax.grid(True, alpha=0.3, linestyle="--")

    # Rotate x-axis labels if many dates
    if len(dates) > 7:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add value labels on points for small datasets
    if len(values) <= 14:
        for i, (date, val) in enumerate(zip(dates, values)):
            ax.annotate(
                f"{val:.0f}",
                xy=(i, val),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                alpha=0.7,
            )


def render_bar_chart(ax, data, title):
    """Render bar chart for comparison data."""
    if not data:
        return

    # Extract categories and values
    categories = [
        item.get("category", item.get("reason", str(i))) for i, item in enumerate(data)
    ]
    values = [
        item.get("value", item.get("consumption", item.get("count", 0)))
        for item in data
    ]
    counts = [item.get("count", 0) for item in data]

    # Create bars with colors
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    bar_colors = [colors[i % len(colors)] for i in range(len(categories))]

    bars = ax.bar(
        categories, values, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=1
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Category", fontsize=11)

    # Determine y-axis label
    if "consumption" in data[0] or "value" in data[0]:
        ax.set_ylabel("Average Energy Consumption (kWh)", fontsize=11)
    else:
        ax.set_ylabel("Count", fontsize=11)

    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, val, count in zip(bars, values, counts):
        height = bar.get_height()
        label_text = f"{val:.0f}"
        if count > 0:
            label_text += f"\n(n={count})"

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label_text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Rotate x-axis labels if needed
    if len(categories) > 3:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def render_pie_chart(ax, data, title):
    """Render pie chart for distribution data."""
    if not data:
        return

    # Extract labels and values
    labels = [
        item.get("reason", item.get("category", str(i))) for i, item in enumerate(data)
    ]
    values = [item.get("count", item.get("value", 0)) for item in data]

    # Filter out zero values
    filtered_data = [(label, val) for label, val in zip(labels, values) if val > 0]
    if not filtered_data:
        return

    labels, values = zip(*filtered_data)

    # Colors
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors[: len(values)],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.85,
    )

    # Style
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    # Add legend with counts
    legend_labels = [f"{label}: {val}" for label, val in zip(labels, values)]
    ax.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    ax.axis("equal")  # Equal aspect ratio ensures circular pie


# Test function
if __name__ == "__main__":
    # Test line chart
    test_forecast = {
        "title": "14-Day Energy Forecast",
        "chart_type": "line",
        "data": [
            {"date": "2026-01-07", "consumption": 1200},
            {"date": "2026-01-08", "consumption": 1250},
            {"date": "2026-01-09", "consumption": 1180},
            {"date": "2026-01-10", "consumption": 1300},
        ],
    }

    result = render_visualization_to_base64(test_forecast)
    print(f"Line chart: {result[:100]}..." if result else "Failed")

    # Test bar chart
    test_comparison = {
        "title": "Energy by Account Type",
        "chart_type": "bar",
        "data": [
            {"category": "Residential", "value": 350.5, "count": 75},
            {"category": "Commercial", "value": 1407.08, "count": 25},
        ],
    }

    result = render_visualization_to_base64(test_comparison)
    print(f"Bar chart: {result[:100]}..." if result else "Failed")

    # Test pie chart
    test_distribution = {
        "title": "Call Reasons",
        "chart_type": "pie",
        "data": [
            {"reason": "Billing", "count": 30},
            {"reason": "Outage", "count": 25},
            {"reason": "Service", "count": 20},
        ],
    }

    result = render_visualization_to_base64(test_distribution)
    print(f"Pie chart: {result[:100]}..." if result else "Failed")
