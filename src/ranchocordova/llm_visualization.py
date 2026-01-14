"""
LLM-Driven Visualization Module
================================

Uses the LLM to analyze user queries and generate Chart.js configurations.
The frontend renders interactive charts in the browser using Chart.js.

Key Functions:
- analyze_visualization_request(): LLM analyzes query to determine chart type
- extract_chart_data(): Extracts data from DataFrames based on LLM spec
- build_chartjs_config(): Builds complete Chart.js configuration
- generate_llm_visualization(): Main entry point
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

VISUALIZATION_ANALYSIS_PROMPT = """You are a data visualization expert. Analyze the user's query and determine the best chart type and data to display.

Available datasets:

1. ENERGY DATA (energy_df):
   - CustomerID: Customer identifier (e.g., RC1001)
   - AccountType: "Residential" or "Commercial"
   - Month: Date in YYYY-MM format (e.g., 2024-05)
   - EnergyConsumption_kWh: Energy usage in kilowatt-hours (numeric)

2. CUSTOMER SERVICE DATA (cs_df):
   - CallID: Call identifier
   - CustomerID: Customer identifier
   - CustomerName: Customer's full name
   - DateTime: Date and time of call (e.g., 2024-05-01 09:03)
   - Reason: Why they called (e.g., "Billing question", "Outage report", "Payment arrangement")
   - Agent: Agent who handled the call
   - Resolution: How the call was resolved

Based on the user's query, respond with a JSON object ONLY (no other text):

{{
    "needs_visualization": true or false,
    "reasoning": "Brief explanation of why this chart type is appropriate",
    "chart_type": "line" | "bar" | "pie" | "doughnut" | null,
    "dataset": "energy" | "customer_service",
    "data_spec": {{
        "x_column": "column name for x-axis/labels",
        "y_column": "column name for y-axis/values",
        "aggregation": "sum" | "mean" | "count",
        "group_by": "optional column to group by" or null,
        "top_n": number or null,
        "filters": [{{"column": "col", "operator": "==", "value": "val"}}] or []
    }},
    "chart_options": {{
        "title": "Descriptive chart title",
        "x_label": "X-axis label",
        "y_label": "Y-axis label"
    }}
}}

Guidelines for chart type selection:
- LINE: For trends over time (e.g., "show consumption trend", "forecast", "over time")
- BAR: For comparisons between categories (e.g., "compare residential vs commercial", "top customers")
- PIE/DOUGHNUT: For showing proportions/distribution (e.g., "breakdown", "distribution", "most common reasons")

User query: "{query}"

Respond with JSON only:"""




# ============================================================================
# CHART COLOR PALETTES
# ============================================================================

CHART_COLORS = {
    "primary": [
        "rgba(54, 162, 235, 0.8)",   # Blue
        "rgba(255, 99, 132, 0.8)",   # Red
        "rgba(75, 192, 192, 0.8)",   # Teal
        "rgba(255, 159, 64, 0.8)",   # Orange
        "rgba(153, 102, 255, 0.8)",  # Purple
        "rgba(255, 205, 86, 0.8)",   # Yellow
        "rgba(201, 203, 207, 0.8)",  # Gray
    ],
    "borders": [
        "rgba(54, 162, 235, 1)",
        "rgba(255, 99, 132, 1)",
        "rgba(75, 192, 192, 1)",
        "rgba(255, 159, 64, 1)",
        "rgba(153, 102, 255, 1)",
        "rgba(255, 205, 86, 1)",
        "rgba(201, 203, 207, 1)",
    ]
}


# ============================================================================
# LLM ANALYSIS
# ============================================================================

def analyze_visualization_request(
    query: str,
    model,
    tokenizer
) -> Optional[Dict]:
    """
    Use LLM to analyze user query and determine visualization needs.
    
    Returns:
        Dict with chart specification or None if no visualization needed
    """
    prompt = VISUALIZATION_ANALYSIS_PROMPT.format(query=query)
    
    messages = [
        {"role": "system", "content": "You are a helpful data visualization assistant. Always respond with valid JSON only."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.1,  # Low temperature for consistent JSON output
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"🤖 LLM Visualization Analysis Response:\n{response[:500]}")
    
    # Parse JSON from response
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            spec = json.loads(json_match.group())
            
            # If LLM returned needs_visualization: false, check if query has viz keywords
            # and use fallback if it does (LLM may have been wrong)
            if not spec.get("needs_visualization", False):
                print("🔍 LLM returned needs_visualization: false - checking for keywords...")
                fallback_spec = _fallback_analysis(query)
                if fallback_spec and fallback_spec.get("needs_visualization", False):
                    print("✅ Fallback detected visualization is needed - overriding LLM")
                    return fallback_spec
            
            return spec
        else:
            # No JSON found - use fallback
            print("⚠️ No JSON found in LLM response - using fallback")
            return _fallback_analysis(query)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error: {e}")
        # Fallback to keyword-based detection
        return _fallback_analysis(query)
    
    return None


def _fallback_analysis(query: str) -> Optional[Dict]:
    """
    Fallback analysis using keyword matching when LLM fails to produce valid JSON.
    """
    query_lower = query.lower()
    
    # Detect if visualization is needed - expanded patterns
    viz_keywords = ["chart", "graph", "plot", "show", "visualize", "display", "trend", "compare", "breakdown"]
    question_patterns = ["what are the", "most common", "top", "how many", "how much"]
    
    has_viz_keyword = any(kw in query_lower for kw in viz_keywords)
    has_question_pattern = any(pattern in query_lower for pattern in question_patterns)
    
    if not (has_viz_keyword or has_question_pattern):
        return {"needs_visualization": False}

    
    # Determine dataset
    energy_keywords = ["energy", "consumption", "kwh", "usage", "electricity", "power"]
    cs_keywords = ["call", "reason", "agent", "customer service", "service"]
    
    dataset = "energy" if any(kw in query_lower for kw in energy_keywords) else "customer_service"
    
    # Determine chart type
    if any(kw in query_lower for kw in ["trend", "over time", "forecast", "monthly"]):
        chart_type = "line"
        if dataset == "energy":
            spec = {
                "x_column": "Month",
                "y_column": "EnergyConsumption_kWh",
                "aggregation": "mean",
                "group_by": None
            }
            title = "Energy Consumption Trend"
        else:
            spec = {
                "x_column": "DateTime",
                "y_column": "CallID",
                "aggregation": "count",
                "group_by": None
            }
            title = "Call Volume Over Time"
    elif any(kw in query_lower for kw in ["compare", "vs", "versus", "comparison"]):
        chart_type = "bar"
        if dataset == "energy":
            spec = {
                "x_column": "AccountType",
                "y_column": "EnergyConsumption_kWh",
                "aggregation": "mean",
                "group_by": "AccountType"
            }
            title = "Energy Consumption by Account Type"
        else:
            spec = {
                "x_column": "Reason",
                "y_column": "CallID",
                "aggregation": "count",
                "group_by": "Reason"
            }
            title = "Calls by Reason"
    elif any(kw in query_lower for kw in ["breakdown", "distribution", "common", "reasons", "pie"]):
        chart_type = "pie"
        if dataset == "customer_service":
            spec = {
                "x_column": "Reason",
                "y_column": "CallID",
                "aggregation": "count",
                "group_by": "Reason"
            }
            title = "Call Reasons Distribution"
        else:
            spec = {
                "x_column": "AccountType",
                "y_column": "EnergyConsumption_kWh",
                "aggregation": "sum",
                "group_by": "AccountType"
            }
            title = "Energy by Account Type"
    else:
        # Default to bar chart
        chart_type = "bar"
        if dataset == "energy":
            spec = {
                "x_column": "AccountType",
                "y_column": "EnergyConsumption_kWh",
                "aggregation": "mean",
                "group_by": "AccountType"
            }
            title = "Energy Consumption Overview"
        else:
            spec = {
                "x_column": "Reason",
                "y_column": "CallID",
                "aggregation": "count",
                "group_by": "Reason"
            }
            title = "Customer Service Overview"
    
    return {
        "needs_visualization": True,
        "reasoning": "Fallback: Detected visualization keywords",
        "chart_type": chart_type,
        "dataset": dataset,
        "data_spec": spec,
        "chart_options": {
            "title": title,
            "x_label": spec["x_column"],
            "y_label": spec["y_column"]
        }
    }


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def _normalize_dataset(dataset: str) -> str:
    """Normalize dataset name from LLM output."""
    if dataset is None:
        return "energy"
    dataset_lower = dataset.lower()
    if "energy" in dataset_lower:
        return "energy"
    elif "service" in dataset_lower or "call" in dataset_lower or "cs" in dataset_lower:
        return "customer_service"
    return "energy"


def _find_column(df: pd.DataFrame, col_name: str) -> str:
    """Find a column in DataFrame using fuzzy matching."""
    if col_name in df.columns:
        return col_name
    
    # Try case-insensitive match
    col_lower = col_name.lower().replace(" ", "").replace("_", "")
    for col in df.columns:
        if col.lower().replace(" ", "").replace("_", "") == col_lower:
            return col
    
    # Try partial match
    for col in df.columns:
        if col_lower in col.lower().replace(" ", "").replace("_", ""):
            return col
        if col.lower().replace(" ", "").replace("_", "") in col_lower:
            return col
    
    return col_name  # Return original if no match found


def extract_chart_data(
    spec: Dict,
    energy_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    allow_fallback: bool = True
) -> Tuple[List[str], List[float]]:
    """
    Extract data from DataFrames based on LLM specification.
    
    Returns:
        Tuple of (labels, values)
    """
    data_spec = spec.get("data_spec", {})
    dataset = _normalize_dataset(spec.get("dataset", "energy"))
    
    # Select the appropriate dataframe
    df = energy_df if dataset == "energy" else cs_df
    
    if df is None or df.empty:
        return [], []
    
    # Get column names and normalize them
    x_col = _find_column(df, data_spec.get("x_column", ""))
    y_col = _find_column(df, data_spec.get("y_column", ""))
    group_by = data_spec.get("group_by")
    if group_by:
        group_by = _find_column(df, group_by)
    
    aggregation = data_spec.get("aggregation", "sum")
    top_n = data_spec.get("top_n")
    filters = data_spec.get("filters", [])
    
    print(f"📊 Data extraction: dataset={dataset}, x={x_col}, y={y_col}, group_by={group_by}")
    
    # Apply filters

    for f in filters:
        col = f.get("column")
        op = f.get("operator", "==")
        val = f.get("value")
        if col in df.columns:
            if op == "==":
                df = df[df[col] == val]
            elif op == "!=":
                df = df[df[col] != val]
            elif op == ">":
                df = df[df[col] > val]
            elif op == "<":
                df = df[df[col] < val]
    
    # Perform aggregation
    if group_by and group_by in df.columns:
        if aggregation == "count":
            grouped = df.groupby(group_by).size()
        elif aggregation == "mean":
            if y_col in df.columns:
                grouped = df.groupby(group_by)[y_col].mean()
            else:
                grouped = df.groupby(group_by).size()
        elif aggregation == "sum":
            if y_col in df.columns:
                grouped = df.groupby(group_by)[y_col].sum()
            else:
                grouped = df.groupby(group_by).size()
        else:
            grouped = df.groupby(group_by).size()
        
        # Sort and optionally limit
        grouped = grouped.sort_values(ascending=False)
        if top_n:
            grouped = grouped.head(top_n)
        
        labels = grouped.index.tolist()
        values = grouped.values.tolist()
    elif x_col in df.columns:
        # No grouping, use x_col directly
        if aggregation == "count":
            grouped = df.groupby(x_col).size()
        elif y_col in df.columns:
            if aggregation == "mean":
                grouped = df.groupby(x_col)[y_col].mean()
            else:
                grouped = df.groupby(x_col)[y_col].sum()
        else:
            grouped = df.groupby(x_col).size()
        
        labels = grouped.index.tolist()
        values = grouped.values.tolist()
    else:
        # Fallback: use available columns
        if dataset == "energy" and "AccountType" in df.columns:
            grouped = df.groupby("AccountType")["EnergyConsumption_kWh"].mean()
            labels = grouped.index.tolist()
            values = grouped.values.tolist()
        elif dataset == "customer_service" and "Reason" in df.columns:
            grouped = df.groupby("Reason").size()
            labels = grouped.index.tolist()
            values = grouped.values.tolist()
        else:
            labels = []
            values = []
    
    # SMART FALLBACK: If only 1 data point, try alternative groupings
    if allow_fallback and len(labels) <= 1:
        print(f"⚠️ Only {len(labels)} data points found - trying alternative grouping")
        if dataset == "energy":
            # Try grouping by AccountType first
            if "AccountType" in df.columns and "EnergyConsumption_kWh" in df.columns:
                grouped = df.groupby("AccountType")["EnergyConsumption_kWh"].mean()
                if len(grouped) > 1:
                    labels = grouped.index.tolist()
                    values = grouped.values.tolist()
                    print(f"✅ Fallback to AccountType grouping: {len(labels)} data points")
                else:
                    # Show top 10 customers by consumption
                    top_customers = df.nlargest(10, "EnergyConsumption_kWh")
                    labels = top_customers["CustomerID"].tolist()
                    values = top_customers["EnergyConsumption_kWh"].tolist()
                    print(f"✅ Fallback to top customers: {len(labels)} data points")
        elif dataset == "customer_service":
            # Try grouping by Reason
            if "Reason" in df.columns:
                grouped = df.groupby("Reason").size().sort_values(ascending=False)
                labels = grouped.index.tolist()
                values = grouped.values.tolist()
                print(f"✅ Fallback to Reason grouping: {len(labels)} data points")
    
    # Convert to serializable types
    labels = [str(l) for l in labels]
    values = [float(v) for v in values]
    

    return labels, values


# ============================================================================
# CHART.JS CONFIG BUILDER
# ============================================================================

def build_chartjs_config(
    chart_type: str,
    labels: List[str],
    values: List[float],
    options: Dict
) -> Dict:
    """
    Build a complete Chart.js configuration object.
    
    Args:
        chart_type: "line", "bar", "pie", "doughnut"
        labels: List of labels for x-axis or pie segments
        values: List of numeric values
        options: Chart options (title, labels, etc.)
    
    Returns:
        Chart.js configuration dict
    """
    title = options.get("title", "Chart")
    x_label = options.get("x_label", "")
    y_label = options.get("y_label", "")
    
    # Build dataset based on chart type
    if chart_type in ["pie", "doughnut"]:
        dataset = {
            "label": title,
            "data": values,
            "backgroundColor": CHART_COLORS["primary"][:len(values)],
            "borderColor": CHART_COLORS["borders"][:len(values)],
            "borderWidth": 2
        }
        
        config = {
            "type": chart_type,
            "data": {
                "labels": labels,
                "datasets": [dataset]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": title,
                        "font": {"size": 16, "weight": "bold"}
                    },
                    "legend": {
                        "position": "right",
                        "labels": {"font": {"size": 12}}
                    }
                }
            }
        }
    else:
        # Line or Bar chart
        dataset = {
            "label": y_label or "Value",
            "data": values,
            "backgroundColor": CHART_COLORS["primary"][0] if chart_type == "bar" else "rgba(54, 162, 235, 0.2)",
            "borderColor": CHART_COLORS["borders"][0],
            "borderWidth": 2,
            "fill": chart_type == "line",
            "tension": 0.3 if chart_type == "line" else 0
        }
        
        config = {
            "type": chart_type,
            "data": {
                "labels": labels,
                "datasets": [dataset]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": title,
                        "font": {"size": 16, "weight": "bold"}
                    },
                    "legend": {
                        "display": True,
                        "position": "top"
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": bool(x_label),
                            "text": x_label
                        }
                    },
                    "y": {
                        "title": {
                            "display": bool(y_label),
                            "text": y_label
                        },
                        "beginAtZero": True
                    }
                }
            }
        }
    
    return config


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def generate_llm_visualization(
    query: str,
    energy_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    model,
    tokenizer
) -> Optional[Dict]:
    """
    Main entry point for LLM-driven visualization generation.
    
    Args:
        query: User's query string
        energy_df: Energy consumption DataFrame
        cs_df: Customer service DataFrame
        model: LLM model
        tokenizer: LLM tokenizer
    
    Returns:
        Chart.js configuration dict or None if no visualization needed
    """
    print(f"📊 Analyzing visualization request: {query[:100]}...")
    
    # Step 1: Analyze with LLM
    spec = analyze_visualization_request(query, model, tokenizer)
    
    if spec is None or not spec.get("needs_visualization", False):
        print("ℹ️ No visualization needed for this query")
        return None
    
    # Normalize chart type to lowercase (Chart.js requirement)
    chart_type = spec.get("chart_type", "bar")
    if chart_type:
        chart_type = chart_type.lower()
    
    print(f"✅ LLM determined chart type: {chart_type}")
    print(f"   Reasoning: {spec.get('reasoning', 'N/A')}")
    
    # CHECK FOR FORECASTING
    # Determine if this is a forecast/trend request BEFORE data extraction
    # This allows us to disable the fallback logic that switches to categorical data
    query_lower = query.lower()
    is_forecast = any(kw in query_lower for kw in ["forecast", "predict", "future", "next"])
    is_trend = any(kw in query_lower for kw in ["trend", "over time"])
    
    # Step 2: Extract data
    # Disable fallback for forecasts, so we can detect the "1 data point" case correctly
    allow_fallback = not (is_forecast or is_trend)
    labels, values = extract_chart_data(spec, energy_df, cs_df, allow_fallback=allow_fallback)
    
    if not labels or not values:
        print("⚠️ No data extracted for visualization")
        return None
    
    print(f"✅ Extracted {len(labels)} data points")
    
    print(f"✅ Extracted {len(labels)} data points")

    # If using single month data but user wants forecast/trend, generate synthetic points
    if (is_forecast or is_trend) and len(labels) < 2:
        print("📊 Forecast/Trend requested but insufficient data. Generating synthetic forecast points...")
        predicted_labels, predicted_values = _generate_forecast_points(model, tokenizer, labels, values)
        
        if predicted_labels:
            labels.extend(predicted_labels)
            values.extend(predicted_values)
            print(f"✅ Added {len(predicted_labels)} forecast points")
            # Force line chart for trends
            chart_type = "line"
            
            # Update title to indicate forecast
            if "chart_options" not in spec:
                spec["chart_options"] = {}
            current_title = spec["chart_options"].get("title", "Chart")
            spec["chart_options"]["title"] = f"{current_title} (inc. Forecast)"
    
    # Step 3: Build Chart.js config
    chart_config = build_chartjs_config(
        chart_type=chart_type,
        labels=labels,
        values=values,
        options=spec.get("chart_options", {})
    )
    
    print(f"✅ Built Chart.js config for {chart_type} chart")
    
    return chart_config


# ============================================================================
# FORECAST GENERATION
# ============================================================================

FORECAST_PROMPT = """You are a predictive analytics engine. Based on the historical data provided, generate a realistic forecast for the next {num_points} periods.

Historical Data: {history}

Detect the trend and seasonality from the data (or general knowledge of energy/call patterns if data is scant) and project future values.
For energy data, assume higher usage in summer (July-Aug) and winter (Jan-Dec) and lower in spring/fall.

Respond with a JSON object ONLY:
{{
    "forecast": [
        {{"label": "Next Period Label", "value": 123.45}},
        ...
    ]
}}
"""

def _generate_forecast_points(
    model, 
    tokenizer, 
    labels: List[str], 
    values: List[float], 
    count: int = 3
) -> Tuple[List[str], List[float]]:
    """
    Generate forecast points using LLM.
    """
    history = [f"{l}: {v}" for l, v in zip(labels[-5:], values[-5:])]
    prompt = FORECAST_PROMPT.format(num_points=count, history=", ".join(history))
    
    messages = [
        {"role": "system", "content": "You are a forecasting assistant. Output valid JSON only."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True, # Allow some creativity for forecasting
        )
    
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            forecast = data.get("forecast", [])
            
            new_labels = [item["label"] for item in forecast]
            new_values = [item["value"] for item in forecast]
            
            return new_labels, new_values
    except Exception as e:
        print(f"⚠️ Forecast generation failed: {e}")
    
    return [], []


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test with sample data
    print("Testing LLM Visualization Module")
    print("=" * 60)
    
    # Create sample DataFrames
    energy_data = {
        "CustomerID": ["RC1001", "RC1002", "RC1003"],
        "AccountType": ["Residential", "Residential", "Commercial"],
        "Month": ["2024-05", "2024-05", "2024-05"],
        "EnergyConsumption_kWh": [373, 415, 1129]
    }
    energy_df = pd.DataFrame(energy_data)
    
    cs_data = {
        "CallID": ["CL0001", "CL0002", "CL0003"],
        "Reason": ["Billing question", "Outage report", "Billing question"],
        "Agent": ["Agent Brown", "Agent Wang", "Agent Smith"]
    }
    cs_df = pd.DataFrame(cs_data)
    
    # Test fallback analysis
    print("\nTesting fallback analysis:")
    queries = [
        "Show energy consumption trend",
        "Compare residential vs commercial",
        "What are the most common call reasons?",
        "Hello, how are you?"
    ]
    
    for q in queries:
        result = _fallback_analysis(q)
        print(f"  Query: '{q}'")
        print(f"    Needs viz: {result.get('needs_visualization')}")
        print(f"    Chart type: {result.get('chart_type')}")
        print()

    print("\nTesting Forecast Logic (Mocking LLM):")
    # We can't easily mock the LLM here without imports, so we will rely on inspection or user testing.
    # However, we can check if the function _generate_forecast_points exists and is importable.
    print(f"  _generate_forecast_points exists: {'_generate_forecast_points' in globals()}")

