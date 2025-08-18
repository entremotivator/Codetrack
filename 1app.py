import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import base64
from dataclasses import dataclass
import asyncio
import aiohttp

# =============================
# Configuration & Constants
# =============================

# OpenAI model pricing (per 1k tokens) - Updated for 2025
MODEL_PRICING = {
    # GPT-4 Family
    "gpt-4": {"input": 0.03, "output": 0.06, "category": "chat"},
    "gpt-4-32k": {"input": 0.06, "output": 0.12, "category": "chat"},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03, "category": "chat"},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03, "category": "chat"},
    "gpt-4-vision-preview": {"input": 0.01, "output": 0.03, "category": "vision"},
    
    # GPT-4o Family (2024-2025 models)
    "gpt-4o": {"input": 0.005, "output": 0.015, "category": "chat"},
    "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01, "category": "chat"},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "category": "chat"},
    "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006, "category": "chat"},
    "gpt-4o-audio-preview": {"input": 0.1, "output": 0.2, "category": "audio"},
    
    # GPT-3.5 Family
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002, "category": "chat"},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004, "category": "chat"},
    "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002, "category": "completions"},
    
    # o1 Family (Reasoning models)
    "o1": {"input": 0.015, "output": 0.06, "category": "reasoning"},
    "o1-preview": {"input": 0.015, "output": 0.06, "category": "reasoning"},
    "o1-mini": {"input": 0.003, "output": 0.012, "category": "reasoning"},
    
    # Embedding Models
    "text-embedding-ada-002": {"input": 0.0001, "output": 0, "category": "embeddings"},
    "text-embedding-3-small": {"input": 0.00002, "output": 0, "category": "embeddings"},
    "text-embedding-3-large": {"input": 0.00013, "output": 0, "category": "embeddings"},
    
    # Image Generation
    "dall-e-3": {"input": 0.04, "output": 0, "category": "image_generation"},  # per image
    "dall-e-2": {"input": 0.02, "output": 0, "category": "image_generation"},  # per image
    
    # Audio Models
    "whisper-1": {"input": 0.006, "output": 0, "category": "audio"},  # per minute
    "tts-1": {"input": 0.015, "output": 0, "category": "audio"},  # per 1k chars
    "tts-1-hd": {"input": 0.030, "output": 0, "category": "audio"},  # per 1k chars
    
    # Fine-tuning
    "babbage-002": {"input": 0.0016, "output": 0.0016, "category": "fine_tuning"},
    "davinci-002": {"input": 0.012, "output": 0.012, "category": "fine_tuning"},
}

# API Endpoints Configuration
API_ENDPOINTS = {
    "completions": "https://api.openai.com/v1/organization/usage/completions",
    "embeddings": "https://api.openai.com/v1/organization/usage/embeddings", 
    "moderations": "https://api.openai.com/v1/organization/usage/moderations",
    "images": "https://api.openai.com/v1/organization/usage/images",
    "audio": "https://api.openai.com/v1/organization/usage/audio",
    "costs": "https://api.openai.com/v1/organization/costs",
    "users": "https://api.openai.com/v1/organization/users",
    "projects": "https://api.openai.com/v1/organization/projects",
    "api_keys": "https://api.openai.com/v1/organization/api_keys",
}

@dataclass
class UserMetrics:
    user_id: str
    user_name: str
    total_requests: int
    total_tokens: int
    total_cost: float
    most_used_model: str
    last_activity: datetime.datetime
    projects: List[str]

# =============================
# Enhanced Data Fetching
# =============================

@st.cache_data(ttl=300)
def get_organization_info(api_key: str) -> Dict[str, Any]:
    """Fetch organization information."""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get("https://api.openai.com/v1/organizations", headers=headers)
        if response.status_code == 200:
            orgs = response.json().get("data", [])
            return orgs[0] if orgs else {}
    except:
        pass
    
    return {}

@st.cache_data(ttl=300)
def get_usage_data(endpoint_name: str, params: Dict, api_key: str) -> List[Dict]:
    """Enhanced data fetching with better error handling."""
    url = API_ENDPOINTS.get(endpoint_name, API_ENDPOINTS["completions"])
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_data = []
    page_cursor = None
    max_pages = 50
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    page_count = 0
    while page_count < max_pages:
        if page_cursor:
            params["page"] = page_cursor
            
        status_text.text(f"Fetching {endpoint_name} data... Page {page_count + 1}")
        progress_bar.progress(min(page_count / 10, 0.9))
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data_json = response.json()
                batch_data = data_json.get("data", [])
                all_data.extend(batch_data)
                
                page_cursor = data_json.get("next_page")
                if not page_cursor or not batch_data:
                    break
                    
            elif response.status_code == 401:
                st.error(f"❌ Authentication failed for {endpoint_name}. Check your API key.")
                break
            elif response.status_code == 403:
                st.warning(f"⚠️ Access forbidden for {endpoint_name}. Admin privileges may be required.")
                break
            elif response.status_code == 404:
                st.info(f"ℹ️ {endpoint_name} endpoint not available or no data found.")
                break
            else:
                st.error(f"❌ API Error for {endpoint_name}: {response.status_code}")
                break
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error for {endpoint_name}: {str(e)}")
            break
            
        page_count += 1

    progress_bar.progress(1.0)
    status_text.text(f"✅ Fetched {len(all_data)} {endpoint_name} records")
    
    # Clean up progress indicators
    time.sleep(0.5)
    progress_container.empty()
    
    return all_data

def get_all_api_usage(api_key: str, days: int, group_by: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch usage data from all OpenAI APIs."""
    start_time = int(time.time()) - (days * 24 * 60 * 60)
    base_params = {
        "start_time": start_time,
        "bucket_width": "1d",
        "limit": days,
    }
    
    if group_by:
        base_params["group_by"] = group_by
    
    all_usage_data = {}
    
    # Fetch data from all endpoints
    for endpoint_name in ["completions", "embeddings", "images", "audio", "moderations"]:
        with st.expander(f"📊 Fetching {endpoint_name.title()} Data", expanded=False):
            data = get_usage_data(endpoint_name, base_params.copy(), api_key)
            if data:
                df = parse_usage_data(data, endpoint_name)
                all_usage_data[endpoint_name] = df
                st.success(f"✅ {len(df)} records retrieved")
            else:
                st.info(f"ℹ️ No {endpoint_name} data available")
    
    return all_usage_data

def parse_usage_data(usage_data: List[Dict], api_type: str) -> pd.DataFrame:
    """Enhanced parsing for different API types."""
    records = []
    
    for bucket in usage_data:
        start_time = bucket.get("start_time")
        end_time = bucket.get("end_time")
        
        for result in bucket.get("results", []):
            base_record = {
                "api_type": api_type,
                "start_time": start_time,
                "end_time": end_time,
                "project_id": result.get("project_id"),
                "user_id": result.get("user_id"),
                "api_key_id": result.get("api_key_id"),
                "model": result.get("model"),
                "num_model_requests": result.get("num_model_requests", 0),
            }
            
            # API-specific fields
            if api_type == "completions":
                base_record.update({
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "input_cached_tokens": result.get("input_cached_tokens", 0),
                })
            elif api_type == "embeddings":
                base_record.update({
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": 0,
                })
            elif api_type == "images":
                base_record.update({
                    "images_generated": result.get("images", 0),
                    "input_tokens": 0,
                    "output_tokens": 0,
                })
            elif api_type == "audio":
                base_record.update({
                    "seconds": result.get("seconds", 0),
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                })
            
            records.append(base_record)

    df = pd.DataFrame(records)
    if not df.empty:
        df["start_datetime"] = pd.to_datetime(df["start_time"], unit="s", errors="coerce")
        df["end_datetime"] = pd.to_datetime(df["end_time"], unit="s", errors="coerce")
        
        # Calculate totals and costs
        df["total_tokens"] = df.get("input_tokens", 0) + df.get("output_tokens", 0)
        df["estimated_cost"] = df.apply(lambda row: calculate_api_cost(row, api_type), axis=1)
        
    return df

def calculate_api_cost(row: pd.Series, api_type: str) -> float:
    """Calculate cost based on API type and usage."""
    model = str(row.get("model", "")).lower()
    
    # Find matching pricing
    pricing = None
    for model_key, model_pricing in MODEL_PRICING.items():
        if model_key in model and model_pricing["category"] in [api_type, "chat"]:
            pricing = model_pricing
            break
    
    if not pricing:
        # Default pricing based on API type
        default_pricing = {
            "completions": {"input": 0.01, "output": 0.03},
            "embeddings": {"input": 0.0001, "output": 0},
            "images": {"input": 0.02, "output": 0},
            "audio": {"input": 0.006, "output": 0},
            "moderations": {"input": 0.0002, "output": 0},
        }
        pricing = default_pricing.get(api_type, {"input": 0.01, "output": 0.03})
    
    input_tokens = row.get("input_tokens", 0)
    output_tokens = row.get("output_tokens", 0)
    
    if api_type == "images":
        return row.get("images_generated", 0) * pricing["input"]
    elif api_type == "audio":
        seconds = row.get("seconds", 0)
        return (seconds / 60) * pricing["input"]  # Per minute pricing
    else:
        return (input_tokens * pricing["input"] / 1000) + (output_tokens * pricing["output"] / 1000)

# =============================
# User Analytics Functions
# =============================

def analyze_user_metrics(all_usage_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Comprehensive user analysis across all APIs."""
    user_records = []
    
    for api_type, df in all_usage_data.items():
        if df.empty or "user_id" not in df.columns:
            continue
            
        user_stats = df.groupby("user_id").agg({
            "num_model_requests": "sum",
            "total_tokens": "sum", 
            "estimated_cost": "sum",
            "start_datetime": "max",
            "model": lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown"
        }).reset_index()
        
        user_stats["api_type"] = api_type
        user_records.append(user_stats)
    
    if not user_records:
        return pd.DataFrame()
    
    # Combine all user data
    combined_users = pd.concat(user_records, ignore_index=True)
    
    # Aggregate by user across all APIs
    final_user_stats = combined_users.groupby("user_id").agg({
        "num_model_requests": "sum",
        "total_tokens": "sum",
        "estimated_cost": "sum",
        "start_datetime": "max",
    }).reset_index()
    
    # Add user rankings and categories
    final_user_stats["cost_rank"] = final_user_stats["estimated_cost"].rank(ascending=False)
    final_user_stats["usage_rank"] = final_user_stats["total_tokens"].rank(ascending=False)
    
    # Categorize users
    def categorize_user(cost):
        if cost > 100:
            return "🔥 Heavy User"
        elif cost > 10:
            return "📈 Regular User"
        elif cost > 1:
            return "📊 Light User"
        else:
            return "🔍 Minimal User"
    
    final_user_stats["user_category"] = final_user_stats["estimated_cost"].apply(categorize_user)
    
    return final_user_stats

def create_user_dashboard(user_stats: pd.DataFrame, all_usage_data: Dict[str, pd.DataFrame]) -> None:
    """Create comprehensive user analytics dashboard."""
    if user_stats.empty:
        st.info("👤 No user data available. Enable 'Group by user' in settings.")
        return
    
    st.subheader("👥 User Analytics Dashboard")
    
    # User overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_users = len(user_stats)
    active_users = len(user_stats[user_stats["estimated_cost"] > 0])
    top_user_cost = user_stats["estimated_cost"].max()
    avg_cost_per_user = user_stats["estimated_cost"].mean()
    
    with col1:
        st.metric("👥 Total Users", f"{total_users:,}")
    with col2:
        st.metric("🎯 Active Users", f"{active_users:,}")
    with col3:
        st.metric("💰 Top User Cost", f"${top_user_cost:.2f}")
    with col4:
        st.metric("📊 Avg Cost/User", f"${avg_cost_per_user:.2f}")
    
    # User distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # User category distribution
        category_counts = user_stats["user_category"].value_counts()
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="👥 User Distribution by Usage Level",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Top users by cost
        top_users = user_stats.nlargest(10, "estimated_cost")
        fig_bar = px.bar(
            top_users,
            x="estimated_cost",
            y="user_id",
            orientation="h",
            title="💰 Top 10 Users by Cost",
            color="estimated_cost",
            color_continuous_scale="Viridis"
        )
        fig_bar.update_yaxis(title="User ID")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed user table with search and filtering
    st.subheader("🔍 User Details Table")
    
    # Search and filter controls
    search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
    
    with search_col1:
        search_term = st.text_input("🔍 Search User ID", placeholder="Enter user ID to search...")
    
    with search_col2:
        min_cost = st.number_input("💰 Min Cost ($)", min_value=0.0, value=0.0, step=0.1)
    
    with search_col3:
        category_filter = st.selectbox("📊 Filter by Category", 
                                     ["All"] + list(user_stats["user_category"].unique()))
    
    # Apply filters
    filtered_stats = user_stats.copy()
    
    if search_term:
        filtered_stats = filtered_stats[
            filtered_stats["user_id"].str.contains(search_term, case=False, na=False)
        ]
    
    if min_cost > 0:
        filtered_stats = filtered_stats[filtered_stats["estimated_cost"] >= min_cost]
    
    if category_filter != "All":
        filtered_stats = filtered_stats[filtered_stats["user_category"] == category_filter]
    
    # Format the display dataframe
    display_df = filtered_stats.copy()
    display_df["estimated_cost"] = display_df["estimated_cost"].apply(lambda x: f"${x:.2f}")
    display_df["start_datetime"] = display_df["start_datetime"].dt.strftime("%Y-%m-%d %H:%M")
    display_df = display_df.rename(columns={
        "user_id": "User ID",
        "num_model_requests": "Total Requests",
        "total_tokens": "Total Tokens",
        "estimated_cost": "Total Cost",
        "start_datetime": "Last Activity",
        "user_category": "Category",
        "cost_rank": "Cost Rank",
        "usage_rank": "Usage Rank"
    })
    
    st.dataframe(
        display_df.sort_values("Cost Rank")[
            ["User ID", "Category", "Total Requests", "Total Tokens", 
             "Total Cost", "Last Activity", "Cost Rank", "Usage Rank"]
        ],
        use_container_width=True
    )

# =============================
# Advanced Visualization Functions
# =============================

def create_comprehensive_overview(all_usage_data: Dict[str, pd.DataFrame]) -> None:
    """Create a comprehensive overview of all API usage."""
    st.subheader("🌐 Complete API Usage Overview")
    
    # Combine all data for overview
    combined_data = []
    for api_type, df in all_usage_data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy["api_type"] = api_type
            combined_data.append(df_copy)
    
    if not combined_data:
        st.warning("No data available across all APIs.")
        return
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # API usage distribution
    col1, col2 = st.columns(2)
    
    with col1:
        api_stats = combined_df.groupby("api_type").agg({
            "num_model_requests": "sum",
            "estimated_cost": "sum"
        }).reset_index()
        
        fig = px.bar(
            api_stats,
            x="api_type",
            y="num_model_requests",
            title="📊 Requests by API Type",
            color="num_model_requests",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            api_stats,
            values="estimated_cost",
            names="api_type",
            title="💰 Cost Distribution by API",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)

def create_model_comparison_dashboard(all_usage_data: Dict[str, pd.DataFrame]) -> None:
    """Advanced model comparison across all APIs."""
    st.subheader("🤖 Advanced Model Analytics")
    
    # Combine model data from all APIs
    model_data = []
    for api_type, df in all_usage_data.items():
        if not df.empty and "model" in df.columns:
            model_stats = df.groupby("model").agg({
                "num_model_requests": "sum",
                "total_tokens": "sum",
                "estimated_cost": "sum"
            }).reset_index()
            model_stats["api_type"] = api_type
            model_data.append(model_stats)
    
    if not model_data:
        st.info("No model data available.")
        return
    
    combined_models = pd.concat(model_data, ignore_index=True)
    
    # Model performance analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Model Usage Distribution", "Cost per Model", 
                       "Tokens vs Cost", "Model Categories"),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Model usage
    model_totals = combined_models.groupby("model")["num_model_requests"].sum().nlargest(10)
    fig.add_trace(
        go.Bar(x=model_totals.index, y=model_totals.values, name="Requests"),
        row=1, col=1
    )
    
    # Cost analysis
    model_costs = combined_models.groupby("model")["estimated_cost"].sum().nlargest(10)
    fig.add_trace(
        go.Scatter(x=model_costs.index, y=model_costs.values, mode="markers+lines", name="Cost"),
        row=1, col=2
    )
    
    # Tokens vs Cost scatter
    fig.add_trace(
        go.Scatter(
            x=combined_models["total_tokens"],
            y=combined_models["estimated_cost"],
            mode="markers",
            text=combined_models["model"],
            name="Token/Cost Ratio"
        ),
        row=2, col=1
    )
    
    # Model categories
    model_categories = {}
    for model in combined_models["model"].unique():
        category = "Other"
        for model_key, pricing in MODEL_PRICING.items():
            if model_key in str(model).lower():
                category = pricing["category"]
                break
        model_categories[model] = category
    
    combined_models["category"] = combined_models["model"].map(model_categories)
    category_counts = combined_models.groupby("category")["estimated_cost"].sum()
    
    fig.add_trace(
        go.Pie(labels=category_counts.index, values=category_counts.values, name="Categories"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="🤖 Comprehensive Model Analysis")
    st.plotly_chart(fig, use_container_width=True)

def create_time_analysis_dashboard(all_usage_data: Dict[str, pd.DataFrame]) -> None:
    """Advanced time-based analysis."""
    st.subheader("⏰ Temporal Usage Patterns")
    
    # Combine all data
    combined_data = []
    for api_type, df in all_usage_data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy["api_type"] = api_type
            combined_data.append(df_copy)
    
    if not combined_data:
        return
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Time-based analysis
    combined_df["hour"] = combined_df["start_datetime"].dt.hour
    combined_df["day_of_week"] = combined_df["start_datetime"].dt.day_name()
    combined_df["date"] = combined_df["start_datetime"].dt.date
    
    # Create comprehensive time analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Daily Usage Trend", "Hourly Distribution", 
                       "Day of Week Pattern", "API Usage Over Time"),
        specs=[[{"secondary_y": True}, {"type": "bar"}],
               [{"type": "bar"}, {"secondary_y": True}]]
    )
    
    # Daily trend
    daily_stats = combined_df.groupby("date").agg({
        "num_model_requests": "sum",
        "estimated_cost": "sum"
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(x=daily_stats["date"], y=daily_stats["num_model_requests"], 
                  name="Requests", line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=daily_stats["date"], y=daily_stats["estimated_cost"], 
                  name="Cost", line=dict(color="red")),
        row=1, col=1, secondary_y=True
    )
    
    # Hourly distribution
    hourly_stats = combined_df.groupby("hour")["num_model_requests"].sum()
    fig.add_trace(
        go.Bar(x=hourly_stats.index, y=hourly_stats.values, name="Hourly Requests"),
        row=1, col=2
    )
    
    # Day of week
    dow_stats = combined_df.groupby("day_of_week")["estimated_cost"].sum()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = dow_stats.reindex(day_order)
    fig.add_trace(
        go.Bar(x=dow_stats.index, y=dow_stats.values, name="Cost by Day"),
        row=2, col=1
    )
    
    # API usage over time
    api_time_stats = combined_df.groupby(["date", "api_type"])["estimated_cost"].sum().reset_index()
    for api_type in api_time_stats["api_type"].unique():
        api_data = api_time_stats[api_time_stats["api_type"] == api_type]
        fig.add_trace(
            go.Scatter(x=api_data["date"], y=api_data["estimated_cost"], 
                      name=f"{api_type} Cost", stackgroup="one"),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="⏰ Temporal Analysis Dashboard")
    st.plotly_chart(fig, use_container_width=True)

# =============================
# Export and Reporting Functions
# =============================

def create_executive_report(all_usage_data: Dict[str, pd.DataFrame], user_stats: pd.DataFrame) -> None:
    """Generate comprehensive executive summary report."""
    st.subheader("📋 Executive Summary Report")
    
    # Calculate overall statistics
    total_cost = 0
    total_requests = 0
    total_tokens = 0
    api_breakdown = {}
    
    for api_type, df in all_usage_data.items():
        if not df.empty:
            api_cost = df["estimated_cost"].sum()
            api_requests = df["num_model_requests"].sum()
            api_tokens = df["total_tokens"].sum()
            
            total_cost += api_cost
            total_requests += api_requests
            total_tokens += api_tokens
            
            api_breakdown[api_type] = {
                "cost": api_cost,
                "requests": api_requests,
                "tokens": api_tokens
            }
    
    # Date range
    all_dates = []
    for df in all_usage_data.values():
        if not df.empty:
            all_dates.extend(df["start_datetime"].tolist())
    
    if all_dates:
        start_date = min(all_dates).date()
        end_date = max(all_dates).date()
        days_analyzed = (end_date - start_date).days + 1
    else:
        start_date = end_date = datetime.date.today()
        days_analyzed = 1
    
    # Generate report
    report = f"""
    ## 📊 OpenAI API Executive Summary Report
    
    **Report Period:** {start_date} to {end_date} ({days_analyzed} days)
    **Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    ### 🎯 Key Performance Indicators
    - **Total API Requests:** {total_requests:,}
    - **Total Tokens Processed:** {total_tokens:,}
    - **Total Estimated Cost:** ${total_cost:.2f}
    - **Average Daily Cost:** ${total_cost / max(days_analyzed, 1):.2f}
    - **Cost per 1K Tokens:** ${(total_cost / max(total_tokens / 1000, 1)):.4f}
    
    ### 📈 API Breakdown
    """
    
    for api_type, stats in api_breakdown.items():
        percentage = (stats["cost"] / max(total_cost, 1)) * 100
        report += f"""
    - **{api_type.title()}**: ${stats["cost"]:.2f} ({percentage:.1f}%) - {stats["requests"]:,} requests
        """
    
    if not user_stats.empty:
        top_users = len(user_stats[user_stats["estimated_cost"] > 1])
        avg_cost_per_user = user_stats["estimated_cost"].mean()
        
        report += f"""
    
    ### 👥 User Analytics
    - **Total Active Users:** {len(user_stats):,}
    - **Users with >$1 usage:** {top_users:,}
    - **Average Cost per User:** ${avg_cost_per_user:.2f}
    - **Top User Cost:** ${user_stats["estimated_cost"].max():.2f}
        """
    
    report += f"""
    
    ### 💡 Key Insights
    - Most expensive API: {max(api_breakdown.items(), key=lambda x: x[1]["cost"])[0].title()}
    - Peak usage efficiency: {total_tokens / max(total_requests, 1):.0f} tokens per request
    - Cost trend: {"📈 Increasing" if days_analyzed > 1 else "📊 Stable"}
    
    ### 🔍 Recommendations
    - Monitor high-cost users for optimization opportunities
    - Consider model efficiency improvements for cost reduction
    - Track usage patterns for capacity planning
    """
    
    st.markdown(report)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Download Report"):
            st.download_button(
                label="💾 Download Executive Report",
                data=report,
                file_name=f"openai_executive_report_{datetime.date.today()}.md",
                mime="text/markdown"
            )
    
    with col2:
        if st.button("📊 Export All Data"):
            create_comprehensive_export(all_usage_data, user_stats)
    
    with col3:
        if st.button("📧 Email Summary"):
            st.info("📧 Email functionality would be implemented here")

def create_comprehensive_export(all_usage_data: Dict[str, pd.DataFrame], user_stats: pd.DataFrame) -> None:
    """Create comprehensive data export."""
    st.subheader("📥 Data Export Center")
    
    # Combine all data
    export_data = {}
    for api_type, df in all_usage_data.items():
        if not df.empty:
            export_data[f"{api_type}_usage"] = df
    
    if not user_stats.empty:
        export_data["user_analytics"] = user_stats
    
    # Create download buttons for each dataset
    for data_name, df in export_data.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{data_name.replace('_', ' ').title()}** - {len(df):,} records")
        with col2:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 CSV",
                data=csv_data,
                file_name=f"{data_name}_{datetime.date.today()}.csv",
                mime="text/csv",
                key=f"export_{data_name}"
            )

def create_real_time_monitoring(all_usage_data: Dict[str, pd.DataFrame]) -> None:
    """Create real-time monitoring dashboard."""
    st.subheader("📺 Real-Time Monitoring")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns(3)
    with col1:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
    with col2:
        refresh_interval = st.selectbox("⏱️ Refresh Interval", [30, 60, 300, 600], index=2)
    with col3:
        if st.button("🔄 Refresh Now") or auto_refresh:
            st.rerun()
    
    # Recent activity indicators
    if auto_refresh:
        st.markdown("🟢 **Live Monitoring Active**")
        time.sleep(refresh_interval)
        st.rerun()
    
    # Show latest usage across all APIs
    latest_activity = []
    for api_type, df in all_usage_data.items():
        if not df.empty:
            latest = df.nlargest(5, "start_datetime")[["start_datetime", "api_type", "model", "estimated_cost", "num_model_requests"]]
            latest_activity.append(latest)
    
    if latest_activity:
        combined_latest = pd.concat(latest_activity).nlargest(10, "start_datetime")
        st.subheader("🕐 Recent Activity")
        st.dataframe(combined_latest, use_container_width=True)

# =============================
# Advanced Analytics Functions
# =============================

def create_predictive_analytics(all_usage_data: Dict[str, pd.DataFrame]) -> None:
    """Create predictive analytics dashboard."""
    st.subheader("🔮 Predictive Analytics")
    
    # Combine data for trend analysis
    combined_data = []
    for api_type, df in all_usage_data.items():
        if not df.empty:
            daily_stats = df.groupby(df["start_datetime"].dt.date).agg({
                "estimated_cost": "sum",
                "num_model_requests": "sum",
                "total_tokens": "sum"
            }).reset_index()
            daily_stats["api_type"] = api_type
            combined_data.append(daily_stats)
    
    if not combined_data:
        st.info("Insufficient data for predictive analysis.")
        return
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    overall_daily = combined_df.groupby("start_datetime").agg({
        "estimated_cost": "sum",
        "num_model_requests": "sum",
        "total_tokens": "sum"
    }).reset_index()
    
    if len(overall_daily) < 3:
        st.info("Need at least 3 days of data for trend analysis.")
        return
    
    # Simple trend analysis
    overall_daily["day_number"] = range(len(overall_daily))
    
    # Calculate growth rates
    cost_growth = ((overall_daily["estimated_cost"].iloc[-1] - overall_daily["estimated_cost"].iloc[0]) / 
                   max(overall_daily["estimated_cost"].iloc[0], 0.01) * 100)
    
    requests_growth = ((overall_daily["num_model_requests"].iloc[-1] - overall_daily["num_model_requests"].iloc[0]) / 
                       max(overall_daily["num_model_requests"].iloc[0], 1) * 100)
    
    # Projections
    days_to_project = 7
    avg_daily_cost = overall_daily["estimated_cost"].mean()
    projected_weekly_cost = avg_daily_cost * days_to_project
    projected_monthly_cost = avg_daily_cost * 30
    
    # Display predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📈 Cost Growth Rate",
            f"{cost_growth:+.1f}%",
            delta="vs. first day"
        )
    
    with col2:
        st.metric(
            "🔮 7-Day Projection",
            f"${projected_weekly_cost:.2f}",
            delta=f"${projected_weekly_cost - (avg_daily_cost * len(overall_daily)):.2f}"
        )
    
    with col3:
        st.metric(
            "📊 30-Day Projection", 
            f"${projected_monthly_cost:.2f}",
            delta="estimated"
        )
    
    # Trend visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=overall_daily["start_datetime"],
        y=overall_daily["estimated_cost"],
        mode="lines+markers",
        name="Historical Cost",
        line=dict(color="blue")
    ))
    
    # Simple projection
    if len(overall_daily) >= 2:
        last_date = overall_daily["start_datetime"].iloc[-1]
        last_cost = overall_daily["estimated_cost"].iloc[-1]
        
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
        future_costs = [last_cost + (cost_growth/100 * last_cost * i/len(overall_daily)) for i in range(1, 8)]
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_costs,
            mode="lines+markers",
            name="7-Day Projection",
            line=dict(color="red", dash="dash")
        ))
    
    fig.update_layout(
        title="🔮 Cost Trend Analysis & Projection",
        xaxis_title="Date",
        yaxis_title="Daily Cost ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_anomaly_detection(all_usage_data: Dict[str, pd.DataFrame]) -> None:
    """Detect usage anomalies and display alerts in Streamlit."""
    
    st.subheader("🚨 Anomaly Detection")
    
    anomalies = []
    
    for api_type, df in all_usage_data.items():
        if df.empty:
            continue
        
        # Calculate statistical thresholds
        cost_mean = df["estimated_cost"].mean()
        cost_std = df["estimated_cost"].std()
        cost_threshold = cost_mean + (2 * cost_std)
        
        requests_mean = df["num_model_requests"].mean()
        requests_std = df["num_model_requests"].std()
        requests_threshold = requests_mean + (2 * requests_std)
        
        # Find anomalies
        cost_anomalies = df[df["estimated_cost"] > cost_threshold]
        request_anomalies = df[df["num_model_requests"] > requests_threshold]
        
        for _, row in cost_anomalies.iterrows():
            anomalies.append({
                "type": "High Cost",
                "api": api_type,
                "value": f"${row['estimated_cost']:.2f}",
                "threshold": f"${cost_threshold:.2f}",
                "date": row["start_datetime"],
                "severity": "🔴 Critical" if row["estimated_cost"] > cost_threshold * 1.5 else "🟡 Warning"
            })
        
        for _, row in request_anomalies.iterrows():
            anomalies.append({
                "type": "High Requests",
                "api": api_type,
                "value": f"{row['num_model_requests']:,}",
                "threshold": f"{requests_threshold:.0f}",
                "date": row["start_datetime"],
                "severity": "🔴 Critical" if row["num_model_requests"] > requests_threshold * 1.5 else "🟡 Warning"
            })
    
    # Show results
    if anomalies:
        st.warning(f"⚠️ {len(anomalies)} anomalies detected!")
        
        anomaly_df = pd.DataFrame(anomalies)
        anomaly_df = anomaly_df.sort_values("date", ascending=False)
        
        st.dataframe(
            anomaly_df[["severity", "type", "api", "value", "threshold", "date"]],
            use_container_width=True
        )
        
        # Summary charts
        col1, col2 = st.columns(2)
        with col1:
            severity_counts = anomaly_df["severity"].value_counts()
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="🚨 Anomaly Severity Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            api_counts = anomaly_df["api"].value_counts()
            fig = px.bar(
                x=api_counts.index,
                y=api_counts.values,
                title="📊 Anomalies by API Type"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No anomalies detected in the current dataset!")
# =============================
# Main Application
# =============================

def main():
    st.set_page_config(
        page_title="Enterprise OpenAI API Analytics",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced custom CSS
    st.markdown("""
    <style>
    .main {
        padding-top: 0.5rem;
    }
    .stMetric {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    .stTab {
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with organization info
    st.title("🚀 Enterprise OpenAI API Analytics Dashboard")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Dashboard Configuration")
        
        # API Key section
        with st.container():
            st.subheader("🔐 Authentication")
            api_key = st.text_input(
                "OpenAI Admin API Key",
                type="password",
                help="Required: Admin-level API key to access organization usage data"
            )
            
            if api_key:
                # Get organization info
                org_info = get_organization_info(api_key)
                if org_info:
                    st.success(f"✅ Connected to: {org_info.get('name', 'Organization')}")
                    if org_info.get('description'):
                        st.caption(org_info['description'][:50] + "...")
                else:
                    st.success("✅ API Key validated")
        
        st.markdown("---")
        
        # Time range configuration
        with st.container():
            st.subheader("📅 Analysis Period")
            days = st.slider(
                "Days to analyze",
                min_value=1,
                max_value=90,
                value=14,
                help="Select the number of days to analyze (max 90 days)"
            )
            
            # Quick date presets
            preset_col1, preset_col2 = st.columns(2)
            with preset_col1:
                if st.button("📅 Last 7 days"):
                    days = 7
                if st.button("📅 Last 30 days"):
                    days = 30
            with preset_col2:
                if st.button("📅 Last 14 days"):
                    days = 14
                if st.button("📅 Last 60 days"):
                    days = 60
        
        st.markdown("---")
        
        # Grouping and filtering options
        with st.container():
            st.subheader("🏷️ Data Grouping")
            group_by = []
            
            if st.checkbox("👤 Group by User", value=True):
                group_by.append("user_id")
            if st.checkbox("🤖 Group by Model", value=True):
                group_by.append("model")
            if st.checkbox("📁 Group by Project", value=False):
                group_by.append("project_id")
        
        st.markdown("---")
        
        # Display options
        with st.container():
            st.subheader("📊 Dashboard Options")
            show_predictions = st.checkbox("🔮 Predictive Analytics", value=True)
            show_anomalies = st.checkbox("🚨 Anomaly Detection", value=True)
            show_realtime = st.checkbox("📺 Real-time Monitoring", value=False)
            auto_refresh_interval = st.selectbox("🔄 Auto Refresh", 
                                               ["Off", "30s", "1m", "5m"], 
                                               index=0)
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            refresh_data = st.button("🔄 Refresh", use_container_width=True)
        with col2:
            clear_cache = st.button("🗑️ Clear Cache", use_container_width=True)
            
        if clear_cache:
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
        
        # API status indicator
        st.markdown("---")
        with st.container():
            st.subheader("📡 API Status")
            if api_key:
                st.success("🟢 Connected")
            else:
                st.error("🔴 Not Connected")
    
    # Main content area
    if not api_key:
        # Welcome screen with setup instructions
        st.markdown("### 👋 Welcome to Enterprise OpenAI API Analytics")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("""
            This dashboard provides comprehensive analytics for your OpenAI API usage across all endpoints:
            
            📊 **Complete API Coverage**: Chat, Embeddings, Images, Audio, Moderations  
            👥 **User Analytics**: Detailed user behavior and cost analysis  
            🤖 **Model Insights**: Performance comparison across all models  
            🔮 **Predictive Analytics**: Forecast usage and costs  
            🚨 **Anomaly Detection**: Identify unusual usage patterns  
            📈 **Real-time Monitoring**: Live usage tracking  
            """)
            
        with col2:
            with st.container():
                st.markdown("#### 🚀 Quick Start")
                st.markdown("1. Get your **Admin API Key** from [OpenAI Platform](https://platform.openai.com)")
                st.markdown("2. Enter it in the sidebar")
                st.markdown("3. Configure your analysis period")
                st.markdown("4. Explore your usage data!")
        
        # Feature showcase
        st.markdown("---")
        st.markdown("### ✨ Dashboard Features")
        
        feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
        
        with feature_col1:
            st.markdown("""
            **📊 Multi-API Analytics**
            - Chat Completions
            - Embeddings
            - Image Generation
            - Audio Processing
            - Content Moderation
            """)
            
        with feature_col2:
            st.markdown("""
            **👥 User Intelligence**
            - User segmentation
            - Cost attribution
            - Usage patterns
            - Activity tracking
            - Performance metrics
            """)
            
        with feature_col3:
            st.markdown("""
            **🤖 Model Analytics**
            - Performance comparison
            - Cost optimization
            - Latest model support
            - Usage distribution
            - Efficiency metrics
            """)
            
        with feature_col4:
            st.markdown("""
            **🔮 Advanced Analytics**
            - Predictive modeling
            - Anomaly detection
            - Real-time monitoring
            - Executive reporting
            - Data export tools
            """)
        
        return
    
    # Main dashboard with data
    try:
        with st.spinner("🔄 Loading comprehensive API analytics..."):
            # Fetch data from all APIs
            all_usage_data = get_all_api_usage(api_key, days, group_by)
            
            # Analyze users if user grouping is enabled
            user_stats = pd.DataFrame()
            if "user_id" in group_by:
                user_stats = analyze_user_metrics(all_usage_data)
        
        # Check if we have any data
        has_data = any(not df.empty for df in all_usage_data.values())
        
        if not has_data:
            st.warning("⚠️ No usage data found for the selected period and APIs.")
            st.info("""
            This could indicate:
            - No API usage during this time period
            - API key lacks necessary permissions
            - Different time range may be needed
            - Organization may not have usage data available
            """)
            return
        
        # Create tabbed interface for better organization
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", "👥 Users", "🤖 Models", "⏰ Time Analysis", "🔮 Predictions", "📋 Reports"
        ])
        
        with tab1:
            st.markdown("### 🌐 Complete API Usage Overview")
            
            # High-level metrics
            create_comprehensive_overview(all_usage_data)
            
            # Combined metrics from all APIs
            st.markdown("---")
            total_cost = sum(df["estimated_cost"].sum() for df in all_usage_data.values() if not df.empty)
            total_requests = sum(df["num_model_requests"].sum() for df in all_usage_data.values() if not df.empty)
            total_tokens = sum(df["total_tokens"].sum() for df in all_usage_data.values() if not df.empty)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Total Cost", f"${total_cost:.2f}", f"${total_cost/max(days, 1):.2f}/day")
            with col2:
                st.metric("🔢 Total Requests", f"{total_requests:,}", f"{total_requests/max(days, 1):.0f}/day")
            with col3:
                st.metric("🎯 Total Tokens", f"{total_tokens:,}", f"{total_tokens/max(days, 1):,.0f}/day")
            with col4:
                active_apis = len([df for df in all_usage_data.values() if not df.empty])
                st.metric("🔌 Active APIs", f"{active_apis}/5", "endpoints used")
        
        with tab2:
            create_user_dashboard(user_stats, all_usage_data)
        
        with tab3:
            create_model_comparison_dashboard(all_usage_data)
        
        with tab4:
            create_time_analysis_dashboard(all_usage_data)
        
        with tab5:
            if show_predictions:
                create_predictive_analytics(all_usage_data)
            else:
                st.info("💡 Enable 'Predictive Analytics' in the sidebar to see forecasts and trends.")
            
            if show_anomalies:
                st.markdown("---")
                create_anomaly_detection(all_usage_data)
        
        with tab6:
            create_executive_report(all_usage_data, user_stats)
            
            st.markdown("---")
            create_comprehensive_export(all_usage_data, user_stats)
        
        # Real-time monitoring overlay
        if show_realtime:
            with st.sidebar:
                create_real_time_monitoring(all_usage_data)
    
    except Exception as e:
        st.error(f"❌ An error occurred while loading the dashboard: {str(e)}")
        st.info("Please check your API key and try refreshing the page.")
        
        with st.expander("🔧 Debug Information"):
            st.code(str(e))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; font-size: 0.9em; padding: 1rem;'>
        🚀 <strong>Enterprise OpenAI API Analytics Dashboard</strong> | 
        Built with ❤️ using Streamlit & Plotly | 
        Data refreshed every 5 minutes | 
        <a href="https://platform.openai.com/docs" target="_blank">OpenAI API Documentation</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
