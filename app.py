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
import random
from faker import Faker

# =============================
# Configuration & Constants
# =============================

# OpenAI model pricing (per 1k tokens) - Updated for 2025
MODEL_PRICING = {
    # GPT-4 Family
    "gpt-4": {"input": 0.03, "output": 0.06, "category": "chat", "tier": "premium"},
    "gpt-4-32k": {"input": 0.06, "output": 0.12, "category": "chat", "tier": "premium"},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03, "category": "chat", "tier": "premium"},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03, "category": "chat", "tier": "premium"},
    "gpt-4-vision-preview": {"input": 0.01, "output": 0.03, "category": "vision", "tier": "premium"},
    
    # GPT-4o Family (2024-2025 models)
    "gpt-4o": {"input": 0.005, "output": 0.015, "category": "chat", "tier": "standard"},
    "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01, "category": "chat", "tier": "standard"},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "category": "chat", "tier": "basic"},
    "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006, "category": "chat", "tier": "basic"},
    "gpt-4o-audio-preview": {"input": 0.1, "output": 0.2, "category": "audio", "tier": "premium"},
    
    # GPT-3.5 Family
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002, "category": "chat", "tier": "basic"},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004, "category": "chat", "tier": "standard"},
    "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002, "category": "completions", "tier": "basic"},
    
    # o1 Family (Reasoning models)
    "o1": {"input": 0.015, "output": 0.06, "category": "reasoning", "tier": "premium"},
    "o1-preview": {"input": 0.015, "output": 0.06, "category": "reasoning", "tier": "premium"},
    "o1-mini": {"input": 0.003, "output": 0.012, "category": "reasoning", "tier": "standard"},
    
    # Embedding Models
    "text-embedding-ada-002": {"input": 0.0001, "output": 0, "category": "embeddings", "tier": "basic"},
    "text-embedding-3-small": {"input": 0.00002, "output": 0, "category": "embeddings", "tier": "basic"},
    "text-embedding-3-large": {"input": 0.00013, "output": 0, "category": "embeddings", "tier": "standard"},
    
    # Image Generation
    "dall-e-3": {"input": 0.04, "output": 0, "category": "image_generation", "tier": "premium"},
    "dall-e-2": {"input": 0.02, "output": 0, "category": "image_generation", "tier": "standard"},
    
    # Audio Models
    "whisper-1": {"input": 0.006, "output": 0, "category": "audio", "tier": "standard"},
    "tts-1": {"input": 0.015, "output": 0, "category": "audio", "tier": "standard"},
    "tts-1-hd": {"input": 0.030, "output": 0, "category": "audio", "tier": "premium"},
    
    # Fine-tuning
    "babbage-002": {"input": 0.0016, "output": 0.0016, "category": "fine_tuning", "tier": "standard"},
    "davinci-002": {"input": 0.012, "output": 0.012, "category": "fine_tuning", "tier": "premium"},
}

# Subscription Plans
SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Free Tier",
        "monthly_limit": 18,
        "rate_limits": {"rpm": 3, "tpm": 40000},
        "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "color": "#6c757d",
        "icon": "🆓"
    },
    "pay_as_you_go": {
        "name": "Pay-as-you-go",
        "monthly_limit": None,
        "rate_limits": {"rpm": 5000, "tpm": 300000},
        "models": ["all"],
        "color": "#28a745",
        "icon": "💳"
    },
    "team": {
        "name": "Team Plan",
        "monthly_limit": 200,
        "rate_limits": {"rpm": 10000, "tpm": 500000},
        "models": ["all"],
        "monthly_fee": 25,
        "color": "#007bff",
        "icon": "👥"
    },
    "enterprise": {
        "name": "Enterprise",
        "monthly_limit": None,
        "rate_limits": {"rpm": 50000, "tpm": 2000000},
        "models": ["all"],
        "monthly_fee": 500,
        "color": "#6f42c1",
        "icon": "🏢"
    }
}

# Demo Organizations
DEMO_ORGS = [
    {
        "id": "org_demo_1",
        "name": "TechCorp Inc.",
        "plan": "enterprise",
        "industry": "Technology",
        "size": "Large (500+ employees)",
        "description": "Leading AI-powered software development company"
    },
    {
        "id": "org_demo_2", 
        "name": "StartupAI Ltd.",
        "plan": "team",
        "industry": "Startup",
        "size": "Small (10-50 employees)",
        "description": "Innovative AI chatbot development startup"
    },
    {
        "id": "org_demo_3",
        "name": "EduTech Solutions",
        "plan": "pay_as_you_go",
        "industry": "Education",
        "size": "Medium (100-500 employees)",
        "description": "Educational technology and e-learning platform"
    }
]

@dataclass
class UserProfile:
    user_id: str
    name: str
    email: str
    role: str
    department: str
    plan: str
    total_requests: int
    total_tokens: int
    total_cost: float
    most_used_model: str
    last_activity: datetime.datetime
    projects: List[str]
    efficiency_score: float
    cost_trend: str
    usage_category: str

@dataclass
class OrganizationMetrics:
    total_users: int
    active_users: int
    total_cost: float
    total_tokens: int
    top_models: Dict[str, int]
    plan_distribution: Dict[str, int]
    department_costs: Dict[str, float]

# =============================
# Demo Data Generation
# =============================

def generate_demo_users(count: int = 50) -> List[UserProfile]:
    """Generate realistic demo user profiles."""
    fake = Faker()
    departments = ["Engineering", "Marketing", "Sales", "Support", "Research", "Design", "Operations", "Legal"]
    roles = ["Developer", "Manager", "Analyst", "Director", "Intern", "Consultant", "Lead", "Specialist"]
    models = list(MODEL_PRICING.keys())
    plans = list(SUBSCRIPTION_PLANS.keys())
    
    users = []
    
    for i in range(count):
        # Generate user characteristics
        dept = random.choice(departments)
        role = random.choice(roles)
        plan = random.choices(plans, weights=[10, 40, 30, 20])[0]  # Weighted distribution
        
        # Generate realistic usage based on role and department
        base_requests = random.randint(50, 5000)
        if dept == "Engineering":
            base_requests *= random.uniform(2, 5)
        elif dept == "Research":
            base_requests *= random.uniform(1.5, 3)
        elif dept == "Marketing":
            base_requests *= random.uniform(1.2, 2.5)
        
        tokens_per_request = random.randint(500, 3000)
        total_tokens = int(base_requests * tokens_per_request)
        
        # Calculate cost based on most used model
        most_used_model = random.choice(models)
        model_pricing = MODEL_PRICING.get(most_used_model, {"input": 0.01, "output": 0.03})
        avg_cost_per_token = (model_pricing["input"] + model_pricing["output"]) / 2000  # Average
        total_cost = total_tokens * avg_cost_per_token
        
        # Add some variation
        total_cost *= random.uniform(0.7, 1.3)
        
        # Generate efficiency score
        efficiency = random.uniform(0.3, 0.95)
        if dept in ["Engineering", "Research"]:
            efficiency = max(efficiency, random.uniform(0.6, 0.9))
        
        # Determine usage category
        if total_cost > 500:
            category = "🔥 Power User"
        elif total_cost > 100:
            category = "📈 Heavy User"  
        elif total_cost > 20:
            category = "📊 Regular User"
        elif total_cost > 5:
            category = "🔍 Light User"
        else:
            category = "👤 Minimal User"
        
        # Cost trend
        trends = ["📈 Increasing", "📉 Decreasing", "➡️ Stable"]
        trend = random.choices(trends, weights=[45, 25, 30])[0]
        
        user = UserProfile(
            user_id=f"user_{i+1:03d}",
            name=fake.name(),
            email=fake.email(),
            role=role,
            department=dept,
            plan=plan,
            total_requests=int(base_requests),
            total_tokens=total_tokens,
            total_cost=round(total_cost, 2),
            most_used_model=most_used_model,
            last_activity=fake.date_time_between(start_date="-30d", end_date="now"),
            projects=[f"Project-{fake.word().title()}-{random.randint(1,99)}" for _ in range(random.randint(1,4))],
            efficiency_score=round(efficiency, 2),
            cost_trend=trend,
            usage_category=category
        )
        
        users.append(user)
    
    # Sort by cost descending
    users.sort(key=lambda x: x.total_cost, reverse=True)
    return users

def generate_demo_usage_data(days: int = 30) -> Dict[str, pd.DataFrame]:
    """Generate comprehensive demo usage data across all APIs."""
    fake = Faker()
    models = list(MODEL_PRICING.keys())
    users = [f"user_{i+1:03d}" for i in range(50)]
    projects = [f"proj_{i}" for i in range(10)]
    
    # Base date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    all_usage_data = {}
    
    # Generate data for each API type
    api_types = {
        "completions": {"weight": 50, "models": [m for m in models if MODEL_PRICING[m]["category"] == "chat"]},
        "embeddings": {"weight": 20, "models": [m for m in models if MODEL_PRICING[m]["category"] == "embeddings"]},
        "images": {"weight": 15, "models": [m for m in models if MODEL_PRICING[m]["category"] == "image_generation"]},
        "audio": {"weight": 10, "models": [m for m in models if MODEL_PRICING[m]["category"] == "audio"]},
        "moderations": {"weight": 5, "models": ["text-moderation-007"]}
    }
    
    for api_type, config in api_types.items():
        records = []
        daily_records = random.randint(50, 300) * config["weight"] // 100
        
        for day in range(days):
            current_date = start_date + datetime.timedelta(days=day)
            
            # Add weekend/weekday variation
            is_weekend = current_date.weekday() >= 5
            daily_multiplier = 0.3 if is_weekend else 1.0
            
            # Add time-of-day variation
            for hour in range(24):
                # Business hours get more activity
                if 9 <= hour <= 17:
                    hour_multiplier = random.uniform(1.5, 3.0)
                elif 18 <= hour <= 22:
                    hour_multiplier = random.uniform(0.8, 1.2)
                else:
                    hour_multiplier = random.uniform(0.1, 0.4)
                
                records_this_hour = max(1, int(daily_records * daily_multiplier * hour_multiplier / 24))
                
                for _ in range(records_this_hour):
                    record_time = current_date.replace(
                        hour=hour,
                        minute=random.randint(0, 59),
                        second=random.randint(0, 59)
                    )
                    
                    user_id = random.choice(users)
                    model = random.choice(config["models"]) if config["models"] else "unknown"
                    project_id = random.choice(projects)
                    
                    # Generate realistic token counts
                    if api_type == "completions":
                        input_tokens = random.randint(100, 4000)
                        output_tokens = random.randint(50, 2000)
                        total_tokens = input_tokens + output_tokens
                        
                        record = {
                            "api_type": api_type,
                            "start_time": int(record_time.timestamp()),
                            "end_time": int(record_time.timestamp()) + 3600,
                            "start_datetime": record_time,
                            "end_datetime": record_time + datetime.timedelta(hours=1),
                            "project_id": project_id,
                            "user_id": user_id,
                            "api_key_id": f"sk-{fake.lexify('?' * 20)}",
                            "model": model,
                            "num_model_requests": random.randint(1, 20),
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "input_cached_tokens": random.randint(0, input_tokens // 4),
                            "total_tokens": total_tokens,
                        }
                        
                    elif api_type == "embeddings":
                        input_tokens = random.randint(50, 1000)
                        record = {
                            "api_type": api_type,
                            "start_time": int(record_time.timestamp()),
                            "end_time": int(record_time.timestamp()) + 3600,
                            "start_datetime": record_time,
                            "end_datetime": record_time + datetime.timedelta(hours=1),
                            "project_id": project_id,
                            "user_id": user_id,
                            "api_key_id": f"sk-{fake.lexify('?' * 20)}",
                            "model": model,
                            "num_model_requests": random.randint(1, 100),
                            "input_tokens": input_tokens,
                            "output_tokens": 0,
                            "total_tokens": input_tokens,
                        }
                        
                    elif api_type == "images":
                        record = {
                            "api_type": api_type,
                            "start_time": int(record_time.timestamp()),
                            "end_time": int(record_time.timestamp()) + 3600,
                            "start_datetime": record_time,
                            "end_datetime": record_time + datetime.timedelta(hours=1),
                            "project_id": project_id,
                            "user_id": user_id,
                            "api_key_id": f"sk-{fake.lexify('?' * 20)}",
                            "model": model,
                            "num_model_requests": random.randint(1, 10),
                            "images_generated": random.randint(1, 4),
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        }
                        
                    elif api_type == "audio":
                        seconds = random.randint(30, 600)  # 30 seconds to 10 minutes
                        record = {
                            "api_type": api_type,
                            "start_time": int(record_time.timestamp()),
                            "end_time": int(record_time.timestamp()) + 3600,
                            "start_datetime": record_time,
                            "end_datetime": record_time + datetime.timedelta(hours=1),
                            "project_id": project_id,
                            "user_id": user_id,
                            "api_key_id": f"sk-{fake.lexify('?' * 20)}",
                            "model": model,
                            "num_model_requests": random.randint(1, 5),
                            "seconds": seconds,
                            "input_tokens": random.randint(0, 500),
                            "output_tokens": 0,
                            "total_tokens": random.randint(0, 500),
                        }
                    
                    else:  # moderations
                        input_tokens = random.randint(10, 500)
                        record = {
                            "api_type": api_type,
                            "start_time": int(record_time.timestamp()),
                            "end_time": int(record_time.timestamp()) + 3600,
                            "start_datetime": record_time,
                            "end_datetime": record_time + datetime.timedelta(hours=1),
                            "project_id": project_id,
                            "user_id": user_id,
                            "api_key_id": f"sk-{fake.lexify('?' * 20)}",
                            "model": "text-moderation-007",
                            "num_model_requests": random.randint(1, 50),
                            "input_tokens": input_tokens,
                            "output_tokens": 0,
                            "total_tokens": input_tokens,
                        }
                    
                    # Calculate cost
                    record["estimated_cost"] = calculate_api_cost(pd.Series(record), api_type)
                    records.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        all_usage_data[api_type] = df
    
    return all_usage_data

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

def create_demo_login_screen():
    """Create an enhanced demo login screen with organization selection."""
    st.markdown("### 🚀 Enterprise OpenAI API Analytics - Demo Mode")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **🎯 Explore comprehensive OpenAI API analytics with realistic demo data!**
        
        This demonstration showcases enterprise-level analytics across:
        - **50+ demo users** with realistic usage patterns
        - **5 API endpoints** (Chat, Embeddings, Images, Audio, Moderation)
        - **Multiple subscription plans** and cost structures
        - **Advanced analytics** including predictions and anomaly detection
        - **Executive reporting** and user management features
        """)
        
        # Organization selection
        st.markdown("#### 🏢 Select Demo Organization")
        
        org_options = {}
        for org in DEMO_ORGS:
            plan_info = SUBSCRIPTION_PLANS[org["plan"]]
            org_display = f"{plan_info['icon']} {org['name']} ({plan_info['name']})"
            org_options[org_display] = org
        
        selected_org_display = st.selectbox(
            "Choose your organization profile:",
            options=list(org_options.keys()),
            help="Each organization has different user counts, usage patterns, and subscription plans"
        )
        
        selected_org = org_options[selected_org_display]
        
        # Display organization details
        plan = SUBSCRIPTION_PLANS[selected_org["plan"]]
        
        st.markdown("---")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            **Organization Profile:**
            - **Industry:** {selected_org['industry']}
            - **Size:** {selected_org['size']}
            - **Plan:** {plan['icon']} {plan['name']}
            """)
        
        with col_b:
            monthly_fee = plan.get('monthly_fee', 0)
            monthly_limit = plan.get('monthly_limit', 'Unlimited')
            st.markdown(f"""
            **Subscription Details:**
            - **Monthly Fee:** ${monthly_fee}/mo
            - **Usage Limit:** ${monthly_limit if monthly_limit else 'Unlimited'}
            - **Rate Limit:** {plan['rate_limits']['rpm']:,} RPM
            """)
        
        # Enter demo mode button
        col_enter, col_real = st.columns(2)
        with col_enter:
            if st.button(f"🎯 Enter Demo Mode - {selected_org['name']}", use_container_width=True):
                st.session_state['demo_mode'] = True
                st.session_state['demo_org'] = selected_org
                st.rerun()
        
        with col_real:
            if st.button("🔑 Use Real API Key Instead", use_container_width=True):
                st.session_state['show_real_login'] = True
                st.rerun()
    
    with col2:
        # Plan comparison
        st.markdown("#### 📊 Subscription Plans")
        
        for plan_id, plan in SUBSCRIPTION_PLANS.items():
            with st.container():
                monthly_fee = plan.get('monthly_fee', 0)
                limit = plan.get('monthly_limit', '∞')
                
                if plan_id == selected_org["plan"]:
                    st.markdown(f"""
                    <div style="border: 2px solid {plan['color']}; padding: 1rem; border-radius: 0.5rem; background: {plan['color']}20;">
                        <h4>{plan['icon']} {plan['name']} ⭐</h4>
                        <p><strong>${monthly_fee}/mo</strong> | ${limit} limit</p>
                        <p>{plan['rate_limits']['rpm']:,} RPM</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="border: 1px solid #dee2e6; padding: 0.8rem; border-radius: 0.5rem;">
                        <h5>{plan['icon']} {plan['name']}</h5>
                        <p>${monthly_fee}/mo | ${limit} limit</p>
                        <p>{plan['rate_limits']['rpm']:,} RPM</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

def create_cost_analysis(user_stats: pd.DataFrame):
    """Create detailed cost analysis dashboard."""
    st.markdown("#### 💰 Cost Analysis & Optimization")
    
    # Cost distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Top spenders
        top_spenders = user_stats.nlargest(15, "total_cost")
        fig = px.bar(
            top_spenders,
            x="total_cost",
            y="name",
            orientation="h",
            title="💰 Top 15 Spenders",
            color="total_cost",
            color_continuous_scale="Reds",
            text="total_cost"
        )
        fig.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Cost by department and plan
        dept_plan_costs = user_stats.groupby(["department", "plan"])["total_cost"].sum().reset_index()
        fig = px.treemap(
            dept_plan_costs,
            path=[px.Constant("Organization"), "department", "plan"],
            values="total_cost",
            title="🏢 Cost Distribution by Department & Plan",
            color="total_cost",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost efficiency analysis
    st.markdown("---")
    st.markdown("#### ⚡ Cost Efficiency Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Efficiency vs Cost scatter
        fig = px.scatter(
            user_stats,
            x="efficiency_score",
            y="total_cost",
            color="department",
            size="total_tokens",
            hover_data=["name", "role"],
            title="⚡ Efficiency vs Cost",
            labels={"efficiency_score": "Efficiency Score", "total_cost": "Total Cost ($)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost per token by plan
        user_stats["cost_per_token"] = user_stats["total_cost"] / user_stats["total_tokens"].replace(0, 1)
        plan_efficiency = user_stats.groupby("plan").agg({
            "cost_per_token": "mean",
            "total_cost": "sum"
        }).reset_index()
        
        plan_labels = []
        for plan in plan_efficiency["plan"]:
            plan_info = SUBSCRIPTION_PLANS[plan]
            plan_labels.append(f"{plan_info['icon']} {plan_info['name']}")
        
        plan_efficiency["plan_label"] = plan_labels
        
        fig = px.bar(
            plan_efficiency,
            x="plan_label",
            y="cost_per_token",
            title="💳 Average Cost per Token by Plan",
            color="total_cost",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Monthly projections by department
        monthly_projections = user_stats.groupby("department")["total_cost"].sum() * 30 / 7  # Weekly to monthly
        fig = px.pie(
            values=monthly_projections.values,
            names=monthly_projections.index,
            title="📊 Projected Monthly Costs",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost optimization recommendations
    st.markdown("---")
    st.markdown("#### 💡 Cost Optimization Recommendations")
    
    # Identify optimization opportunities
    high_cost_low_efficiency = user_stats[
        (user_stats["total_cost"] > user_stats["total_cost"].quantile(0.8)) &
        (user_stats["efficiency_score"] < user_stats["efficiency_score"].quantile(0.5))
    ]
    
    if not high_cost_low_efficiency.empty:
        st.warning(f"⚠️ {len(high_cost_low_efficiency)} users have high costs but low efficiency scores")
        
        with st.expander("🎯 Users Needing Optimization"):
            for _, user in high_cost_low_efficiency.head(10).iterrows():
                potential_savings = user["total_cost"] * (0.8 - user["efficiency_score"])
                st.markdown(f"""
                **{user['name']}** ({user['department']})
                - Current Cost: ${user['total_cost']:,.2f}
                - Efficiency: {user['efficiency_score']:.1%}
                - Potential Monthly Savings: ${potential_savings * 4:.2f}
                """)

def create_enhanced_user_dashboard(demo_users: List[UserProfile] = None, all_usage_data: Dict[str, pd.DataFrame] = None) -> None:
    """Create comprehensive user analytics dashboard with plans and detailed insights."""
    st.subheader("👥 User Analytics & Management Dashboard")
    
    if demo_users:
        # Convert demo users to DataFrame for analysis
        user_data = []
        for user in demo_users:
            user_data.append({
                "user_id": user.user_id,
                "name": user.name,
                "email": user.email,
                "role": user.role,
                "department": user.department,
                "plan": user.plan,
                "total_requests": user.total_requests,
                "total_tokens": user.total_tokens,
                "total_cost": user.total_cost,
                "most_used_model": user.most_used_model,
                "last_activity": user.last_activity,
                "efficiency_score": user.efficiency_score,
                "cost_trend": user.cost_trend,
                "usage_category": user.usage_category,
                "projects": len(user.projects)
            })
        
        user_stats = pd.DataFrame(user_data)
    else:
        st.info("👤 No user data available. Enable demo mode or provide API key.")
        return
    
    # Enhanced overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_users = len(user_stats)
    active_users = len(user_stats[user_stats["total_cost"] > 1])
    total_cost = user_stats["total_cost"].sum()
    avg_efficiency = user_stats["efficiency_score"].mean()
    top_department = user_stats.groupby("department")["total_cost"].sum().idxmax()
    
    with col1:
        st.metric("👥 Total Users", f"{total_users:,}")
    with col2:
        st.metric("🎯 Active Users", f"{active_users:,}", f"{active_users/total_users*100:.1f}%")
    with col3:
        st.metric("💰 Total Spend", f"${total_cost:,.2f}")
    with col4:
        st.metric("⚡ Avg Efficiency", f"{avg_efficiency:.1%}")
    with col5:
        st.metric("🏆 Top Department", top_department)
    
    # User distribution and insights
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Plan distribution
        plan_counts = user_stats["plan"].value_counts()
        plan_data = []
        for plan_id, count in plan_counts.items():
            plan_info = SUBSCRIPTION_PLANS[plan_id]
            plan_data.append({
                "Plan": f"{plan_info['icon']} {plan_info['name']}",
                "Users": count,
                "Percentage": count / total_users * 100
            })
        
        fig_pie = px.pie(
            pd.DataFrame(plan_data),
            values="Users",
            names="Plan",
            title="📊 User Distribution by Plan",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Department costs
        dept_costs = user_stats.groupby("department")["total_cost"].sum().sort_values(ascending=True)
        fig_bar = px.bar(
            x=dept_costs.values,
            y=dept_costs.index,
            orientation="h",
            title="💼 Department Spending",
            color=dept_costs.values,
            color_continuous_scale="Viridis"
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col3:
        # Usage categories
        category_counts = user_stats["usage_category"].value_counts()
        fig_donut = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="🎯 User Categories",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    
    # Advanced user analytics
    st.markdown("---")
    st.subheader("🔍 Advanced User Analytics")
    
    # Create tabs for different views
    user_tab1, user_tab2, user_tab3, user_tab4 = st.tabs([
        "📋 User Directory", "💰 Cost Analysis", "📈 Performance", "🎯 Management"
    ])
    
    with user_tab1:
        create_user_directory(user_stats)
    
    with user_tab2:
        create_cost_analysis(user_stats)
    
    with user_tab3:
        create_performance_analysis(user_stats)
    
    with user_tab4:
        create_user_management(user_stats)

def create_user_directory(user_stats: pd.DataFrame):
    """Create searchable user directory with detailed profiles."""
    st.markdown("#### 👥 User Directory & Profiles")
    
    # Advanced search and filtering
    search_col1, search_col2, search_col3, search_col4 = st.columns(4)
    
    with search_col1:
        search_term = st.text_input("🔍 Search Users", placeholder="Name, email, or ID...")
    
    with search_col2:
        dept_filter = st.selectbox("🏢 Department", ["All"] + sorted(user_stats["department"].unique()))
    
    with search_col3:
        plan_filter = st.selectbox("📋 Plan", ["All"] + list(user_stats["plan"].unique()))
    
    with search_col4:
        category_filter = st.selectbox("🎯 Category", ["All"] + list(user_stats["usage_category"].unique()))
    
    # Apply filters
    filtered_users = user_stats.copy()
    
    if search_term:
        mask = (
            filtered_users["name"].str.contains(search_term, case=False, na=False) |
            filtered_users["email"].str.contains(search_term, case=False, na=False) |
            filtered_users["user_id"].str.contains(search_term, case=False, na=False)
        )
        filtered_users = filtered_users[mask]
    
    if dept_filter != "All":
        filtered_users = filtered_users[filtered_users["department"] == dept_filter]
    
    if plan_filter != "All":
        filtered_users = filtered_users[filtered_users["plan"] == plan_filter]
    
    if category_filter != "All":
        filtered_users = filtered_users[filtered_users["usage_category"] == category_filter]
    
    st.info(f"📊 Showing {len(filtered_users)} of {len(user_stats)} users")
    
    # Enhanced user cards
    for idx, user in filtered_users.head(20).iterrows():  # Show top 20
        plan_info = SUBSCRIPTION_PLANS[user["plan"]]
        
        with st.expander(f"👤 {user['name']} ({user['usage_category']})", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **👤 Profile Information**
                - **Name:** {user['name']}
                - **Email:** {user['email']}
                - **Role:** {user['role']}
                - **Department:** {user['department']}
                - **User ID:** `{user['user_id']}`
                """)
            
            with col2:
                st.markdown(f"""
                **💰 Usage & Costs**
                - **Total Spent:** ${user['total_cost']:,.2f}
                - **Total Requests:** {user['total_requests']:,}
                - **Total Tokens:** {user['total_tokens']:,}
                - **Efficiency Score:** {user['efficiency_score']:.1%}
                - **Cost Trend:** {user['cost_trend']}
                """)
            
            with col3:
                monthly_fee = plan_info.get('monthly_fee', 0)
                st.markdown(f"""
                **📋 Plan & Limits**
                - **Plan:** {plan_info['icon']} {plan_info['name']}
                - **Monthly Fee:** ${monthly_fee}
                - **Rate Limit:** {plan_info['rate_limits']['rpm']:,} RPM
                - **Favorite Model:** {user['most_used_model']}
                - **Projects:** {user['projects']} active
                """)
            
            # Usage visualization for this user
            if st.button(f"📊 View {user['name']}'s Analytics", key=f"analytics_{user['user_id']}"):
                st.session_state[f'show_user_details_{user["user_id"]}'] = True

def create_performance_analysis(user_stats: pd.DataFrame):
    """Create user performance analytics."""
    st.markdown("#### 📈 Performance & Productivity Analysis")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    avg_requests_per_user = user_stats["total_requests"].mean()
    top_performer = user_stats.loc[user_stats["efficiency_score"].idxmax(), "name"]
    most_active_dept = user_stats.groupby("department")["total_requests"].sum().idxmax()
    growth_users = len(user_stats[user_stats["cost_trend"] == "📈 Increasing"])
    
    with col1:
        st.metric("📊 Avg Requests/User", f"{avg_requests_per_user:,.0f}")
    with col2:
        st.metric("🏆 Top Performer", top_performer)
    with col3:
        st.metric("🚀 Most Active Dept", most_active_dept)
    with col4:
        st.metric("📈 Growing Users", f"{growth_users}")
    
    # Performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Efficiency distribution
        fig = px.histogram(
            user_stats,
            x="efficiency_score",
            nbins=20,
            title="⚡ Efficiency Score Distribution",
            color_discrete_sequence=["#1f77b4"]
        )
        fig.add_vline(x=user_stats["efficiency_score"].mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Average: {user_stats['efficiency_score'].mean():.1%}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Productivity by role
        role_productivity = user_stats.groupby("role").agg({
            "total_requests": "mean",
            "efficiency_score": "mean",
            "total_cost": "mean"
        }).reset_index()
        
        fig = px.scatter(
            role_productivity,
            x="total_requests",
            y="efficiency_score",
            size="total_cost",
            text="role",
            title="👔 Productivity by Role",
            labels={"total_requests": "Avg Requests", "efficiency_score": "Avg Efficiency"}
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    
    # Department performance comparison
    st.markdown("---")
    st.markdown("#### 🏢 Department Performance Comparison")
    
    dept_metrics = user_stats.groupby("department").agg({
        "total_cost": ["sum", "mean"],
        "total_requests": ["sum", "mean"],
        "total_tokens": ["sum", "mean"],
        "efficiency_score": "mean",
        "user_id": "count"
    }).round(2)
    
    dept_metrics.columns = ["Total Cost", "Avg Cost/User", "Total Requests", "Avg Requests/User", 
                           "Total Tokens", "Avg Tokens/User", "Avg Efficiency", "User Count"]
    
    # Add rankings
    dept_metrics["Cost Rank"] = dept_metrics["Total Cost"].rank(ascending=False).astype(int)
    dept_metrics["Efficiency Rank"] = dept_metrics["Avg Efficiency"].rank(ascending=False).astype(int)
    
    st.dataframe(
        dept_metrics.style.format({
            "Total Cost": "${:,.2f}",
            "Avg Cost/User": "${:,.2f}",
            "Total Requests": "{:,.0f}",
            "Avg Requests/User": "{:,.0f}",
            "Total Tokens": "{:,.0f}",
            "Avg Tokens/User": "{:,.0f}",
            "Avg Efficiency": "{:.1%}"
        }),
        use_container_width=True
    )

def create_user_management(user_stats: pd.DataFrame):
    """Create user management and administrative tools."""
    st.markdown("#### 🎯 User Management & Administration")
    
    # Management actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🔧 Bulk Actions")
        if st.button("📧 Send Usage Report to All Users"):
            st.success("📧 Usage reports sent to all users!")
        
        if st.button("⚠️ Alert High-Cost Users"):
            high_cost_users = user_stats[user_stats["total_cost"] > user_stats["total_cost"].quantile(0.9)]
            st.info(f"⚠️ Alerts sent to {len(high_cost_users)} high-cost users")
        
        if st.button("🎯 Optimize User Plans"):
            st.info("🎯 Plan optimization recommendations generated")
    
    with col2:
        st.markdown("##### 📊 Plan Management")
        if st.button("📈 Upgrade Eligible Users"):
            eligible_users = user_stats[
                (user_stats["plan"] == "free") & 
                (user_stats["total_cost"] > 5)
            ]
            st.info(f"📈 {len(eligible_users)} users eligible for upgrade")
        
        if st.button("💰 Review Plan Efficiency"):
            st.info("💰 Plan efficiency analysis completed")
        
        if st.button("🔄 Update Rate Limits"):
            st.success("🔄 Rate limits updated based on usage patterns")
    
    with col3:
        st.markdown("##### 🚨 Alerts & Monitoring")
        if st.button("🔴 Set Cost Alerts"):
            st.success("🔴 Cost alert thresholds configured")
        
        if st.button("📈 Enable Usage Monitoring"):
            st.success("📈 Real-time usage monitoring activated")
        
        if st.button("🎯 Generate Executive Report"):
            st.info("🎯 Executive report generated and emailed")

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
    
    # Enhanced API usage distribution
    col1, col2, col3 = st.columns(3)
    
    with col1:
        api_stats = combined_df.groupby("api_type").agg({
            "num_model_requests": "sum",
            "estimated_cost": "sum",
            "total_tokens": "sum"
        }).reset_index()
        
        fig = px.bar(
            api_stats,
            x="api_type",
            y="num_model_requests",
            title="📊 Requests by API Type",
            color="num_model_requests",
            color_continuous_scale="Blues",
            text="num_model_requests"
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
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
    
    with col3:
        fig = px.funnel(
            api_stats.sort_values("total_tokens", ascending=False),
            x="total_tokens",
            y="api_type",
            title="🎯 Token Usage Funnel",
            color="total_tokens",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================
# Main Application Entry Point
# =============================

def main():
    st.set_page_config(
        page_title="Enterprise OpenAI API Analytics - Enhanced",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced custom CSS with better styling
    st.markdown("""
    <style>
    .main {
        padding-top: 0.5rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 1.2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        color: white;
    }
    .stMetric label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500;
    }
    .stMetric .metric-value {
        color: white !important;
        font-weight: 700;
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    .stTab {
        font-size: 16px;
        font-weight: 600;
    }
    .user-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .plan-card {
        background: white;
        border: 2px solid #e9ecef;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .plan-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .demo-org-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'demo_mode' not in st.session_state:
        st.session_state['demo_mode'] = False
    if 'show_real_login' not in st.session_state:
        st.session_state['show_real_login'] = False
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = None
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🚀 Enterprise OpenAI API Analytics Dashboard</h1>
        <p style="font-size: 1.2em; color: #6c757d;">Comprehensive analytics, user management, and cost optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Authentication flow
    if not st.session_state['demo_mode'] and not st.session_state['api_key']:
        if st.session_state.get('show_real_login', False):
            st.info("Real API login functionality would go here")
        else:
            create_demo_login_screen()
    else:
        # Main dashboard content
        if st.session_state['demo_mode']:
            # Demo mode with enhanced features
            org = st.session_state['demo_org']
            
            # Enhanced sidebar for demo mode
            with st.sidebar:
                st.markdown(f"""
                <div class="demo-org-card">
                    <h3>{SUBSCRIPTION_PLANS[org['plan']]['icon']} {org['name']}</h3>
                    <p>{org['description']}</p>
                    <p><strong>Plan:</strong> {SUBSCRIPTION_PLANS[org['plan']]['name']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.header("⚙️ Dashboard Settings")
                
                # Time range
                days = st.slider("📅 Analysis Period (days)", 7, 90, 30)
                
                # Demo options
                st.subheader("🎯 Demo Features")
                user_count = st.slider("👥 Number of Demo Users", 20, 100, 50)
                show_advanced = st.checkbox("🔮 Advanced Analytics", value=True)
                
                st.markdown("---")
                
                # Plan information
                plan = SUBSCRIPTION_PLANS[org['plan']]
                st.markdown(f"""
                **📋 Current Plan Details**
                - **Monthly Fee:** ${plan.get('monthly_fee', 0)}
                - **Rate Limit:** {plan['rate_limits']['rpm']:,} RPM
                - **Token Limit:** {plan['rate_limits']['tpm']:,} TPM
                """)
                
                if st.button("🔄 Generate New Demo Data"):
                    st.cache_data.clear()
                    st.rerun()
            
            # Generate demo data
            with st.spinner("🎯 Loading demo analytics..."):
                demo_users = generate_demo_users(user_count)
                all_usage_data = generate_demo_usage_data(days)
            
            # Organization overview
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem;">
                <h2>{SUBSCRIPTION_PLANS[org['plan']]['icon']} {org['name']} - Analytics Overview</h2>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div><strong>Industry:</strong> {org['industry']}</div>
                    <div><strong>Size:</strong> {org['size']}</div>
                    <div><strong>Plan:</strong> {SUBSCRIPTION_PLANS[org['plan']]['name']}</div>
                    <div><strong>Period:</strong> Last {days} days</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics overview
            total_cost = sum(df["estimated_cost"].sum() for df in all_usage_data.values())
            total_requests = sum(df["num_model_requests"].sum() for df in all_usage_data.values())
            total_tokens = sum(df["total_tokens"].sum() for df in all_usage_data.values())
            active_users = len([u for u in demo_users if u.total_cost > 1])
            avg_efficiency = sum(u.efficiency_score for u in demo_users) / len(demo_users)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("👥 Total Users", f"{len(demo_users):,}", f"{active_users} active")
            with col2:
                st.metric("💰 Total Spend", f"${total_cost:,.2f}", f"${total_cost/days:.2f}/day")
            with col3:
                st.metric("🔢 API Requests", f"{total_requests:,.0f}", f"{total_requests/days:.0f}/day")
            with col4:
                st.metric("🎯 Tokens Used", f"{total_tokens:,.0f}M", f"{total_tokens/1000000:.1f}M")
            with col5:
                st.metric("⚡ Avg Efficiency", f"{avg_efficiency:.1%}")
            with col6:
                monthly_projection = total_cost * 30 / days
                st.metric("📊 Monthly Proj.", f"${monthly_projection:,.0f}")
            
            # Enhanced tabbed interface
            tab1, tab2, tab3, tab4 = st.tabs([
                "🌐 Overview", "👥 Users & Plans", "💰 Cost Analysis", "🎯 Management"
            ])
            
            with tab1:
                create_comprehensive_overview(all_usage_data)
            
            with tab2:
                create_enhanced_user_dashboard(demo_users, all_usage_data)
            
            with tab3:
                # Convert users to DataFrame for cost analysis
                user_data = []
                for user in demo_users:
                    user_data.append({
                        "name": user.name,
                        "department": user.department,
                        "plan": user.plan,
                        "total_cost": user.total_cost,
                        "total_tokens": user.total_tokens,
                        "efficiency_score": user.efficiency_score,
                        "cost_trend": user.cost_trend,
                        "role": user.role
                    })
                user_stats = pd.DataFrame(user_data)
                create_cost_analysis(user_stats)
            
            with tab4:
                st.info("🚧 Management features would be implemented here")

if __name__ == "__main__":
    main()
