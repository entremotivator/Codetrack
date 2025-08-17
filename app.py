import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import json

# =============================
# Helper Functions
# =============================

def get_data(url, params, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_data = []
    page_cursor = None

    while True:
        if page_cursor:
            params["page"] = page_cursor
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data_json = response.json()
            all_data.extend(data_json.get("data", []))
            page_cursor = data_json.get("next_page")
            if not page_cursor:
                break
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            break

    return all_data


def parse_usage(usage_data):
    records = []
    for bucket in usage_data:
        start_time = bucket.get("start_time")
        end_time = bucket.get("end_time")
        for result in bucket.get("results", []):
            records.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "input_cached_tokens": result.get("input_cached_tokens", 0),
                    "input_audio_tokens": result.get("input_audio_tokens", 0),
                    "output_audio_tokens": result.get("output_audio_tokens", 0),
                    "num_model_requests": result.get("num_model_requests", 0),
                    "project_id": result.get("project_id"),
                    "user_id": result.get("user_id"),
                    "api_key_id": result.get("api_key_id"),
                    "model": result.get("model"),
                    "batch": result.get("batch"),
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        df["start_datetime"] = pd.to_datetime(df["start_time"], unit="s", errors="coerce")
        df["end_datetime"] = pd.to_datetime(df["end_time"], unit="s", errors="coerce")
    return df


def plot_token_usage(df):
    plt.figure(figsize=(12, 6))
    width = 0.35
    indices = range(len(df))

    plt.bar(indices, df["input_tokens"], width=width, label="Input Tokens", alpha=0.7)
    plt.bar(
        [i + width for i in indices],
        df["output_tokens"],
        width=width,
        label="Output Tokens",
        alpha=0.7,
    )

    plt.xlabel("Time Bucket")
    plt.ylabel("Number of Tokens")
    plt.title("Daily Input vs Output Token Usage")
    plt.xticks(
        [i + width / 2 for i in indices],
        [dt.strftime("%Y-%m-%d") for dt in df["start_datetime"]],
        rotation=45,
    )
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


def plot_model_distribution(df):
    if "model" in df.columns and not df["model"].isnull().all():
        model_counts = df.groupby("model")[["input_tokens", "output_tokens"]].sum()
        plt.figure(figsize=(6, 6))
        plt.pie(model_counts["input_tokens"] + model_counts["output_tokens"],
                labels=model_counts.index,
                autopct="%1.1f%%",
                startangle=140)
        plt.title("Token Distribution by Model")
        st.pyplot(plt)
    else:
        st.info("No model information available. Try setting group_by=['model'].")


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="OpenAI API Usage Dashboard", layout="wide")

st.title("📊 OpenAI API Usage & Cost Dashboard")

# Sidebar
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI **Admin API Key**", type="password")

days = st.sidebar.slider("Days to track", min_value=1, max_value=90, value=30)
group_by_model = st.sidebar.checkbox("Group by model", value=False)

if api_key:
    st.success("API Key set ✅")

    # Usage API endpoint
    url = "https://api.openai.com/v1/organization/usage/completions"
    start_time = int(time.time()) - (days * 24 * 60 * 60)

    params = {
        "start_time": start_time,
        "bucket_width": "1d",
        "limit": days,
    }
    if group_by_model:
        params["group_by"] = ["model"]

    usage_data = get_data(url, params, api_key)
    df = parse_usage(usage_data)

    if not df.empty:
        st.subheader("Usage Data")
        st.dataframe(df)

        # Token usage chart
        st.subheader("📈 Token Usage Over Time")
        plot_token_usage(df)

        # Model distribution
        st.subheader("🔄 Model Distribution")
        plot_model_distribution(df)

        # Cost API (very similar to usage API)
        st.subheader("💰 API Cost Estimation")
        cost_url = "https://api.openai.com/v1/organization/costs"
        cost_params = {"start_time": start_time, "limit": days}
        cost_data = get_data(cost_url, cost_params, api_key)
        if cost_data:
            st.json(cost_data[:3])  # show a preview
        else:
            st.info("No cost data available.")
    else:
        st.warning("No usage data available for the selected period.")
else:
    st.warning("Please enter your API key in the sidebar to fetch data.")
