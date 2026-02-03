import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from src.forecast_usage_and_access import ForecastAccessAndUsage

# ======================
# Load Data
# ======================
@st.cache_data
def load_data():
    obs = pd.read_csv("src/ethiopia_fi_unified_data.csv", parse_dates=["observation_date"])
    events = pd.read_csv("src/events.csv", parse_dates=["observation_date"])
    impact_links = pd.read_csv("src/impact_links.csv")
    return obs, events, impact_links

obs, events, links = load_data()

# Example indicator metadata
INDICATORS = {
    "ACC_OWNERSHIP": {"name": "Account Ownership", "unit": "% Adults"},
    "DIG_PAY": {"name": "Digital Payment Active Users", "unit": "% Adults"}
}

# ======================
# Initialize Forecast Model
# ======================
forecast_model = ForecastAccessAndUsage(obs, events, links, INDICATORS)

# ======================
# Sidebar
# ======================
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Trends", "Forecasts", "Inclusion Projections"])

# ======================
# Overview Page
# ======================
if page == "Overview":
    st.title("Financial Inclusion Overview")
    
    # Key Metrics
    latest_year = obs["observation_date"].dt.year.max()
    latest_data = obs[obs["observation_date"].dt.year == latest_year]
    
    acc_own = latest_data[latest_data["indicator_code"]=="ACC_OWNERSHIP"]["value_numeric"].mean()
    dp_active = latest_data[latest_data["indicator_code"]=="DIG_PAY"]["value_numeric"].mean()
    
    st.metric("Account Ownership", f"{acc_own:.1f}%")
    st.metric("Digital Payment  ", f"{dp_active:.1f}%")
    
    # P2P / ATM Crossover Ratio
    p2p_sum = latest_data[latest_data["indicator_code"]=="USG_P2P_VALUE"]["value_numeric"].sum()
    atm_sum = latest_data[latest_data["indicator_code"]=="USG_ATM_VALUE"]["value_numeric"].sum()
    crossover_ratio = p2p_sum / atm_sum if atm_sum != 0 else 0
    st.metric("P2P / ATM Crossover Ratio", f"{crossover_ratio:.2f}")
    
    st.markdown("### Growth Highlights")
    st.line_chart(latest_data.pivot(index="observation_date", columns="indicator_code", values="value_numeric"))

# ======================
# Trends Page
# ======================
elif page == "Trends":
    st.title("Historical Trends")
    
    indicator = st.selectbox("Select Indicator", list(INDICATORS.keys()))
    
    data = obs[obs["indicator_code"]==indicator]
    
    start_date, end_date = st.date_input("Select Date Range", [data["observation_date"].min(), data["observation_date"].max()])
    
    filtered = data[(data["observation_date"] >= pd.to_datetime(start_date)) & 
                    (data["observation_date"] <= pd.to_datetime(end_date))]
    
    fig = px.line(filtered, x="observation_date", y="value_numeric", title=f"{INDICATORS[indicator]['name']} over Time")
    st.plotly_chart(fig, use_container_width=True)

# ======================
# Forecasts Page
# ======================
elif page == "Forecasts":
    st.title("Forecasts")
    
    indicator = st.selectbox("Select Indicator for Forecast", list(INDICATORS.keys()))
    scenario = st.selectbox("Select Scenario", ["baseline", "with_events", "optimistic", "pessimistic"])
    
    # Generate forecasts if not done
    forecast_model.generate_scenarios(indicator)
    
    df = forecast_model.scenario_forecasts[indicator][scenario]
    
    fig = px.line(df, x="year", y="forecast", title=f"{INDICATORS[indicator]['name']} Forecast ({scenario})")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Event Contributions")
    forecast_model.explain_contributions(indicator)

# ======================
# Inclusion Projections Page
# ======================
elif page == "Inclusion Projections":
    st.title("Financial Inclusion Projections")
    
    indicator = "ACC_OWNERSHIP"
    forecast_model.generate_scenarios(indicator)
    
    df_base = forecast_model.scenario_forecasts[indicator]["with_events"]
    df_opt = forecast_model.scenario_forecasts[indicator]["optimistic"]
    df_pess = forecast_model.scenario_forecasts[indicator]["pessimistic"]
    
    fig = px.line(df_base, x="year", y="forecast", title="Financial Inclusion Projections")
    fig.add_scatter(x=df_opt["year"], y=df_opt["forecast"], mode="lines", name="Optimistic")
    fig.add_scatter(x=df_pess["year"], y=df_pess["forecast"], mode="lines", name="Pessimistic")
    st.plotly_chart(fig, use_container_width=True)
    
    # Target line
    st.markdown("**Progress toward 60% target**")
    target_line = pd.DataFrame({"year": df_base["year"], "target": [60]*len(df_base)})
    st.line_chart(target_line.set_index("year"))
