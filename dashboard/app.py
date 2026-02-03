import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from io import BytesIO

# -------------------------------------------------------
# Fix imports
# -------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.forecast_usage_and_access import ForecastAccessAndUsage


class Dashboard:
    """Ethiopia Financial Inclusion Dashboard"""

    def __init__(self):
        st.set_page_config(
            page_title="Ethiopia FI Forecast Dashboard",
            layout="wide"
        )

        self.obs, self.events, self.impact_links = self.load_data()
        self.indicators = {
            "ACC_OWNERSHIP": {"name": "Account Ownership (%)", "unit": "% Adults"},
            "DIG_PAY": {"name": "Digital Payment Usage (%)", "unit": "% Adults"},
        }
        self.forecast_model = ForecastAccessAndUsage(
            observations_df=self.obs,
            events_df=self.events,
            impact_links_df=self.impact_links,
            indicators_metadata=self.indicators
        )

        # Sidebar navigation
        self.page = st.sidebar.radio(
            "📌 Dashboard Navigation",
            ["Overview", "Trends", "Forecasts", "Inclusion Projections", "Download Data"]
        )

        # Route page
        self.route_page()

    @staticmethod
    @st.cache_data
    def load_data():
        """Load datasets from Excel"""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        file_path = os.path.join(project_root, "data", "processed", "ethiopia_fi_unified_data.xlsx")

        main_df = pd.read_excel(file_path)
        impact_df = pd.read_excel(file_path, sheet_name="Impact_sheet")

        observations = main_df[main_df["record_type"] == "observation"].copy()
        events = main_df[main_df["record_type"] == "event"].copy()

        observations["observation_date"] = pd.to_datetime(observations["observation_date"])
        events["observation_date"] = pd.to_datetime(events["observation_date"])

        # Keep only gender=all for aggregated metrics
        observations_all = observations[observations["gender"] == "all"].copy()

        return observations_all, events, impact_df

    def route_page(self):
        if self.page == "Overview":
            self.page_overview()
        elif self.page == "Trends":
            self.page_trends()
        elif self.page == "Forecasts":
            self.page_forecasts()
        elif self.page == "Inclusion Projections":
            self.page_inclusion()
        elif self.page == "Download Data":
            self.page_download()

    # =======================================================
    # Overview Page
    # =======================================================
    def page_overview(self):
        st.title("🇪🇹 Ethiopia Financial Inclusion Overview")

        # Latest value per indicator
        latest_values = {}
        for code in self.indicators.keys():
            df_ind = self.obs[self.obs["indicator_code"] == code]
            if df_ind.empty:
                latest_values[code] = None
            else:
                latest_row = df_ind.loc[df_ind["observation_date"].idxmax()]
                latest_values[code] = latest_row["value_numeric"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Account Ownership (%)",
                    f"{latest_values['ACC_OWNERSHIP']:.1f}" if latest_values['ACC_OWNERSHIP'] else "No data")
        col2.metric("Digital Payment Usage (%)",
                    f"{latest_values['DIG_PAY']:.1f}" if latest_values['DIG_PAY'] else "No data")

        # P2P / ATM Crossover Ratio
        df_p2p = self.obs[self.obs["indicator_code"] == "USG_P2P_VALUE"]
        df_atm = self.obs[self.obs["indicator_code"] == "USG_ATM_VALUE"]
        p2p_sum = df_p2p["value_numeric"].sum() if not df_p2p.empty else 0
        atm_sum = df_atm["value_numeric"].sum() if not df_atm.empty else 0
        crossover_ratio = p2p_sum / atm_sum if atm_sum != 0 else 0
        col3.metric("P2P / ATM Crossover Ratio", f"{crossover_ratio:.2f}")

        st.divider()
        st.subheader("📈 Trends Over Time")

        df_plot = self.obs[self.obs["indicator_code"].isin(["ACC_OWNERSHIP", "DIG_PAY"])]
        fig = px.line(df_plot, x="observation_date", y="value_numeric",
                      color="indicator_code", markers=True,
                      title="Account Ownership vs Digital Payment Usage (Gender=All)")
        fig.update_xaxes(dtick="M12", tickformat="%Y")  # Only integer years
        st.plotly_chart(fig, use_container_width=True)

    # =======================================================
    # Trends Page
    # =======================================================
    def page_trends(self):
        st.title("📊 Historical Trends")

        indicator = st.selectbox("Select Indicator", list(self.indicators.keys()))
        df = self.obs[self.obs["indicator_code"] == indicator].copy()
        df = df.sort_values("observation_date")

        start_year, end_year = st.slider(
            "Select Year Range",
            int(df["observation_date"].dt.year.min()),
            int(df["observation_date"].dt.year.max()),
            (int(df["observation_date"].dt.year.min()), int(df["observation_date"].dt.year.max()))
        )

        filtered = df[(df["observation_date"].dt.year >= start_year) &
                      (df["observation_date"].dt.year <= end_year)]

        fig = px.line(filtered, x="observation_date", y="value_numeric", markers=True,
                      title=f"{self.indicators[indicator]['name']} Trend ({start_year}-{end_year})")
        fig.update_xaxes(dtick="M12", tickformat="%Y")
        st.plotly_chart(fig, use_container_width=True)

    # =======================================================
    # Forecasts Page
    # =======================================================
    def page_forecasts(self):
        st.title("🔮 Forecasts (2025–2027)")

        indicator = st.selectbox("Forecast Indicator", list(self.indicators.keys()))
        scenario = st.radio("Scenario", ["baseline", "with_events", "optimistic", "pessimistic"])

        self.forecast_model.generate_scenarios(indicator)
        df = self.forecast_model.scenario_forecasts[indicator][scenario]

        y_col = "baseline" if scenario == "baseline" else "forecast"

        fig = px.line(df, x="year", y=y_col, markers=True,
                      title=f"{self.indicators[indicator]['name']} Forecast ({scenario})")
        fig.update_xaxes(dtick=1)  # Integer years
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📌 Event Contributions")
        self.forecast_model.explain_contributions(indicator)

    # =======================================================
    # Inclusion Projections Page
    # =======================================================
    def page_inclusion(self):
        st.title("🎯 Inclusion Projections (2025–2027)")

        scenario = st.selectbox("Select Scenario", ["with_events", "optimistic", "pessimistic"])

        for indicator in ["ACC_OWNERSHIP", "DIG_PAY"]:
            st.subheader(f"{self.indicators[indicator]['name']} Projection")
            self.forecast_model.generate_scenarios(indicator)
            df = self.forecast_model.scenario_forecasts[indicator][scenario]

            fig = px.line(df, x="year", y="forecast", markers=True,
                          title=f"{self.indicators[indicator]['name']} Projection ({scenario})")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)

            if indicator == "ACC_OWNERSHIP":
                st.markdown("### Progress Toward 60% Target")
                target_df = pd.DataFrame({"year": df["year"], "target": [60] * len(df)})
                fig2 = px.line(target_df, x="year", y="target", markers=True, title="60% Target")
                fig2.update_xaxes(dtick=1)
                st.plotly_chart(fig2, use_container_width=True)

    # =======================================================
    # Download Page
    # =======================================================
    def page_download(self):
        st.title("⬇️ Download Datasets (Excel)")

        # Observations
        buffer_obs = BytesIO()
        with pd.ExcelWriter(buffer_obs, engine="xlsxwriter") as writer:
            self.obs.to_excel(writer, index=False, sheet_name="observations")
        st.download_button(
            "Download Observations (Excel)",
            data=buffer_obs.getvalue(),
            file_name="ethiopia_observations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Impact Links
        buffer_links = BytesIO()
        with pd.ExcelWriter(buffer_links, engine="xlsxwriter") as writer:
            self.impact_links.to_excel(writer, index=False, sheet_name="impact_links")
        st.download_button(
            "Download Impact Links (Excel)",
            data=buffer_links.getvalue(),
            file_name="ethiopia_impact_links.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# =======================================================
# Run Dashboard
# =======================================================
if __name__ == "__main__":
    Dashboard()
