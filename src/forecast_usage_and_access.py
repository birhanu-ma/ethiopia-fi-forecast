import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from IPython.display import display


class ForecastAccessAndUsage:
    """
    Task 4 Forecasting Model (2025–2027)

    Supports:
    -------------------
    ✅ Baseline linear trend forecast
    ✅ Event-augmented forecast using impact_links
    ✅ Scenario forecasting (baseline / optimistic / pessimistic)
    ✅ Visualization
    ✅ Contribution explanation
    """

    def __init__(self, observations_df, events_df, impact_links_df, indicators_metadata,
                 forecast_years=None):

        self.obs = observations_df.copy()
        self.events = events_df.copy()
        self.links = impact_links_df.copy()

        self.indicators = indicators_metadata
        self.forecast_years = forecast_years or [2025, 2026, 2027]

        # Ensure datetime conversion
        self.obs["observation_date"] = pd.to_datetime(self.obs["observation_date"])
        self.events["observation_date"] = pd.to_datetime(self.events["observation_date"])

        # Outputs
        self.baseline_forecasts = {}
        self.event_forecasts = {}
        self.scenario_forecasts = {}

    # =====================================================
    # 1. Baseline Forecast
    # =====================================================
    def fit_baseline(self, indicator_code):
        """
        Baseline linear regression forecast using only National Total (gender='all')
        """
        # 1. Initial filter for the indicator
        hist = self.obs[self.obs["indicator_code"] == indicator_code].copy()
    
        # 2. THE FIX: Filter out male/female rows to remove the vertical 'stack'
        # This ensures rows like REC_0004 and REC_0005 are ignored
        if 'gender' in hist.columns:
            hist = hist[hist['gender'] == 'all']
    
        # 3. Handle the year extraction
        # Using .dt.year is good, but we ensure it handles the 2024 'all' string properly
        hist["year"] = pd.to_datetime(hist["observation_date"]).dt.year
        
        # 4. Prepare for Regression
        X = hist["year"].values.reshape(-1, 1)
        y = hist["value_numeric"].values
    
        # 5. Fit and Predict
        model = LinearRegression()
        model.fit(X, y)
    
        future_X = np.array(self.forecast_years).reshape(-1, 1)
        pred = model.predict(future_X)
    
        df = pd.DataFrame({
            "year": self.forecast_years,
            "indicator_code": indicator_code,
            "baseline": pred
        })
    
        self.baseline_forecasts[indicator_code] = df
        return df

    # =====================================================
    # 2. Event-Augmented Forecast (No ImpactModel Needed)
    # =====================================================
    def fit_event_augmented(self, indicator_code):
        """
        Forecast baseline + add event impacts from impact_links
        """

        if indicator_code not in self.baseline_forecasts:
            self.fit_baseline(indicator_code)

        baseline = self.baseline_forecasts[indicator_code].copy()

        impacts = []

        # Filter links relevant to this indicator
        relevant_links = self.links[self.links["related_indicator"] == indicator_code]

        for _, link in relevant_links.iterrows():

            event_id = link["parent_id"]
            impact = link["impact_estimate"]
            lag_months = link["lag_months"]

            # Find event date
            event_row = self.events[self.events["record_id"] == event_id]

            if event_row.empty:
                continue

            event_date = event_row.iloc[0]["observation_date"]
            event_year = event_date.year + int(lag_months / 12)

            # Apply effect only after lag
            for year in self.forecast_years:
                if year >= event_year:
                    impacts.append((year, impact))

        # Aggregate yearly impacts
        impact_df = pd.DataFrame(impacts, columns=["year", "event_effect"])
        impact_sum = impact_df.groupby("year")["event_effect"].sum().reset_index()

        merged = baseline.merge(impact_sum, on="year", how="left")
        merged["event_effect"] = merged["event_effect"].fillna(0)

        merged["forecast"] = merged["baseline"] + merged["event_effect"]

        self.event_forecasts[indicator_code] = merged
        return merged

    # =====================================================
    # 3. Scenario Forecasts
    # =====================================================
    def generate_scenarios(self, indicator_code):
    
        base = self.fit_baseline(indicator_code)
        with_events = self.fit_event_augmented(indicator_code)
    
        optimistic = with_events.copy()
        optimistic["forecast"] = optimistic["baseline"] + optimistic["event_effect"] * 1.2
    
        pessimistic = with_events.copy()
        pessimistic["forecast"] = pessimistic["baseline"] + pessimistic["event_effect"] * 0.6
    
        optimistic["forecast"] = optimistic["forecast"].clip(0, 100)
        pessimistic["forecast"] = pessimistic["forecast"].clip(0, 100)
    
        self.scenario_forecasts[indicator_code] = {
            "baseline": base,
            "with_events": with_events,
            "optimistic": optimistic,
            "pessimistic": pessimistic
        }
    
        return self.scenario_forecasts[indicator_code]
    
    # =====================================================
    # 4. Plot Forecasts
    # =====================================================
    # =====================================================
    # 4. Plot Forecasts (Corrected for Gender Stacking)
    # =====================================================
    def plot_forecasts(self, indicator_code):
        """
        Visualizes historical data (filtered for 'all' gender) 
        alongside various forecast scenarios.
        """
        if indicator_code not in self.scenario_forecasts:
            raise ValueError(f"Run generate_scenarios('{indicator_code}') first")
    
        plt.figure(figsize=(12, 6))
    
        # --- 1. Filter and Prepare Historical Data ---
        hist = self.obs[self.obs["indicator_code"] == indicator_code].copy()
        
        # Filter for 'all' to remove the vertical gender stack in 2021
        if 'gender' in hist.columns:
            hist = hist[hist['gender'] == 'all']
            
        # Ensure chronological order so the line doesn't zig-zag
        hist = hist.sort_values("observation_date")
        hist["year"] = hist["observation_date"].dt.year
    
        # Plot clean historical line
        plt.plot(
            hist["year"],
            hist["value_numeric"],
            marker="o",
            label="Historical (National Total)",
            linewidth=3,
            color="#1f77b4" # Strong blue
        )
    
        # --- 2. Plot Scenario DataFrames ---
        scenarios = self.scenario_forecasts[indicator_code]
        
        # Color mapping for clarity
        colors = {
            "baseline": "#ff7f0e",    # Orange
            "with_events": "#2ca02c", # Green
            "optimistic": "#d62728",  # Red
            "pessimistic": "#9467bd"  # Purple
        }
    
        for scen_name, df in scenarios.items():
            df = df.sort_values("year")
    
            # Use 'forecast' column if it exists, otherwise use 'baseline'
            y_values = df["forecast"] if "forecast" in df.columns else df["baseline"]
    
            plt.plot(
                df["year"],
                y_values,
                marker="o",
                linestyle="--", # Dashed line for forecasts to distinguish from history
                linewidth=2,
                label=scen_name.replace("_", " ").title(),
                color=colors.get(scen_name)
            )
    
        # --- 3. Formatting ---
        plt.title(f"Forecast for {indicator_code} (Cleaned Trend)", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("% Adults", fontsize=12)
        plt.ylim(0, 100) # Percentages usually look better 0-100
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left")
        
        # Tight layout helps prevent label clipping
        plt.tight_layout()
        plt.show()

    # =====================================================
    # 5. Explain Contributions
    # =====================================================
    def explain_contributions(self, indicator_code):

        links = self.links[self.links["related_indicator"] == indicator_code]

        if links.empty:
            print("No event impacts found.")
            return

        display(links[[
            "parent_id",
            "impact_estimate",
            "lag_months",
            "confidence",
            "evidence_basis"
        ]])

    # =====================================================
    # 6. Forecast Summary Table
    # =====================================================
    def forecast_summary(self, year, indicator_code):
        """
        Compare forecasts across scenarios for a given year
        """
        if indicator_code not in self.scenario_forecasts:
            raise ValueError(f"Run generate_scenarios('{indicator_code}') first")
    
        summary = []
    
        # Get the scenario dict for this indicator
        scenarios = self.scenario_forecasts[indicator_code]
    
        for scen_name, df in scenarios.items():
            row = df[df["year"] == year]
    
            if not row.empty:
                # Safely get the single value
                forecast_value = row["forecast"].iloc[0] if "forecast" in row else row["baseline"].iloc[0]
    
                summary.append({
                    "Scenario": scen_name,
                    "Forecast": float(forecast_value)
                })
    
        summary_df = pd.DataFrame(summary)
        display(summary_df)
        return summary_df
    