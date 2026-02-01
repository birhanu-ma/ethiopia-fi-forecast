import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .logging_config import logging 


class EdaAnalysis:
    """
    Task 2: Exploratory Data Analysis for Ethiopia Financial Inclusion Forecasting
    """

    def __init__(self, filepath_or_df, sheet_name=None):
        """
        Initialize with either:
        - A pandas DataFrame (existing workflow), or
        - A path to an Excel file (new workflow) with optional sheet_name.
        """
        # If a string is passed, treat it as Excel file path
        if isinstance(filepath_or_df, str):
            self.df = pd.read_excel(filepath_or_df, sheet_name=sheet_name)
        else:
            # Assume it's a DataFrame
            self.df = filepath_or_df.copy()

        # -----------------------------
        # Safe datetime conversion
        # -----------------------------
        self.df["observation_date"] = pd.to_datetime(
            self.df["observation_date"], errors="coerce"
        )

        # -----------------------------
        # Safe year extraction
        # -----------------------------
        self.df["year"] = self.df["observation_date"].dt.year

        if "fiscal_year" in self.df.columns:
            self.df["year"] = self.df["year"].fillna(self.df["fiscal_year"])

        self.df["year"] = pd.to_numeric(self.df["year"], errors="coerce")

        print("EDA Class Initialized Successfully")
        logging.info("EDA Class Initialized Successfully")



    # ==========================================================
    # TASK 2.1 Dataset Overview
    # ==========================================================
    def get_dataset_overview(self):
        print("\n--- Dataset Overview ---\n")
        logging.info("Dataset Overview")


        record_summary = self.df['record_type'].value_counts()
        pillar_summary = self.df['pillar'].value_counts()
        source_summary = self.df['source_type'].value_counts()
    
        # Plot confidence distribution
        plt.figure(figsize=(7, 4))
        sns.countplot(data=self.df, x="confidence")
        plt.title("Confidence Level Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        return record_summary, pillar_summary, source_summary


    # ==========================================================
    # TASK 2.1 Temporal Coverage Heatmap (Square + Values)
    # ==========================================================
    def plot_temporal_coverage(self):

        obs_df = self.df[self.df["record_type"] == "observation"].copy()

        obs_df = obs_df.dropna(subset=["indicator_code", "year"])

        pivot = obs_df.pivot_table(
            index="indicator_code",
            columns="year",
            values="value_numeric",
            aggfunc="count"
        ).fillna(0)

        plt.figure(figsize=(10, 10))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            square=True,
            linewidths=0.5
        )

        plt.title("Temporal Coverage Heatmap (Observations per Year)")
        plt.xlabel("Year")
        plt.ylabel("Indicator Code")
        plt.tight_layout()
        plt.show()

    # ==========================================================
    # TASK 2.2 Access + Gender Gap Analysis
    # ==========================================================
    def plot_access_and_gender(self):

        acc = self.df[self.df["indicator_code"] == "ACC_OWNERSHIP"].copy()

        if acc.empty:
            print("ACC_OWNERSHIP data not available.")
            return

        acc = acc.dropna(subset=["year", "value_numeric"])
        acc = acc.sort_values("year")

        plt.figure(figsize=(10, 5))

        # National trend
        nat = acc[acc["gender"] == "all"]
        plt.plot(
            nat["year"],
            nat["value_numeric"],
            marker="o",
            linewidth=3,
            label="National Total"
        )

        # Gender disaggregation
        for g in ["male", "female"]:
            gdata = acc[acc["gender"] == g]
            if not gdata.empty:
                plt.plot(
                    gdata["year"],
                    gdata["value_numeric"],
                    marker="s",
                    linestyle="--",
                    label=f"{g.title()}"
                )

        plt.title("Account Ownership Trajectory (2011–2024)")
        plt.xlabel("Year")
        plt.ylabel("Account Ownership (%)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ==========================================================
    # Growth Rate Between Survey Years
    # ==========================================================
    def plot_growth_rates(self):

        acc = self.df[self.df["indicator_code"] == "ACC_OWNERSHIP"].copy()
        acc = acc[(acc["gender"] == "all")].dropna(subset=["year", "value_numeric"])

        acc = acc.sort_values("year")

        acc["growth_pp"] = acc["value_numeric"].diff()

        plt.figure(figsize=(8, 4))
        plt.bar(acc["year"], acc["growth_pp"])

        plt.title("Growth Rate in Account Ownership (pp change)")
        plt.xlabel("Year")
        plt.ylabel("Growth (percentage points)")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("\nGrowth Table:\n")
        print(acc[["year", "value_numeric", "growth_pp"]])
        logging.info("Growth Table")
        logging.info("[year, value_numeric, growth_pp]")
        

    # ==========================================================
    # TASK 2.3 Usage vs Registration Gap
    # ==========================================================
    def plot_usage_vs_registration(self):

        usage_codes = ["ACC_MM_ACCOUNT", "USG_MM_ACTIVE", "USG_DIGITAL_PAYMENT"]

        usage = self.df[self.df["indicator_code"].isin(usage_codes)].copy()

        if usage.empty:
            print("Usage indicators not found.")
            return

        usage = usage.dropna(subset=["year", "value_numeric"])
        usage = usage.sort_values("year")

        plt.figure(figsize=(10, 5))

        sns.barplot(
            data=usage,
            x="year",
            y="value_numeric",
            hue="indicator_code"
        )

        plt.title("Registered vs Active Usage Gap")
        plt.ylabel("Value (%)")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    

    # ==========================================================
    # TASK 2.5 Event Timeline (Clean + Readable)
    # ==========================================================
    def plot_event_timeline(self):

        events = self.df[self.df["record_type"] == "event"].copy()
        events = events.dropna(subset=["observation_date"])

        if events.empty:
            print("No events available.")
            return

        events["year"] = events["observation_date"].dt.year

        plt.figure(figsize=(12, 4))

        plt.scatter(events["year"], np.ones(len(events)), s=120)

        for _, row in events.iterrows():
            plt.text(
                row["year"],
                1.05,
                row["indicator"],
                rotation=45,
                ha="right",
                fontsize=8
            )

        plt.title("Timeline of Financial Inclusion Events in Ethiopia")
        plt.yticks([])
        plt.xlabel("Year")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ==========================================================
    # Event Overlay on Indicator Trend
    # ==========================================================
    def plot_event_impact_overlay(self, target_indicator="ACC_OWNERSHIP"):

        trend = self.df[
            (self.df["indicator_code"] == target_indicator)
            & (self.df["gender"] == "all")
        ].dropna(subset=["year", "value_numeric"])

        trend = trend.sort_values("year")

        if trend.empty:
            print("No trend data found.")
            return

        plt.figure(figsize=(12, 6))

        plt.plot(
            trend["year"],
            trend["value_numeric"],
            marker="o",
            linewidth=3,
            label="Account Ownership"
        )

        # Overlay events
        events = self.df[self.df["record_type"] == "event"].dropna(
            subset=["observation_date"]
        )

        for _, row in events.iterrows():
            yr = row["observation_date"].year

            plt.axvline(x=yr, linestyle="--", alpha=0.5)

        plt.title("Account Ownership with Event Overlay")
        plt.xlabel("Year")
        plt.ylabel("Account Ownership (%)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ==========================================================
    # Impact Link Summary
    # ==========================================================
    def summarize_impact_links(self):

        links = self.df[self.df["record_type"] == "impact_link"]

        if links.empty:
            print("No impact links found.")
            return

        summary = links.groupby(
            ["related_indicator", "impact_direction"]
        ).size().reset_index(name="count")

        print("\n--- Impact Link Summary ---\n")
        print(summary)
        logging.info("Impact Link Summary")
        logging.info("Summary")
# ==========================================================
    # TASK 2.4 Correlation Matrix (Clean, Squared, Readable)
    # ==========================================================
    def get_key_correlations(self, threshold=0.5):
        obs = self.df[self.df["record_type"] == "observation"]
        pivot = obs.pivot_table(
            index="year",
            columns="indicator_code",
            values="value_numeric",
            aggfunc="mean"
        ).dropna(axis=1, how='all') # Remove empty columns to reduce clutter

        corr = pivot.corr()

        # Increase figure size so labels have room to breathe
        plt.figure(figsize=(14, 12))

        # Create the heatmap
        ax = sns.heatmap(
            corr,
            annot=True,            # Show the numbers
            fmt=".2f",             # Two decimal places
            cmap="coolwarm",       # Blue-to-Red
            square=True,           # Keep it squared
            linewidths=0.5,        # THIN border for definition without clutter
            linecolor='#EEEEEE',   # Subtle light gray border
            center=0,
            cbar_kws={"shrink": .7},
            annot_kws={"size": 9}  # Slightly smaller font to fit inside squares
        )

        # FIXING THE LABELS FOR READABILITY
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        plt.title("Correlation Matrix: Financial Inclusion Indicators", fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.show()

        