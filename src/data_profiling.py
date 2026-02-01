import pandas as pd
import numpy as np
from IPython.display import display
from .logging_config import logging


class InclusionDataProfiler:
    """
    InclusionDataProfiler

    Performs dataset profiling for Ethiopia Financial Inclusion indicators:

    - Schema & pillar composition
    - Temporal coverage
    - Indicator completeness
    - Data quality (nulls + confidence)
    - Impact-event relationship validation
    """

    def __init__(self, df: pd.DataFrame, impact_df: pd.DataFrame = None):
        """
        Initialize the profiler with main dataset + optional impact relationship dataset.
        """

        try:
            if df is None or not isinstance(df, pd.DataFrame):
                raise ValueError("Input df must be a valid pandas DataFrame.")

            if df.empty:
                raise ValueError("Input df is empty. Profiling cannot proceed.")

            self.df = df
            self.impact_df = impact_df

            # Convert observation_date safely
            if "observation_date" in self.df.columns:
                self.df["observation_date"] = pd.to_datetime(
                    self.df["observation_date"],
                    errors="coerce"
                )

            logging.info("InclusionDataProfiler initialized successfully.")

        except Exception as e:
            logging.error(f"Profiler initialization failed: {e}")
            raise

    # --------------------------------------------------------
    # 1. Schema Overview
    # --------------------------------------------------------
    def schema_overview(self):
        """
        Displays dataset schema summary including record type and pillar distribution.
        """

        try:
            print("\n====== SCHEMA & RECORD TYPE SUMMARY ======")
            logging.info("Running schema overview...")

            if "record_type" not in self.df.columns:
                raise KeyError("Missing required column: record_type")

            summary = self.df.groupby("record_type").size().to_frame("Count")
            display(summary)

            print("\n--- Pillar Distribution ---")
            if "pillar" not in self.df.columns:
                raise KeyError("Missing required column: pillar")

            display(self.df["pillar"].value_counts().to_frame("Count"))

        except Exception as e:
            logging.error(f"Schema overview failed: {e}")
            print(f"❌ Schema Overview Error: {e}")

    # --------------------------------------------------------
    # 2. Temporal Coverage Analysis
    # --------------------------------------------------------
    def temporal_analysis(self):
        """
        Checks observation range and lists cataloged events.
        """

        try:
            print("\n====== TEMPORAL RANGE ======")
            logging.info("Running temporal analysis...")

            if "observation_date" not in self.df.columns:
                raise KeyError("Missing required column: observation_date")

            obs = self.df[self.df["record_type"] == "observation"]

            if not obs.empty:
                min_date = obs["observation_date"].min()
                max_date = obs["observation_date"].max()

                print(f"Observations Range: {min_date.date()} → {max_date.date()}")
                logging.info(f"Observation range: {min_date} to {max_date}")

            # Events section
            events = self.df[self.df["record_type"] == "event"]
            print(f"Total Events Cataloged: {len(events)}")
            logging.info(f"Total Events Cataloged: {len(events)}")

            if not events.empty:
                if "indicator" in events.columns:
                    display(
                        events[["indicator", "observation_date"]]
                        .sort_values("observation_date")
                    )

        except Exception as e:
            logging.error(f"Temporal analysis failed: {e}")
            print(f"❌ Temporal Analysis Error: {e}")

    # --------------------------------------------------------
    # 3. Indicator Coverage
    # --------------------------------------------------------
    def indicator_coverage(self):
        """
        Displays indicator-level data density and numeric statistics.
        """

        try:
            print("\n====== INDICATOR COVERAGE ======")
            logging.info("Running indicator coverage analysis...")

            required_cols = ["indicator", "value_numeric", "observation_date"]
            for col in required_cols:
                if col not in self.df.columns:
                    raise KeyError(f"Missing required column: {col}")

            coverage = self.df.groupby("indicator").agg(
                count=("value_numeric", "count"),
                mean=("value_numeric", "mean"),
                min=("value_numeric", "min"),
                max=("value_numeric", "max"),
                first_date=("observation_date", "min"),
                last_date=("observation_date", "max"),
            )

            display(coverage)

        except Exception as e:
            logging.error(f"Indicator coverage failed: {e}")
            print(f"❌ Indicator Coverage Error: {e}")

    # --------------------------------------------------------
    # 4. Missing Values + Confidence Review
    # --------------------------------------------------------
    def missing_value_summary(self):
        """
        Reports null values and confidence distribution.
        """

        try:
            print("\n====== DATA QUALITY (NULLS & CONFIDENCE) ======")
            logging.info("Running missing value + confidence profiling...")

            null_count = self.df.isnull().sum()
            null_percent = (null_count / len(self.df)) * 100

            quality_df = pd.DataFrame(
                {"Missing": null_count, "Percent (%)": null_percent}
            )

            display(quality_df[quality_df["Missing"] > 0])

            # Confidence distribution
            if "confidence" in self.df.columns:
                print("\n--- Confidence Level Distribution ---")
                display(self.df["confidence"].value_counts().to_frame("Count"))
            else:
                print("⚠️ Confidence column not found in dataset.")

        except Exception as e:
            logging.error(f"Missing value summary failed: {e}")
            print(f"❌ Data Quality Error: {e}")

    # --------------------------------------------------------
    # 5. Impact Link Relationship Review
    # --------------------------------------------------------
    def impact_link_review(self):
        """
        Reviews mapping between events and indicator impacts.
        """

        try:
            if self.impact_df is None:
                print("\n⚠️ No Impact Link dataset provided.")
                return

            if self.impact_df.empty:
                print("\n⚠️ Impact Link dataset is empty.")
                return

            required_cols = ["related_indicator", "impact_direction"]
            for col in required_cols:
                if col not in self.impact_df.columns:
                    raise KeyError(f"Missing required column in impact_df: {col}")

            print("\n====== IMPACT LINK RELATIONSHIPS ======")
            print(f"Total Relationships Captured: {len(self.impact_df)}")

            display(
                self.impact_df.groupby(
                    ["related_indicator", "impact_direction"]
                ).size().unstack(fill_value=0)
            )

            logging.info("Impact link review completed successfully.")

        except Exception as e:
            logging.error(f"Impact link review failed: {e}")
            print(f"❌ Impact Link Review Error: {e}")

    # --------------------------------------------------------
    # Run Full Profiling Suite
    # --------------------------------------------------------
    def run_all(self):
        """
        Executes all profiling modules sequentially.
        """

        try:
            print("\n🚀 Starting Full Data Profiling Suite...\n")
            logging.info("Full profiling suite started.")

            self.schema_overview()
            self.temporal_analysis()
            self.indicator_coverage()
            self.missing_value_summary()
            self.impact_link_review()

            print("\n✅ Profiling Completed Successfully.")
            logging.info("Full profiling suite completed successfully.")

        except Exception as e:
            logging.error(f"Run-all profiling failed: {e}")
            print(f"❌ Profiling Execution Error: {e}")
