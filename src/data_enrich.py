import pandas as pd
import os
import sys

# Project root setup
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(project_root)

from .logging_config import logging


class DataEnrichment:
    """
    DataEnrichment Class

    Handles:
    - Adding new financial inclusion records into an existing dataset
    - Preventing duplicate record IDs
    - Saving enriched outputs safely
    - Providing summary statistics
    """

    def __init__(self, df):
        """
        Initialize the enrichment object with an existing DataFrame.
        Includes validation and date conversion.
        """

        try:
            if df is None or not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a valid pandas DataFrame.")

            # Fix: If record_id is stored as index, reset it back to column
            if df.index.name == "record_id":
                df = df.reset_index()

            self.df = df

            # Convert observation_date if available
            if "observation_date" in self.df.columns:
                self.df["observation_date"] = pd.to_datetime(
                    self.df["observation_date"],
                    errors="coerce"
                )

            logging.info("DataEnrichment initialized successfully.")

        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise

    def enrich_data(self, new_records_list):
        """
        Adds new records into the dataset safely.

        Parameters:
        - new_records_list (list of dicts): New financial inclusion rows
        """

        try:
            if new_records_list is None or len(new_records_list) == 0:
                raise ValueError("New records list is empty or None.")

            if not isinstance(new_records_list, list):
                raise TypeError("New records must be provided as a list of dictionaries.")

            # Convert new records into DataFrame
            new_df = pd.DataFrame(new_records_list)

            if new_df.empty:
                raise ValueError("New DataFrame is empty. No enrichment applied.")

            # Ensure record_id exists
            if "record_id" not in new_df.columns:
                raise KeyError("New records must include a 'record_id' field.")

            # Convert observation_date safely
            if "observation_date" in new_df.columns:
                new_df["observation_date"] = pd.to_datetime(
                    new_df["observation_date"],
                    errors="coerce"
                )

            # Append new records
            self.df = pd.concat([self.df, new_df], ignore_index=True)

            # Remove duplicates based on record_id
            self.df = self.df.drop_duplicates(subset=["record_id"], keep="last")

            # Log success
            print("\n--- Enrichment Success ---")
            print(f"Total Records After Enrichment: {len(self.df)}")

            logging.info("Enrichment completed successfully.")
            logging.info(f"Total Rows After Enrichment: {len(self.df)}")

        except Exception as e:
            logging.error(f"Enrichment failed: {e}")
            print(f"❌ Enrichment Error: {e}")

    def get_summary(self):
        """
        Returns a grouped summary of the dataset by record_type and pillar.
        """

        try:
            required_cols = ["record_type", "pillar"]

            for col in required_cols:
                if col not in self.df.columns:
                    raise KeyError(f"Missing required column: {col}")

            return self.df.groupby(["record_type", "pillar"]).size()

        except Exception as e:
            logging.error(f"Summary generation failed: {e}")
            print(f"❌ Summary Error: {e}")
            return None
