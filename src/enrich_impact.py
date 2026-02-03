import pandas as pd
import os
from .logging_config import logging


class ImpactDataErichment:
    """
    DataEnrichment

    Responsible for:
    - Adding new indicator/event records into the dataset
    - Preventing duplicate record IDs
    - Enforcing a consistent schema before export
    - Saving enriched data safely
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize enrichment object with an existing dataset.
        """

        try:
            if df is None or not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a valid pandas DataFrame.")

            # Reset record_id index if needed
            if df.index.name == "record_id":
                df = df.reset_index()

            self.df = df

            # Safe datetime conversion
            if "observation_date" in self.df.columns:
                self.df["observation_date"] = pd.to_datetime(
                    self.df["observation_date"],
                    errors="coerce"
                )

            logging.info("DataEnrichment initialized successfully.")

        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise

    # ---------------------------------------------------------
    # Enrich Dataset
    # ---------------------------------------------------------
    def enrich_data(self, new_records_list):
        """
        Append new records and remove duplicates.

        Parameters:
        - new_records_list (list of dicts)
        """

        try:
            if new_records_list is None or len(new_records_list) == 0:
                raise ValueError("New records list is empty or None.")

            if not isinstance(new_records_list, list):
                raise TypeError("New records must be provided as a list of dictionaries.")

            new_df = pd.DataFrame(new_records_list)

            if new_df.empty:
                raise ValueError("New records DataFrame is empty.")

            # Ensure record_id exists
            if "record_id" not in new_df.columns:
                raise KeyError("New records must contain a 'record_id' field.")

            # Convert observation_date safely
            if "observation_date" in new_df.columns:
                new_df["observation_date"] = pd.to_datetime(
                    new_df["observation_date"],
                    errors="coerce"
                )

            # Merge datasets
            self.df = pd.concat([self.df, new_df], ignore_index=True)

            # Remove duplicates by record_id
            self.df = self.df.drop_duplicates(subset=["record_id"], keep="last")

            print("\n--- Enrichment Success ---")
            print(f"Total Records After Enrichment: {len(self.df)}")

            logging.info("Enrichment completed successfully.")
            logging.info(f"Total Records: {len(self.df)}")

        except Exception as e:
            logging.error(f"Enrichment failed: {e}")
            print(f"❌ Enrichment Error: {e}")

    