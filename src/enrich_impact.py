import pandas as pd
import os

class DataEnrichment:
    def __init__(self, df):
        # Move record_id to column if it's the index
        if df is not None and df.index.name == 'record_id':
            df = df.reset_index()
        self.df = df
        
        # Convert dates to datetime objects
        if self.df is not None and 'observation_date' in self.df.columns:
            self.df['observation_date'] = pd.to_datetime(self.df['observation_date'])

    def enrich_data(self, new_records_list):
        new_df = pd.DataFrame(new_records_list)
        if 'observation_date' in new_df.columns:
            new_df['observation_date'] = pd.to_datetime(new_df['observation_date'])
        
        # Merge and remove duplicates
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.df = self.df.drop_duplicates(subset=['record_id'], keep='last')
        
        print(f"--- Enrichment Success ---")
        print(f"Total Records: {len(self.df)}")

    def save_to_csv(self, filename):
        # Create directory if missing
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Reorder/Reindex to standard 34 columns before saving
        schema = [
            "record_id", "parent_id", "record_type", "category", "pillar", "indicator", 
            "indicator_code", "indicator_direction", "value_numeric", "value_text", 
            "value_type", "unit", "observation_date", "period_start", "period_end", 
            "fiscal_year", "gender", "location", "region", "source_name", "source_type", 
            "source_url", "confidence", "related_indicator", "relationship_type", 
            "impact_direction", "impact_magnitude", "impact_estimate", "lag_months", 
            "evidence_basis", "comparable_country", "collected_by", "collection_date", 
            "original_text", "notes"
        ]
        self.df = self.df.reindex(columns=schema)
        self.df.to_csv(filename, index=False)
        print(f"✅ File saved as: {filename}")