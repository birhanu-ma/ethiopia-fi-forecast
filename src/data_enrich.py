import pandas as pd

class DataEnrichment:
    def __init__(self, df):
        # FIX: If record_id is the index, move it back to being a column
        if df.index.name == 'record_id':
            df = df.reset_index()
        self.df = df
        
        if 'observation_date' in self.df.columns:
            self.df['observation_date'] = pd.to_datetime(self.df['observation_date'])

    def enrich_data(self, new_records_list):
        new_df = pd.DataFrame(new_records_list)
        
        if 'observation_date' in new_df.columns:
            new_df['observation_date'] = pd.to_datetime(new_df['observation_date'])
        
        # FIX: ignore_index=True resets the row numbers (0, 1, 2...) 
        # but keeps your 'record_id' column safe
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        
        # Clean up duplicates just in case you run the cell twice
        self.df = self.df.drop_duplicates(subset=['record_id'], keep='last')
        
        print(f"--- Enrichment Success ---")
        print(f"Current Columns: {list(self.df.columns)}")
        print(f"Total Rows: {len(self.df)}")

    def save_to_csv(self, filename="ethiopia_fi_enriched.csv"):
        # CRITICAL: index=False ensures we don't save the row numbers
        # which keeps 'record_id' as your first real column.
        self.df.to_csv(filename, index=False)
        print(f"File saved as {filename}")

    def get_summary(self):
        return self.df.groupby(['record_type', 'pillar']).size()