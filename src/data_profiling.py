import pandas as pd
import numpy as np
from IPython.display import display

class InclusionDataProfiler:
    def __init__(self, df: pd.DataFrame, impact_df: pd.DataFrame = None):
        self.df = df
        self.impact_df = impact_df
        # Ensure date conversion
        if 'observation_date' in self.df.columns:
            self.df['observation_date'] = pd.to_datetime(self.df['observation_date'])

    def schema_overview(self):
        print("\n====== SCHEMA & RECORD TYPE SUMMARY ======")
        # Count records by type (Task 1.2 requirement)
        summary = self.df.groupby('record_type').size().to_frame('Count')
        display(summary)
        
        print("\n--- Pillar Distribution ---")
        display(self.df['pillar'].value_counts().to_frame('Count'))

    def temporal_analysis(self):
        print("\n====== TEMPORAL RANGE ======")
        obs = self.df[self.df['record_type'] == 'observation']
        if not obs.empty:
            print(f"Observations Range: {obs['observation_date'].min().date()} to {obs['observation_date'].max().date()}")
        
        events = self.df[self.df['record_type'] == 'event']
        print(f"Total Events Cataloged: {len(events)}")
        display(events[['indicator', 'observation_date']].sort_values('observation_date'))

    def indicator_coverage(self):
        print("\n====== INDICATOR COVERAGE ======")
        # Unique indicators and their data point counts
        coverage = self.df.groupby('indicator').agg({
            'value_numeric': ['count', 'mean', 'min', 'max'],
            'observation_date': ['min', 'max']
        })
        display(coverage)

    def missing_value_summary(self):
        print("\n====== DATA QUALITY (NULLS & CONFIDENCE) ======")
        null_count = self.df.isnull().sum()
        null_percent = (null_count / len(self.df)) * 100
        
        quality_df = pd.DataFrame({'Missing': null_count, '%': null_percent})
        display(quality_df[quality_df['Missing'] > 0])
        
        print("\n--- Confidence Level Distribution ---")
        display(self.df['confidence'].value_counts().to_frame('Count'))

    def impact_link_review(self):
        if self.impact_df is None:
            print("\n!!! No Impact Link data provided for review. !!!")
            return
        
        print("\n====== IMPACT LINK RELATIONSHIPS ======")
        print(f"Total Relationships Captured: {len(self.impact_df)}")
        # Shows how events map to indicators (Task 1.3 requirement)
        display(self.impact_df.groupby(['related_indicator', 'impact_direction']).size().unstack(fill_value=0))

    def run_all(self):
        if self.df.empty:
            print("!!! ERROR: DataFrame is empty. !!!")
            return
        self.schema_overview()
        self.temporal_analysis()
        self.indicator_coverage()
        self.missing_value_summary()
        self.impact_link_review()
