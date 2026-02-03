import pandas as pd
import numpy as np
from IPython.display import display

# Inside the class ImpactModel:

class ImpactModel:
    """
    Event Impact Modeling Engine (Full Task 3)

    Features:
    -------------------
    ✅ Event–Indicator Matrix
    ✅ Lag + gradual adoption dynamics
    ✅ Evidence weighting (comparable country)
    ✅ Multi-event additive combination
    ✅ Forecast simulation
    ✅ Historical validation
    ✅ Confidence + methodology export
    """

    def __init__(self, data_df, impact_links_df):
        self.data = data_df.copy()
        self.links = impact_links_df.copy()

        self._align_schema()

        self.matrix = None
        self.full_map = None

    # ------------------------------------------------------------------
    # 1. Schema Alignment
    # ------------------------------------------------------------------
    def _align_schema(self):
        """Standardizes ID columns to prevent KeyErrors."""
        id_aliases = ['id', 'record_id', 'event_id', 'id_code']
        for col in self.data.columns:
            if col.lower() in id_aliases:
                self.data.rename(columns={col: 'id'}, inplace=True)

        link_aliases = ['parent_id', 'event_id', 'source_id', 'parent_record']
        for col in self.links.columns:
            if col.lower() in link_aliases:
                self.links.rename(columns={col: 'parent_id'}, inplace=True)

    # ------------------------------------------------------------------
    # 2. Event–Indicator Matrix Builder
    # ------------------------------------------------------------------
    def generate_matrix(self):
        """
        Builds Event–Indicator Association Matrix using 'impact_estimate' as main value.
        """
        # Extract event rows
        events = self.data[self.data['record_type'].str.lower() == 'event'].copy()
        events['event_date'] = pd.to_datetime(events['observation_date'])

        # Merge links with event metadata
        full_map = self.links.merge(
            events[['id', 'category', 'indicator', 'event_date']],
            left_on='parent_id',
            right_on='id',
            how="left"
        )

        # Evidence weighting (empirical/literature/theoretical)
        evidence_weights = {"empirical": 1.0, "literature": 0.7, "theoretical": 0.4}
        full_map['evidence_weight'] = full_map['evidence_basis'].map(evidence_weights).fillna(0.5)

        # Magnitude fallback mapping
        mag_map = {"high": 0.8, "medium": 0.5, "low": 0.2}

        def calculate_weight(row):
            estimate = row.get("impact_estimate", 0)
            if pd.isna(estimate) or estimate == 0:
                val = mag_map.get(str(row.get("impact_magnitude")).lower(), 0.1)
            else:
                val = float(estimate)
            direction = str(row.get("impact_direction", "increase")).lower()
            multiplier = -1 if "dec" in direction or "neg" in direction else 1
            return val * multiplier * row["evidence_weight"]

        full_map["weight"] = full_map.apply(calculate_weight, axis=1)

        # Confidence labeling
        def confidence_label(evidence):
            if evidence == "empirical": return "High Confidence"
            if evidence == "literature": return "Medium Confidence"
            return "Low Confidence"

        full_map['confidence_level'] = full_map['evidence_basis'].apply(confidence_label)

        self.full_map = full_map

        # Pivot → Event–Indicator Matrix
        self.matrix = full_map.pivot_table(
            index='parent_id',
            columns='related_indicator',
            values='weight',
            aggfunc='sum'
        ).fillna(0.0)

        return self.matrix

    # ------------------------------------------------------------------
    # 3. Lag + Gradual Adoption Dynamics
    # ------------------------------------------------------------------
    def event_effect(self, t_months, impact, lag, k=0.3):
        """
        Gradual adoption curve:
        Effect(t) = impact * (1 - exp(-k * (t - lag)))
        """
        if t_months < lag:
            return 0
        return impact * (1 - np.exp(-k * (t_months - lag)))


    # ------------------------------------------------------------------
    # 4. Forecast Indicator Trajectory
    # ------------------------------------------------------------------
    def simulate_indicator(self, indicator_code, start="2021-01-01", end="2027-12-31", show=True):
        """
        Returns forecast DataFrame for a given indicator and optionally displays it.
        """
        if self.full_map is None:
            raise ValueError("Run generate_matrix() first.")

        timeline = pd.date_range(start, end, freq="ME")
        forecast = pd.DataFrame({"date": timeline})

        # Baseline observations
        obs = self.data[
            (self.data['record_type'] == 'observation') &
            (self.data['indicator_code'] == indicator_code)
        ].sort_values("observation_date")

        if obs.empty:
            raise ValueError(f"No baseline found for {indicator_code}")

        baseline_value = obs['value_numeric'].iloc[0]
        forecast['baseline'] = baseline_value

        # Relevant impacts
        relevant = self.full_map[self.full_map['related_indicator'] == indicator_code]

        # Apply lagged impacts
        for _, row in relevant.iterrows():
            event_date = row['event_date']
            impact     = row['weight']
            lag        = row.get('lag_months', 0)

            forecast[row['parent_id']] = forecast['date'].apply(
                lambda d: self.event_effect((d - event_date).days / 30, impact, lag)
            )

        # Combine multi-event effects
        event_cols = list(relevant['parent_id'])
        forecast['total_event_effect'] = forecast[event_cols].sum(axis=1)
        forecast['predicted'] = forecast['baseline'] + forecast['total_event_effect']

        if show:
            display(forecast)

        return forecast

    # ------------------------------------------------------------------
    # 5. Historical Validation
    # ------------------------------------------------------------------
    def validate_event(self, indicator_code, year_start, year_end, show=True):
        """
        Returns a validation summary dict and optionally displays as a table.
        """
        obs = self.data[
            (self.data['record_type'] == 'observation') &
            (self.data['indicator_code'] == indicator_code)
        ]
        obs['year'] = pd.to_datetime(obs['observation_date']).dt.year

        v_start = obs[obs['year'] == year_start]['value_numeric'].mean()
        v_end   = obs[obs['year'] == year_end]['value_numeric'].mean()
        observed_change = v_end - v_start

        forecast = self.simulate_indicator(indicator_code, show=False)
        pred_end = forecast[forecast['date'].dt.year == year_end]['predicted'].iloc[-1]
        predicted_change = pred_end - v_start

        summary = pd.DataFrame({
            "Indicator": [indicator_code],
            "Observed Change": [observed_change],
            "Predicted Change": [predicted_change],
            "Difference": [predicted_change - observed_change]
        })

        if show:
            display(summary)

        return summary
   # ------------------------------------------------------------------
    # 6. Methodology Export
    # ------------------------------------------------------------------
    def methodology_text(self):
        return """
# Event Impact Modeling Methodology

## Overview
Each event (policy, product launch, infrastructure reform) is modeled as a shock
that affects one or more financial inclusion indicators.

## Impact Links
Each impact link provides:
- Indicator affected
- Direction (increase/decrease)
- Magnitude estimate
- Lag in months
- Evidence basis (empirical/literature/theoretical)

## Time Dynamics
Impacts do not occur instantly.  
We apply a gradual adoption curve:

Effect(t) = Impact × (1 − exp(−k(t − lag)))

## Combining Multiple Events
Indicator trajectories are additive:

Predicted(t) = Baseline(t) + Σ EventEffects(t)

## Comparable Country Evidence
When Ethiopia-specific pre/post data is limited:
- Literature evidence from Kenya, India, Tanzania is used
- These impacts are down-weighted relative to empirical estimates

## Validation
Telebirr launch was validated against observed mobile money adoption:
4.7% (2021) → 9.45% (2024)

## Limitations
- Assumes additive impacts (ignores interaction effects)
- Lag assumptions are approximate
- Sparse time-series data limits calibration
"""