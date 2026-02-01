# ethiopia-fi-forecast
Ethiopia Financial Inclusion Forecasting (Selam Analytics)
📌 Project Overview
This project provides a data-driven framework to analyze and forecast financial inclusion trajectories in Ethiopia. Using a unified dataset of World Bank Findex surveys, National Bank of Ethiopia (NBE) reports, and private sector milestones (Telebirr, Safaricom/M-Pesa), this tool identifies why digital payment usage is exploding while account ownership remains plateaued.

The "Inclusion Gap" Challenge: Despite reaching over 139 million mobile money accounts in 2025, account ownership rates have only grown moderately since 2021. This repository models the impact of key policy enablers—like the Fayda National ID and FX Liberalization—to predict if Ethiopia will meet its 2027 inclusion targets.

🏗 Project Structure
The repository is organized following data science best practices:

Plaintext
├── data/
│   ├── raw/                # Immutable original datasets (Findex, NBE, IMF)
│   ├── processed/          # Unified schema after task-1 cleaning
│   └── enrichment_log.md   # Documentation of new proxy indicators added
├── notebooks/
│   ├── 01_data_cleaning.ipynb    # Task 1: Unification & Enrichment
│   └── 02_eda_analysis.ipynb     # Task 2: Trend & Gap Analysis
├── src/
│   ├── eda_engine.py       # Custom Python class for automated EDA
│   └── forecasting.py      # Time-series models (SARIMA/Prophet)
├── reports/
│   └── insights_summary.pdf# Final analysis of the "Access Stagnation"
├── requirements.txt        # Python dependencies
└── README.md               # You are here
🚀 Getting Started
1. Installation
Clone the repository and install dependencies:

Bash
git clone https://github.com/birhanu-ma/ethiopia-fi-forecast.git
cd ethiopia-fi-forecasting
pip install -r requirements.txt
2. Running the Analysis
To view the Exploratory Data Analysis (Task 2):

Navigate to notebooks/02_eda_analysis.ipynb.

Use the EdaAnalysis class from src/ to generate visualizations.

📊 Key Analytical Features
Access vs. Usage Analysis: Visualizes the "decoupling" between new account registration and active digital transaction volume.

Event Overlay: Maps policy shifts (e.g., Mandatory Fuel Digitization) directly onto trend lines to quantify impact.

Gender & Infrastructure Gaps: Tracks the persistent 14% gender gap and identifies smartphone penetration (21.7%) as a primary barrier to advanced inclusion.

Forecasting: Predictive modeling of account ownership targets for 2027 based on the Digital Ethiopia 2030 roadmap.

🛠 Tech Stack
Language: Python 3.10+

Data Handling: Pandas, NumPy

Visualization: Plotly (Interactive), Seaborn, Matplotlib

Forecasting: Scikit-learn, Statsmodels

🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the project.

Create your Feature Branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the Branch (git push origin feature/AmazingFeature).

Open a Pull Request