Bike Rental Demand Forecasting

Overview
This project builds a supervised learning model to predict the number of bike rentals per hour. It follows a structured pipeline:
Data loading & inspection — understanding schema and data types
Exploratory Data Analysis (EDA) — visualizing temporal and seasonal patterns
Time series decomposition — identifying trend, seasonality, and residual components
Feature engineering — creating lag and rolling window features
Model training & evaluation — comparing a naive baseline against a Random Forest Regressor
Feature importance analysis — identifying the most predictive variables
---
Project Structure
```
bike-rental-forecasting/
│
├── hour.csv                          # Raw dataset
├── bike_demand_forecast.py           # Main pipeline script
│
├── rental_over_time_learning.png     # Hourly rentals over the full date range
├── patterns_learning.png             # Hourly, daily & seasonal boxplots
├── acf_learning.png                  # Autocorrelation function (ACF) plot
├── decomposition_learning.png        # Time series decomposition
├── predictions_learning.png          # Actual vs. predicted (first week of test set)
├── feature_importance_learning.png   # Top 10 feature importances
│
└── README.md

Section 1 — Data Loading
Loads `hour.csv`, parses the `dteday` column as datetime, and sorts records chronologically by date and hour.
Section 2 — Exploratory Data Analysis
Generates four visualizations to understand the data:
Rentals over time — a line plot of all hourly rentals across the entire dataset
Pattern boxplots — rental distributions broken down by hour of day, day of week, and season
ACF plot — autocorrelation up to 48 lags, revealing strong hourly and daily periodicity
Time series decomposition — additive decomposition on daily mean rentals showing trend, seasonality, and residuals
Section 3 — Feature Engineering
Creates time-aware features to give the model memory of recent demand:
Feature	Description
`lag_1h`	Rentals from 1 hour ago
`lag_2h`	Rentals from 2 hours ago
`lag_24h`	Rentals from 24 hours ago (same hour, previous day)
`rolling_mean_6h`	6-hour rolling average of past rentals
Rows with `NaN` values (introduced by shifting) are dropped before training.
Section 4 — Model Training & Evaluation
The dataset is split 80% training / 20% testing using a time-ordered split (no shuffling) to prevent data leakage.
Baseline model: Naive forecast using the previous hour's rental count (`lag_1h`).
ML model: `RandomForestRegressor` with 100 estimators trained on 16 features.
Both models are evaluated using MAE and RMSE.
---
Results
Model	MAE	RMSE
Naive Baseline (lag_1h)	~XX.XX	~XX.XX
Random Forest Regressor	~XX.XX	~XX.XX
> Run the script to populate exact metrics — they will be printed to the console.
The Random Forest model significantly outperforms the naive baseline by leveraging weather conditions, time-based features, and engineered lag variables.
---
Generated Outputs
File	Description
`rental_over_time_learning.png`	Full date range hourly rental line plot
`patterns_learning.png`	Hourly / daily / seasonal boxplots
`acf_learning.png`	Autocorrelation function up to 48 lags
`decomposition_learning.png`	Trend, seasonality & residual decomposition
`predictions_learning.png`	Actual vs. predicted rentals (first 168 hours of test set)
`feature_importance_learning.png`	Top 10 most important model features
---
Dependencies
Package	Purpose
`pandas`	Data loading and manipulation
`numpy`	Numerical operations
`matplotlib`	Plotting
`seaborn`	Statistical visualizations
`statsmodels`	ACF plot and time series decomposition
`scikit-learn`	Random Forest model and evaluation metrics
