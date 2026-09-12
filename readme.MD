# Store Sales – Time Series Forecasting (Corporación Favorita)

End-to-end time series forecasting pipeline evaluating 1,782 distinct retail series (54 stores across 33 product families) in Ecuador over a 15-day forecasting horizon. 

This repository documents the end-to-end workflow: exploratory data analysis, evaluation of individual classical econometric/statistical baselines (SARIMAX, Prophet, ARX-GARCH), global gradient-boosted decision trees (GBDTs), and an empirical analysis of why global machine learning models only marginally outperform local classical specifications under multi-step lag constraints.

---

## Repository Structure & Quickstart

```text
├── app.py                      # Streamlit interactive EDA dashboard
├── classical_dashboard.py      # Streamlit classical model diagnostics dashboard
├── trial.ipynb      # Jupyter notebook with exploration
├── classical_timeseries_evaluation_metrics.csv                       # Classical timeseries model results
├── ml_timeseries_evaluation_metrics.csv                       # ML model results
└── README.md
```

### Running the Dashboards

```bash
# 1. Exploratory Data Analysis Dashboard
streamlit run app.py

# 2. Classical Baselines & Diagnostics Dashboard
streamlit run classical_dashboard.py
```

---

## 1. Domain Nuances & EDA Findings

The dataset consists of 7 relational tables covering transactions, store metadata, promotional activity, oil prices, and local/national holidays from 2013 to mid-2017.

* **Inflation vs. Foot Traffic:** Average gross sales grew steadily from 2013 to 2017, whereas transaction counts (foot traffic) trended flat or slightly negative. This divergence indicates that revenue expansion was primarily driven by unit-price inflation rather than organic volume growth.
* **Shift to Discounting:** Starting in 2015, promotional frequency and volume surged. Approximately 20% of transactions involved a promotion, but accounted for ~65% of total sales volume (correlation of $r = 0.428$ between sales and items on promotion).
* **Autocorrelation & Seasonality:** Rigid weekly seasonality dominates (strong ACF/PACF peaks at lags 7, 14, 21, 28) with mid-week troughs (Wed/Thu) and weekend peaks (Sat/Sun). Secondary monthly spikes coincide with Ecuador’s bi-weekly public/private paydays (the 15th and the final day of the month).
* **Geographic Concentration:** Quito accounts for **52% of national sales**, with Type A stores and Cluster 5 generating the highest unit throughput.
* **Macroeconomic Shocks & Anomalies:**
  * **2014–2016 Oil Crash:** Ecuador's petroleum-dependent economy experienced significant volatility when oil prices collapsed from late 2014 through early 2016, driving retail sales volatility and subsequent inflation.
  * **April 16, 2016 Earthquake:** A 7.8 magnitude earthquake in Manabí caused an immediate ~2-month spike in essential goods stockpiling and relief donations, introducing a severe structural break.
* **Holiday Mechanics:** Standard holidays have less impact than expected due to Ecuadorian labour transfer laws. Mid-week holidays are frequently shifted to Mondays/Fridays, and subsequent Saturdays are designated as mandatory "Work Days" to compensate for bridge days, reversing typical weekend shopping profiles.

---

## 2. Classical Baselines Benchmark

Three distinct classical specifications were fit across all 1,782 series independently: **SARIMAX**, **Prophet**, and **ARX-GARCH**.

| Model | Win Rate (% of series) | Bias Direction | Mean Daily Bias | Median Volatility Ratio ($\frac{\text{Var}_{\text{pred}}}{\text{Var}_{\text{true}}}$) |
| :--- | :---: | :---: | :---: | :---: |
| **SARIMAX** | **40.1%** | Positive (+5.14 units) | Prone to positive spikes | **0.4228** |
| **Prophet** | **39.4%** | Symmetrical (-1.46 units) | Balanced (50.6% overforecast rate) | 0.2174 |
| **ARX-GARCH** | 20.5% | Severe Negative (-161.76 units) | Systematic underforecasting | 0.0461 |

### Diagnostic Takeaways

1. **Failure of Conditional Heteroskedasticity (GARCH):** ARX-GARCH collapsed on high-volume, intermittent series. Forcing mean-variance equations onto zero-inflated daily sales under strict stationarity constraints compressed point forecasts downward, capturing only ~5% of true variance.
2. **Variance Suppression:** All classical models systematically underestimate empirical retail variance ($\text{Ratio} < 1.0$). SARIMAX captured the highest variance (~42%) due to autoregressive terms, whereas Prophet yielded heavily smoothed trend-seasonal curves (~22%).
3. **Stationarity Sensitivity:** ARX-GARCH was strictly limited to stationary series (confirmed via ADF tests). Conversely, SARIMAX (via $d=1$ differencing) and Prophet (via piecewise linear trend changepoints) successfully adapted to non-stationary series undergoing structural shifts.
4. **Volume Tiers:** Continuous Gaussian assumptions broke down on low-volume, intermittent categories (e.g., `BOOKS`, `BABY CARE`). Fractional predictions on discrete counts inflated relative error metrics (RMSLE).

---

## 3. Global Machine Learning Architecture (GBDT)

To model cross-series relationships and non-linear interactions, global Gradient Boosted Decision Trees (HistGradientBoosting, XGBoost) were engineered.

### Feature Engineering
* **Autoregressive Lags:** Strictly constrained to $t \ge 16$ (`sales_lag_16` to `sales_lag_30`) to prevent data leakage over the 15-day forecasting horizon.
* **Rolling Statistics:** Multi-window statistics computed on valid historical lags (`sales_lag_16_roll_mean_7`, `roll_mean_30`, `roll_std_7`).
* **Calendar & Institutional Signals:** Day of week, day of month, month, weekend indicators, and a dedicated `is_pay_day` binary flag (15th and month-end).
* **Exogenous Variables:** Daily forward-filled oil price, rolling oil price deltas, promotional flags, and location-matched holiday indicators (filtered strictly to city/state/national scopes).
* **Objective Function:** Evaluated Tweedie loss ($\text{variance\_power} \in [1, 2]$) to handle zero-inflated, right-skewed intermittent categories without requiring arbitrary log-transform offsets.

---

## 4. Empirical Discussion: ML vs. Classical Baselines

While global GBDT models attained the lowest aggregate error, their margin of improvement over univariate classical baselines was marginal. The empirical drivers were:

### A. Local Specialization vs. Global Generalization
Classical models were parameterized independently across each of the 1,782 series, fitting 1,782 bespoke trend lines, seasonal Fourier terms, and AR/MA parameters. 
A single global tree ensemble must partition split criteria across all series simultaneously. While it captures shared non-linear promotional lifts well, it introduces inductive generalization bias that offsets its capacity advantage relative to perfectly localized fits.

### B. The Horizon Lag Penalty ($t-16$ Bottleneck)
* In a 15-day direct forecast, tree models cannot use lags $t-1$ through $t-15$ without rolling recursive inference (which compounds autoregressive errors). Features were therefore anchored at $t-16$.
* Classical models (particularly SARIMAX) utilize the immediate internal state up to $t-1$ to prime their forecast trajectories, effectively closing the predictive gap through recent autoregressive momentum.

### C. Ineffectiveness of Post-Earthquake Truncation
Discarding pre-June 2016 data to remove the April 2016 earthquake anomaly yielded no noticeable performance gain:
1. **Window Context Loss:** Tree models rely on long-term historical windows (`group_rolling_mean_90`) to determine base scale. Truncation destabilized feature initialization for early 2016 samples.
2. **Regime Loss:** The 2014–2015 oil crash, while noisy, provided the only training examples of how retail demand behaved under sharp macroeconomic contraction.
3. **Differencing Invariance:** Mathematical differencing ($d=1$) in SARIMAX and adaptive changepoints in Prophet already isolated local trajectory shifts from baseline level changes, making manual historical truncation redundant.

---

## 5. Architectural Next Steps: Two-Stage Hybrid

Given the complementary strengths of local autoregression and global exogenous non-linear modeling, the optimal target architecture is a **Residual Hybrid**:

$$\hat{y}_{i, t} = \hat{y}^{\text{Base}}_{i, t} + \hat{r}^{\text{GBDT}}_{i, t}$$

1. **Stage 1 (Local Linear / State-Space Base):** Run individual SARIMAX or Prophet models per series to generate base predictions $\hat{y}^{\text{Base}}_{i, t}$, capturing the local level, immediate autoregressive continuity, and primary weekly/annual seasonal patterns.
2. **Stage 2 (Global Residual Model):** Train a single global LightGBM/XGBoost model to predict the residuals $e_{i, t} = y_{i, t} - \hat{y}^{\text{Base}}_{i, t}$, conditioning solely on promotional intensity, macro shocks (oil prices), payday interactions, and store-family hierarchy metadata.
