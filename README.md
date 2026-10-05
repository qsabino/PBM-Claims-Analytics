# PBM Claims Analytics

Analysis of simulated Pharmacy Benefit Manager (PBM) claims data, 75,000 claims across 50 clients. Run through four tools end to end: **Python, PostgreSQL, Excel, Power BI**. One dataset, the same business questions, answered four ways, to compare how each tool actually handles the work.

**Added machine learning layer in scikit-learn.** A classification model to predict claim denial risk and a regression model to estimate claim amounts. Both models evaluated with standard metrics.

## Dataset

A synthetic PBM claims warehouse: 5 tables (clients, drugs, members, claims, pa_requests), ~75,000 claims, ~14,000 PA requests, 12 months.

## Key findings

- **A small group of clients drives a disproportionate share of operational risk.** 6 of 50 clients deny claims at 14.6%, more than double the 6.4% rate for the rest of the book, and breach their contracted PA turnaround SLA 73.8% of the time vs. 30.9% elsewhere.
- **Raw SLA breach rates are misleading without controlling for contract terms.** Clients on a 24-hour SLA breach ~9x more often than clients on a 72-hour SLA. A statistical test run within each SLA tier, not across the whole book, is what separates real outliers from 5 confirmed accounts.
- **Specialty (tier 4) drugs drive 65.2% of total plan-paid cost** despite being a small share of claim volume.
- **An estimated $907K in rebate dollars (12.3%) was left on the table** against contract-implied targets, concentrated in specialty and non-formulary categories.
- **Unsupervised clustering (KMeans) independently rediscovered 4 of 5 known at-risk clients** using only denial rate, SLA breach rate, rebate capture, and PMPM (no label given to the model) and flagged one additional client worth a manual look.
- **A predictive model for claim denial converges to ROC-AUC ~0.70 across three different algorithms** (logistic regression, random forest, XGBoost)
- **A volume-driven noise pattern shows up in every analysis**: small clients produce unreliable rates, trends, and slopes purely from low claim counts. Every analysis below checks for this (via z-tests, volume floors, or variance checks) before drawing a conclusion.

## Python

| Notebook | What it answers |
|---|---|
| [`01_data_validation.py`](python/01_data_validation.py) | Load all 5 tables, confirm row counts, nulls, and key denial/turnaround figures. |
| [`02_denial_analysis.py`](python/02_denial_analysis.py) | Denial rate by client, tier, and reason; a proportion z-test separates real outlier clients from small-sample noise; denial reasons ranked by both volume and dollar exposure. |
| [`03_pa_turnaround.py`](python/03_pa_turnaround.py) | PA turnaround distribution (mean vs. median), urgent vs. standard requests, and SLA breach detection — including why raw breach rates must be compared within SLA tier, not across the whole book. |
| [`04_rebate_analysis.py`](python/04_rebate_analysis.py) | Rebate capture rate (actual vs. contract-target) by tier and by client, the total dollar gap, and why capture-rate variance shrinks with claim volume. |
| [`05_cost_trend_pmpm.py`](python/05_cost_trend_pmpm.py) | Per-member-per-month (PMPM) cost trend, cost breakdown by drug tier, client-level trend-slope fitting, and a drill-down into what's actually driving the fastest-rising clients' costs. |
| [`06_client_segmentation.py`](python/06_client_segmentation.py) | KMeans clustering on four client-level metrics to segment "healthy" vs. "at-risk" accounts, validated against the known outlier clients. |

Each notebook is self-contained — running `01` isn't required before `02`

## Data Science

| Notebook | Result |
|---|---|
| [`01_denial_classification.py`](data-science/01_denial_classification.py) | Logistic Regression, Random Forest, and XGBoost all converge to ROC-AUC ~0.70. A feature set designed to prevent data leakage, a majority-class baseline, and 5-fold CV included |
| [`02_claim_cost_regression.py`](data-science/02_claim_cost_regression.py) | R² = 0.996 after replacing broad drug tiers with actual drug prices, showing that the relationship is better captured when using continuous price data. |

## PostgreSQL

- Creates the matching Postgres schema with load instructions.
- 4 query files (denial_rate.sql, pa_sla_breach.sql, rebate_capture.sql, pmpm_trend.sql) reproducing the same findings as the Python notebooks.
- client_summary_view.sql is a single reusable view rolling up denial, SLA, rebate, and PMPM metrics per client.

## Excel

pbm_claims_dashboard.xlsx covers Power Query import, Power Pivot Relationships, DAX measures and interactive dashboard

## Power BI

pbm_claims_model.pbix covers get data, model relationships, the same DAX measures as the Excel guide (identical formulas, same underlying engine), native report visuals theming.

## Tech stack

- Python: panda, numpy
- Machine learning: scikit-learn, XGBoost
- Database: PostgreSQL
- Excel: Power Query, Power Pivot, DAX, interactive report visuals
- Power BI Desktop: Power Query, Data Model, DAX, native report visuals

## Why this project

Every analysis here checks whether a finding is statistically real and whether it would generalize before acting on it: z-tests before calling a client an outlier, volume floors before trusting a rate or a trend, leakage checks before trusting a model.


**Visualizing client segmentation.**
<Figure size 700x600 with 1 Axes><img width="690" height="590" alt="image" src="https://github.com/user-attachments/assets/862d3347-ae19-4595-9fe3-f7e4d9cc3ffd" />

**scikit-learn layer added**
<img width="1090" height="440" alt="image" src="https://github.com/user-attachments/assets/68759a8d-d00d-4ffe-95e0-354b71ef0a87" />

**Excel reference**
<img width="1920" height="1032" alt="image" src="https://github.com/user-attachments/assets/24738510-4bb5-48d6-979c-cca5d6163d09" />

**Power BI reference**
<img width="1369" height="748" alt="image" src="https://github.com/user-attachments/assets/b951dfa3-0ffe-44a3-b234-532b98c04ca9" />
