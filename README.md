# PBM Claims Analytics

Analysis of Pharmacy Benefit Manager (PBM) claims data (75,000 claims across 50 clients). 
Mirrors a real PBM Client Financial & Account Operations (CFAO) team run day to day: denial trends, SLA compliance, rebate performance, cost trend, and client risk segmentation.

**This is v1: the Python analysis.** The same dataset and business questions will be re-run in PostgreSQL, Excel/Power Pivot, and Power BI in later versions, so the same findings can be compared across tools.

## Key findings

- **A small group of clients drives a disproportionate share of operational risk.**
- **Without controlling for contract terms, raw SLA breach rates are misleading.** 
- **Specialty (tier 4) drugs drive 65% of total plan-paid cost** despite being a small share of claim volume. The first place to focus any cost containment conversation.
- **An estimated $907K in rebate dollars (12.3%) was left on the table** against contract-implied targets, concentrated in specialty and non-formulary categories.
- **Unsupervised clustering (KMeans) independently rediscovered 4 of 4 known at-risk clients** using only denial rate, SLA breach rate, rebate capture, and PMPM (no label given to the model) and flagged one additional client worth a manual look.
- **A volume-driven noise pattern shows up in every analysis**: Every notebook below separates real signal from this kind of noise (via z-tests, volume floors, or variance checks) before drawing conclusions.

## Notebooks

| Notebook | What it answers |
|---|---|
| [`01_data_validation.py`](python/01_data_validation.ipynb) | Load all 5 tables, confirm row counts, nulls, and key denial/turnaround figures. |
| [`02_denial_analysis.py`](python/02_denial_analysis.ipynb) | Denial rate by client, tier, and reason; a proportion z-test separates real outlier clients from small-sample noise; denial reasons ranked by both volume and dollar exposure. |
| [`03_pa_turnaround.py`](python/03_pa_turnaround.ipynb) | PA turnaround distribution (mean vs. median), urgent vs. standard requests, and SLA breach detection — including why raw breach rates must be compared within SLA tier, not across the whole book. |
| [`04_rebate_analysis.py`](python/04_rebate_analysis.ipynb) | Rebate capture rate (actual vs. contract-target) by tier and by client, the total dollar gap, and why capture-rate variance shrinks with claim volume. |
| [`05_cost_trend_pmpm.py`](python/05_cost_trend_pmpm.ipynb) | Per-member-per-month (PMPM) cost trend, cost breakdown by drug tier, client-level trend-slope fitting, and a drill-down into what's actually driving the fastest-rising clients' costs. |
| [`06_client_segmentation.py`](python/06_client_segmentation.ipynb) | KMeans clustering on four client-level metrics to segment "healthy" vs. "at-risk" accounts, validated against the known outlier clients. |

Each notebook is self-contained — running `01` isn't required before `02`

## Tech stack

Python, pandas, numpy, matplotlib, scikit-learn (StandardScaler, KMeans, PCA, silhouette_score), VS Code.

## Roadmap

- [x] Python analysis (this version)
- [ ] PostgreSQL — same five business questions, written as SQL (window functions, CTEs), against a real relational schema
- [ ] Excel / Power Pivot — DAX measures for the same KPIs
- [ ] Power BI — client-facing dashboard built on the same model
- [ ] `tool_comparison.md` — notes on where each tool was the easier or harder way to answer the same question
