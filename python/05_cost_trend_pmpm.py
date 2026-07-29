# GOAL: 
# Per-member-per-month (PMPM) cost trend, 
# cost breakdown by drug tier, 
# client-level trend-slope fitting, 
# a drill-down into what's actually driving the fastest-rising clients' costs.
#===========================================================================



# %%
# Global setup
import pandas as pd
import numpy as np
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

#===========================================================================



# %%
# Load and merge data
clients = pd.read_csv("../data/raw/clients.csv")
drugs = pd.read_csv("../data/raw/drugs.csv")
members = pd.read_csv("../data/raw/members.csv")
claims = pd.read_csv("../data/raw/claims.csv", parse_dates=["fill_date"])

merged = (
    claims
    .merge(members[["member_id", "client_id"]], on="member_id")
    .merge(clients[["client_id", "client_name"]], on="client_id")
    .merge(drugs[["ndc_code", "drug_tier"]], on="ndc_code")
)
print(f"{len(merged):,} claims merged with client and drug tier context")

#===========================================================================



# %%
# Counting how many members each client has?
member_counts = members.groupby("client_id").size().rename("n_members")
member_counts.head()

#===========================================================================



# %%
# Overall PMPM trend
total_members = members.shape[0]

merged["month"] = merged["fill_date"].dt.to_period("M")
overall_pmpm = merged.groupby("month")["plan_paid"].sum()/total_members
overall_pmpm

#===========================================================================



# %%
# Plot overall-pmpm
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize= (8, 4))
overall_pmpm.plot(ax=ax, marker="o", color="steelblue")
ax.set_xlabel("Month")
ax.set_ylabel("PMPM ($)")
ax.set_title("Overall book-of-business PMPM trend")
plt.tight_layout()
plt.show()
# Observed: PMPM bounces around a stable range, no real upward or downward drift

#===========================================================================



# %%
# Cost breakdown by drug tier over time
tier_month = merged.groupby(["month", "drug_tier"])["plan_paid"].sum().unstack()
tier_month

#===========================================================================



# %%
# Plot cost breakdown by drug tier over time
import matplotlib.pylab as plt

fig, ax = plt.subplots(figsize=(8, 4))
tier_month.plot(ax=ax, marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Toatl plan-paid ($)")
ax.set_title("Plan-paid cost break down by drug tier over time")
ax.legend(title="Drug tier")
plt.tight_layout()
plt.show()
# Observed: Tier 4 (specialty) dominates total cost. 
# Specialty drugs are the primary cost driver despite low utilization. 
# Any cost-reduction conversation with a client should start here

#===========================================================================



# %%
# Client-level PMPM trend
client_monthly = merged.groupby(["client_id", "month"])["plan_paid"].sum().reset_index()
client_monthly = client_monthly.merge(member_counts, on="client_id")
client_monthly["pmpm"] = client_monthly["plan_paid"] / client_monthly["n_members"]

# Function fit a simple linear trend (slope) to each client's monthly PMPM
def fit_slope(group):
    x = np.arange(len(group))
    y = group["pmpm"].values
    if len(x) < 3:
        return np.nan
    return np.polyfit(x, y, 1)[0]  # slope = $ change in PMPM per month

# Applied function to each client
slopes = (
    client_monthly.groupby("client_id")
    .apply(fit_slope, include_groups=False)
    .rename("pmpm_slope")
    .reset_index()
)

# Bringing in context for readability, does not effect the analysis
slopes = (
    slopes
    .merge(member_counts, on="client_id")
    .merge(clients[["client_id", "client_name"]], on="client_id")
)

# Sorts so the steepest upward slopes land on top, the clients whose PMPM is rising fastest
slopes.sort_values("pmpm_slope", ascending=False).head(10)

#===========================================================================



# %%
# Plot n_members vs. abs(pmpm_slope) of each client
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(slopes["n_members"], slopes["pmpm_slope"].abs(), alpha=0.6, color="steelblue")
ax.set_xscale("log")
ax.set_xlabel("Number of members (log scale)")
ax.set_ylabel("|PMPM slope| ($/month)")
ax.set_title("Trend steepness vs. client size")
plt.tight_layout()
plt.show()

#===========================================================================



# %%
# Correlation between client size and |slope|
corr = slopes["n_members"].corr(slopes["pmpm_slope"].abs())
print(f"Correlation between client size and absolute value of pmpm_slope:{corr:.3f}")
# Observed: A negative correlation means: as one variable increases, the other tends to decrease
# Larger clients (more members) tend to have smaller PMPM slope magnitude — meaning their per-member-per-month costs are more stable, trending less sharply up or down.
# Smaller clients (fewer members) tend to have larger PMPM slope magnitude — meaning their costs are more volatile, swinging up or down more sharply.
# |corr| = 0.292 show a weak correlation 
# --> variation in PMPM slope volatility (cost trend stability) is not explained by client size alone,
# other factors (drug mix, specific high-cost claims, plan design, client-specific formulary rules, etc.) are likely playing a much bigger role.

#===========================================================================



# %%
# To not mistaking noise for a trend, we filter to clients with enough members
# --> so that a trend fit actually means something.
MIN_MEMBERS = 100  # arbitrary value

reliable_trends = slopes[slopes["n_members"] >= MIN_MEMBERS]
reliable_trends = reliable_trends.sort_values("pmpm_slope", ascending=False)

print(f"{len(reliable_trends)} of {len(slopes)} clients have enough members to trust a trend fit")
reliable_trends.head(10)

#===========================================================================



# %%
# Drill-down: what's actually driving the top reliable clients?
# A rising PMPM slope tells you that cost is climbing, not why. 
# Before recommending anything to these clients, pull their own cost-by-tier breakdown over time
# The same view we built book-wide, but filtered down to just each client.
top_clients = reliable_trends.head(3)
top_clients

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

for ax, (_, row) in zip(axes, top_clients.iterrows()):
    cid, name, n_mem = row["client_id"], row["client_name"], row["n_members"]

    client_claims = merged[merged["client_id"] == cid]
    client_tier_month = (
        client_claims.groupby(["month", "drug_tier"])["plan_paid"].sum().unstack(fill_value=0)
    )
    client_tier_pmpm = client_tier_month / n_mem

    client_tier_pmpm.plot(ax=ax, marker="o", legend=False)
    ax.set_title(f"{name}\n({n_mem} members)")
    ax.set_xlabel("Month")
    ax.set_ylabel("PMPM ($)")

axes[0].legend(title="Drug tier", loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()
# Observed: Tier 4 (specialty) is the volatile line in all three charts 
# For these clients, the accelerating cost story concentrated in specialty utilization
# Drill-down to member-level detail on which individuals are driving the tier 4 spend or why

#===========================================================================



# %%
# Summary
summary = pd.DataFrame({
    "metric": [
        "Overall PMPM range",
        "Tier 4 share of total plan-paid cost",
        "Correlation: client size vs. |trend slope|",
        "Clients with reliable trend fit (n_members >= 100)",
        "Steepest reliable upward trend",
    ],
    "value": [
        f"${overall_pmpm.min():.2f} - ${overall_pmpm.max():.2f}",
        f"{tier_month[4].sum() / tier_month.sum().sum():.1%}",
        f"{slopes['n_members'].corr(slopes['pmpm_slope'].abs()):.3f}",
        f"{len(reliable_trends)} of {len(slopes)}",
        f"{reliable_trends.iloc[0]['client_name']} (${reliable_trends.iloc[0]['pmpm_slope']:.2f}/mo)"
        if len(reliable_trends) > 0 else "n/a",
    ],
})
summary
# %%