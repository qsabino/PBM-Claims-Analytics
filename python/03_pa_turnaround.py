# GOAL: 
# PA turnaround distribution (mean vs. median), 
# urgent vs. standard requests, 
# SLA breach detection — including why raw breach rates must be compared within SLA tier, not across the whole book
#===========================================================================



# %%
# Global setting
import pandas as pd
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

#===========================================================================



# %%
# Load and merge data
clients = pd.read_csv("../data/raw/clients.csv")
members = pd.read_csv("../data/raw/members.csv")
claims = pd.read_csv("../data/raw/claims.csv", parse_dates=["fill_date"])
pa = pd.read_csv("../data/raw/pa_requests.csv", parse_dates=["submitted_ts", "decision_ts"])

pa["turnaround_hours"] = (pa["decision_ts"] - pa["submitted_ts"]).dt.total_seconds() / 3600

merged = (
    pa
    .merge(claims[["claim_id", "member_id"]], on="claim_id")
    .merge(members[["member_id", "client_id"]], on="member_id")
    .merge(clients[["client_id", "client_name", "pa_sla_hours", "is_problem_account"]], on="client_id")
)
print(f"{len(merged):,} PA requests merged with client context")

#===========================================================================



# %%
# Turnaround distribution: mean vs. median
print(f"{merged["turnaround_hours"].describe()[["min", "mean", "50%", "max"]]}")
print(f"90th percentile: {merged["turnaround_hours"].quantile(0.9):.1f} hours")

# Slowest 5% turnaround time
slowest_5pct_pa_turnaround = merged[
    merged["turnaround_hours"] > merged["turnaround_hours"].quantile(0.95)
]

#===========================================================================



# %%
# Turnaround distribution, mean and median
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(merged["turnaround_hours"], bins=60, color="steelblue", edgecolor="white")
ax.axvline(merged["turnaround_hours"].mean(), color="firebrick", linestyle="--", label="mean")
ax.axvline(merged["turnaround_hours"].median(), color="darkgreen", linestyle="--", label="median")
ax.set_xlabel("Turnaround hours")
ax.set_ylabel("Number of PA requests")
ax.set_title("PA turnaround time distribution")
ax.legend()
plt.tight_layout()
plt.show()

#===========================================================================



# %%
# Urgent vs. standard requests: mean, median, count
urgent_comparison = merged.groupby("urgent_flag")["turnaround_hours"].agg(["mean", "median", "count"])
urgent_comparison.index = urgent_comparison.index.map({True:"Urgent", False:"Standard"})
urgent_comparison

#===========================================================================



# %%
# RAW SLA-breach-rate: overall and by_client
merged["breach"] = merged["turnaround_hours"] > merged["pa_sla_hours"]
overall_breach_rate = merged["breach"].mean()
print(f"Overall SLA breach rate: {overall_breach_rate:.1%}")

by_client_raw = merged.groupby(["client_id", "client_name", "pa_sla_hours"]).agg(
    n=("pa_id", "count"), 
    breach_rate=("breach", "mean")
).reset_index().sort_values("breach_rate", ascending=False)

by_client_raw.head(10)
# Opserved: Comparing raw breach rates across clients with different SLA is not a fair comparison

#===========================================================================



# %%
# Breach rate by sla tier
breach_by_sla_tier = merged.groupby("pa_sla_hours")["breach"].mean()
print("Breach rate by SLA tier (pooled across all clients on that tier):")
breach_by_sla_tier

#===========================================================================



# %%
# CORRECT SLA-breach-rate: within each SLA tier, calculate breach_rate and z_score of each client

# set up the loop, one SLA tier at a time
results = []
for sla, group in merged.groupby("pa_sla_hours"):
    tier_rate = group["breach"].mean()
    # collapse each tier down to one row per client
    by_client = group.groupby(["client_id", "client_name"]).agg(
        n=("pa_id", "count"), 
        breach_rate=("breach", "mean")
    ).reset_index()
    # compute the z-score for each client
    by_client["z_score"] = (
        (by_client["breach_rate"] - tier_rate)
        / ((tier_rate * (1 - tier_rate)) / by_client["n"]) ** 0.5
    )
    # tag which tier this batch came from, and stash it
    by_client["sla_tier"] = sla
    results.append(by_client)

# stack all three tiers into one table
by_client_corrected = pd.concat(results, ignore_index=True)

# Clients with breach_rate z-score > 2
flagged = by_client_corrected[by_client_corrected["z_score"] > 2].sort_values("z_score", ascending=False)

# filter to the outliers
print(f"Clients flagged within SLA tier (z > 2): {len(flagged)}")
flagged[["client_id", "client_name", "sla_tier", "n", "breach_rate", "z_score"]]

#===========================================================================



# %%
# By client, denial rate vs. breach rate: are they related?
denial_by_client = claims.merge(members[["member_id", "client_id"]], on="member_id").groupby("client_id")["denied"].mean()
breach_by_client = merged.groupby("client_id")["breach"].mean()

# glues the two Series together side by side as columns, aligns them by their shared index client_id
comparison = pd.concat(
    [denial_by_client, breach_by_client], 
    axis=1, 
    keys=["denial_rate", "breach_rate"]
).dropna()

# computes the pairwise correlation coefficient between every pair of numeric columns in the DataFrame.
comparison.corr()

#===========================================================================



# %%
# Breach rate trend over time
merged["month"] = merged["submitted_ts"].dt.to_period("M")
monthly_breach = merged.groupby("month")["breach"].mean()
monthly_breach

#===========================================================================



# %%
# Plot breach rate trend
fig, ax = plt.subplots(figsize=(8, 4))
monthly_breach.plot(ax=ax, marker="o", color="steelblue")
ax.axhline(overall_breach_rate, color="gray", linestyle="--", label="overall average")
ax.set_xlabel("Month")
ax.set_ylabel("SLA breach rate")
ax.set_title("PA SLA breach rate by month")
ax.legend()
plt.tight_layout()
plt.show()

#===========================================================================



# %%
# Summary
summary = pd.DataFrame({
    "metric": [
        "Mean turnaround (hrs)",
        "Median turnaround (hrs)",
        "Urgent median (hrs)",
        "Standard median (hrs)",
        "Overall SLA breach rate",
        "Clients flagged (raw, pooled)",
        "Clients flagged (corrected, within SLA tier)",
        "Denial rate vs. breach rate correlation",
    ],
    "value": [
        f"{merged['turnaround_hours'].mean():.1f}",
        f"{merged['turnaround_hours'].median():.1f}",
        f"{urgent_comparison.loc['Urgent', 'median']:.1f}",
        f"{urgent_comparison.loc['Standard', 'median']:.1f}",
        f"{overall_breach_rate:.1%}",
        f"{(by_client_raw['breach_rate'] > by_client_raw['breach_rate'].mean() + 2*by_client_raw['breach_rate'].std()).sum()}",
        f"{len(flagged)}",
        f"{comparison.corr().iloc[0,1]:.3f}",
    ],
})
summary
# %%