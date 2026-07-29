# GOAL: 
# Denial rate by client, tier, and reason; 
# a proportion z-test separates real outlier clients from small-sample noise; 
# denial reasons ranked by both volume and dollar exposure.
#===========================================================================



# %%
# Global setting
import pandas as pd
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

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
# Denial rate by client
by_client = merged.groupby(["client_id", "client_name"]).agg(
    n_claims = ("claim_id", "count"),
    denial_rate = ("denied", "mean"),
).reset_index().sort_values("denial_rate", ascending=False)
by_client.head(15)

#===========================================================================



# %%
# Statistical significance check denial rate by client
overall_rate = merged["denied"].mean()
print(f"Book-of-business denial rate: {overall_rate:.2%}")

by_client["z_score"] = (
    (by_client["denial_rate"] - overall_rate)
    / ((overall_rate * (1 - overall_rate)) / by_client["n_claims"]) ** 0.5
)

flagged = by_client[by_client["z_score"] > 2].sort_values("z_score", ascending=False)
print(f"\nClients flagged as statistically elevated (z > 2): {len(flagged)}")
flagged[["client_id", "client_name", "n_claims", "denial_rate", "z_score"]]

#===========================================================================



# %%
# Denial reason breakdown (denial_reason, count, pct_of_denial)
denied = merged[merged["denied"]]

reason_volume = denied["denial_reason"].value_counts() # returns a series where the index is the reason and the values are the counts.
reason_pct = (denied["denial_reason"].value_counts(normalize=True) * 100).round(1) # change to proportions then pct

reason_summary = pd.DataFrame({
    "count": reason_volume,
    "pct_of_denials": reason_pct,
}) # it works because they share the same index (the denial reason strings)
reason_summary

#===========================================================================



# %%
# Denial reason breakdown (denial_reason, count, pct_of_denial, awp_exposure)
reason_dollars = (
    denied.groupby("denial_reason")["awp"]
    .sum()
    .sort_values(ascending=False)
    .round(0)
)
reason_summary["awp_exposure"] = reason_dollars # value match by index_label
reason_summary.sort_values("awp_exposure", ascending=False)

#===========================================================================



# %%
# Denial rate by drug tier
by_tier = merged.groupby("drug_tier")["denied"].agg(["mean", "count"])
by_tier.columns = ["denial_rate", "n_claims"]
by_tier

#===========================================================================



# %%
# Denial rate trend over time
merged["month"] = merged["fill_date"].dt.to_period("M")
monthly_trend = merged.groupby("month")["denied"].mean()
monthly_trend

#===========================================================================



# %%
# Summary
summary = {
    "metric": [
        "Overall denial rate",
        "Clients flagged (z > 2)",
        "Top denial reason (by count)",
        "Top denial reason (by AWP exposure)",
        "Highest-denying tier",
    ],
    "value": [
        f"{overall_rate:.1%}",
        len(flagged),
        reason_summary['count'].idxmax(),
        reason_summary['awp_exposure'].idxmax(),
        f"Tier {by_tier["denial_rate"].idxmax()} ({by_tier["denial_rate"].max()})",
    ],
}

summary = pd.DataFrame(summary)
summary
# %%