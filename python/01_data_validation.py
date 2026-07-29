# GOAL: 
# Load 5 tables,
# confirm the data behaves the way it's supposed to
#===========================================================================



# %%
# Global formatting setting on how pandas DISPLAYS floating-point numbers
import pandas as pd
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
# pd.reset_option("display.float_format")

#===========================================================================



# %%
# Load 5 tables, check their shape()
import pandas as pd
clients = pd.read_csv("../data/raw/clients.csv")
drugs = pd.read_csv("../data/raw/drugs.csv")
members = pd.read_csv("../data/raw/members.csv")
claims = pd.read_csv("../data/raw/claims.csv", parse_dates=["fill_date"])
pa = pd.read_csv("../data/raw/pa_requests.csv", parse_dates=["submitted_ts", "decision_ts"])

for name, df in [("clients", clients), ("drugs", drugs), ("members", members),
                  ("claims", claims), ("pa_requests", pa)]:
    print(f"{name} {df.shape()}")

#===========================================================================    



# %%
# Count null per column, per table
print("Null per column, per table")
for name, df in [("clients", clients), ("drugs", drugs), ("members", members),
                  ("claims", claims), ("pa_requests", pa)]:
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty: # checks whether the series has zero elements left
        print(f"  {name:12s} no nulls")
    else:
        print(f"  {name:12s}")
        for col, n in nulls.items():
            print(f"      {col:15s} {n:,} nulls")

#===========================================================================



# %%
# orphan check: every member's client_id should exist in clients
orphan_members = ~members["client_id"].isin(clients["client_id"])
print(f"\nOrphaned members (members with no client_id): {orphan_members.sum()}")

# %%
# orphan check: every claim's member_id should exist in members
orphan_claims = ~claims["member_id"].isin(members["member_id"])
print(f"Orphaned claims (claims with no member_id): {orphan_claims.sum()}")

#===========================================================================



# %%
# Overall denial rate
overall_denial_rate = claims["denied"].mean()
print(f"Overall denial rate: {overall_denial_rate:.1%}")

#===========================================================================



# %%
# Denial rate by drug tier
claims_tier = claims.merge(drugs[["ndc_code", "drug_tier"]], on = "ndc_code")
denial_by_tier = claims_tier.groupby("drug_tier")["denied"].mean()
print("Denial rate by drug tier:")
denial_by_tier

#===========================================================================



# %%
# Denial rate within problem accounts vs. everyone else
claims_client = (
    claims
    .merge(members[["member_id", "client_id"]], on="member_id")
    .merge(clients[["client_id", "is_problem_account"]], on="client_id")
)
denial_by_account_health = claims_client.groupby("is_problem_account")["denied"].mean()
print("Denial rate within problem account vs. within everyone else")
denial_by_account_health

#===========================================================================



# %%
# Within pa table, calculate turnaround_hours of all requests
pa["turnaround_hours"] = (pa["decision_ts"] - pa["submitted_ts"]).dt.total_seconds()/3600
print("PA turnaround, all request")
pa["turnaround_hours"].describe()[["min", "mean", "50%", "max"]]

#===========================================================================



# %%
# Overall SLA breach rate
pa_full =(
    pa
    .merge(claims[["claim_id", "member_id"]], on="claim_id")
    .merge(members[["member_id", "client_id"]], on="member_id")
    .merge(clients[["client_id", "pa_sla_hours", "is_problem_account"]], on="client_id")
)
pa_full["breach"] = pa_full["turnaround_hours"] > pa_full["pa_sla_hours"]
overall_breach_rate = pa_full["breach"].mean()
print(f"Overall SLA breach rate: {overall_breach_rate:.1%}")

#===========================================================================



# %%
# SLA breach rate within problem accounts vs. everyone else
breach_by_account_health = pa_full.groupby("is_problem_account")["breach"].mean()
print("SLA breach rate within problem accounts vs. within everyone else")
breach_by_account_health

#===========================================================================



# %%
# Sanity-check summary
summary = {
    "metric": [
        "Overall denial rate",
        "Denial rate — problem accounts",
        "Denial rate — other accounts",
        "Overall SLA breach rate",
        "SLA breach rate — problem accounts",
        "SLA breach rate — other accounts",
        "Median PA turnaround (hrs)",
    ],

    "value": [
        f"{overall_denial_rate:.1%}",
        f"{denial_by_account_health[True]:.1%}",
        f"{denial_by_account_health[False]:.1%}",
        f"{overall_breach_rate:.1%}",
        f"{breach_by_account_health[True]:.1%}",
        f"{breach_by_account_health[False]:.1%}",
        f"{pa['turnaround_hours'].median():.1f}",
    ],
}

checks = pd.DataFrame(summary)
checks
# %%