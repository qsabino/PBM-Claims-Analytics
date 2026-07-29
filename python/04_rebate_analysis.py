# GOAL: 
# Rebate capture rate (actual vs. contract-target) by tier and by client, 
# the total dollar gap, 
# why capture-rate variance shrinks with claim volume.
#===========================================================================



# %%
# Global setup
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
print(f"{len(merged):,} total claims")

#===========================================================================



# %%
# Fill to claims that actually carry a rebate, claims that were actually paid, not denied claims
paid = merged[~merged["denied"]].copy()
print(f"{len(paid):,} paid claims out of {len(merged):,} total ({len(paid)/len(merged):.1%})")

#===========================================================================



# %%
# Set the rebate contract target by drug_tier, usually as a target percentage of AWP
target_pct_by_tier = {1: 0.01, 2: 0.05, 3: 0.15, 4: 0.22}

# Calculate expected_rebate for each paid claim (to compare with actually rebate)
paid["target_pct"] = paid["drug_tier"].map(target_pct_by_tier)
paid["expected_rebate"] = paid["awp"]*paid["target_pct"]

# View
paid[["drug_tier", "awp", "target_pct", "expected_rebate", "rebate_amount"]].head()

#===========================================================================



# %%
# Calculate: capture rate = actual / expected. 
# 100% means the target was hit exactly; below 100% means dollars were left on the table.
overall_capture_rate = paid["rebate_amount"].sum() / paid["expected_rebate"].sum()
print(f"Overall rebate capture rate: {overall_capture_rate:.1%}")

# Capture rate by drug_tier
capture_by_tier = paid.groupby("drug_tier").agg(
    actual=("rebate_amount", "sum"),
    expected=("expected_rebate", "sum"),
)
capture_by_tier["capture_rate"] = capture_by_tier["actual"] / capture_by_tier["expected"]
capture_by_tier

#===========================================================================



# %%
# Capture rate by client
by_client = paid.groupby(["client_id", "client_name"]).agg(
    n_claims=("claim_id", "count"),
    total_actual=("rebate_amount", "sum"),
    total_expected=("expected_rebate", "sum"),
).reset_index()

by_client["capture_rate"] = by_client["total_actual"] / by_client["total_expected"]
by_client["gap_dollars"] = by_client["total_expected"] - by_client["total_actual"]

by_client.sort_values("capture_rate").head(10)

#===========================================================================



# %%
# Check whether low capture rate is actually tied to low claim volume
by_client["size_bucket"] = pd.qcut(by_client["n_claims"], q=3, labels=["small", "medium", "large"])

by_client.groupby("size_bucket").agg(
    avg_n_claims=("n_claims", "mean"),
    capture_rate_std=("capture_rate", "std"),
).round(3)
# Observed: Small clients show roughly 7x more spread in capture rate than large ones

#===========================================================================



# %%
# Plot n_claims vs capture_rate
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(by_client["n_claims"], by_client["capture_rate"], alpha=0.6, color="steelblue")
ax.axhline(overall_capture_rate, color="firebrick", linestyle="--", label="overall capture rate")
ax.set_xlabel("Number of claims")
ax.set_ylabel("Capture rate")
ax.set_title("Rebate capture rate vs. client claim volume")
ax.legend()
plt.tight_layout()
plt.show()

#===========================================================================



# %%
# Total dollars left on the table across the whole book-of-business
total_expected = by_client["total_expected"].sum()
total_actual = by_client["total_actual"].sum()
total_gap = total_expected - total_actual

print(f"Total expected rebate:  ${total_expected:,.0f}")
print(f"Total actual rebate:    ${total_actual:,.0f}")
print(f"Total gap:              ${total_gap:,.0f}")
print(f"Overall capture rate:   {total_actual/total_expected:.1%}")

#===========================================================================



# %%
# Capture rate trend over time
paid["month"] = paid["fill_date"].dt.to_period("M")
monthly_capture = paid.groupby("month").apply(
    lambda g: g["rebate_amount"].sum() / g["expected_rebate"].sum(),
    include_groups=False,
)
monthly_capture

#===========================================================================



# %%
# Plot capture rate trend
fig, ax = plt.subplots(figsize=(8, 4))
monthly_capture.plot(ax=ax, marker="o", color="steelblue")
ax.axhline(overall_capture_rate, color="gray", linestyle="--", label="overall capture rate")
ax.set_xlabel("Month")
ax.set_ylabel("Captute rate")
ax.set_title("Capture rate by month")
ax.legend()
plt.tight_layout()
plt.show()

#===========================================================================



# %%
# Summary
summary = pd.DataFrame({
    "metric": [
        "Paid claims (of total)",
        "Overall capture rate",
        "Total expected rebate",
        "Total actual rebate",
        "Total dollar gap",
        "Capture rate std — small clients",
        "Capture rate std — large clients",
    ],
    "value": [
        f"{len(paid):,} of {len(merged):,}",
        f"{overall_capture_rate:.1%}",
        f"${total_expected:,.0f}",
        f"${total_actual:,.0f}",
        f"${total_gap:,.0f}",
        f"{by_client.groupby('size_bucket')['capture_rate'].std()['small']:.3f}",
        f"{by_client.groupby('size_bucket')['capture_rate'].std()['large']:.3f}",
    ],
})
summary
# %%