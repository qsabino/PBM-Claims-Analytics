# GOAL: 
# KMeans clustering on four client-level metrics to segment "healthy" vs. "at-risk" accounts, 
# validated against the known outlier clients.
#===========================================================================



# %%
# Global setup
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

#===========================================================================



# %%
# Load data
clients = pd.read_csv("../data/raw/clients.csv")
drugs = pd.read_csv("../data/raw/drugs.csv")
members = pd.read_csv("../data/raw/members.csv")
claims = pd.read_csv("../data/raw/claims.csv", parse_dates=["fill_date"])
pa = pd.read_csv("../data/raw/pa_requests.csv", parse_dates=["submitted_ts", "decision_ts"])

#===========================================================================



# %%
# Build one row per client: denial_rate, avg_pmpm, rebate_capture, breach_rate

# Find denial_rate and avg_pmpm
member_counts = members.grooupby("client_id").size().rename("n_members")
claims_c = claims.merge(members[["member_id", "client_id"]], on="member_id")

denial_rate = claims_c.groupby("client_id")["denied"].mean().rename("denial_rate")

avg_pmpm = (claims_c.groupby("client_id")["plan_paid"].sum()/member_counts/12).rename("avg_pmpm")

#===========================================================================



# %%
# Find rebate_capture
claims_d = claims_c.merge(drugs[["ndc_code", "drug_tier"]], on="ndc_code")

paid = claims_d[~claims_d["denied"]].copy()
target_pct_by_tier = {1: 0.01, 2: 0.05, 3: 0.15, 4: 0.22}

paid["expected_rebate"] = paid["awp"] * paid["drug_tier"].map(target_pct_by_tier)

rebate_capture = (
    paid.groupby("client_id")["rebate_amount"].sum()
    /
    paid.groupby("client_id")["expected_rebate"].sum()
).rename("rebate_capture")

#===========================================================================



# %%
# Find breach_rate
pa_c = (
    pa
    .merge(claims[["claim_id", "member_id"]], on="claim_id")
    .merge(members[["member_id", "client_id"]], on="member_id")
    .merge(clients[["client_id", "pa_sla_hours"]], on="client_id")
)

pa_c["turnaround_hours"] = (pa_c["decision_ts"] - pa_c["submitted_ts"]).dt.total_seconds() / 3600
pa_c["breach"] = pa_c["turnaround_hours"] > pa_c["pa_sla_hours"]

breach_rate = pa_c.groupby("client_id")["breach"].mean().rename("breach_rate")
breach_rate.head()

#===========================================================================



# %%
# Combind these 4 features, placing them side by side as separate feature columns using shared indexed client_id
features = pd.concat(
    [denial_rate, avg_pmpm, rebate_capture, breach_rate, member_counts],
    axis=1
).reset_index() # reset index client_id into a real column, used to join

#  client info and its features
client_features = features.merge(
    clients[["client_id", "client_name", "is_problem_account"]],
    on="client_id"
)

client_features.head()

#===========================================================================



# %%
# Remove clients with so few members that their denial/breach/capture rates are mostly noise
MIN_MEMBERS = 100
df = client_features[client_features["n_members"]>MIN_MEMBERS].reset_index()

# How many clients left
print(f"{len(df)} of {len(client_features)} clients have enough volumn of members to avoid noise")

# Any Nulls remaining in df
print(f" Nulls remaining: {df.isnull().sum().sum()}")

#===========================================================================



# %%
# Client segmentation using KMeans
# Standardized berfore clustering
feature_cols = ["denial_rate", "avg_pmpm", "rebate_capture", "breach_rate"]
X = df[feature_cols].values # a plain matrix math X needed for ML, X=(35 clients, 4 features)
X_scaled = StandardScaler().fit_transform(X)

#===========================================================================



# %%
# How many clusters?
# Silhouette score measures how well-separated clusters are.
# Try a range of k=n_clusters, higher is better, max 1.0
for k in range(2, 6):
    km = KMeans(n_clusters=k, 
                random_state=42, # result reproducible
                n_init=10 # run 10 times then keep best fitting clusters
                ).fit(X_scaled) # run the algorithm on the scaled data
                # km stores which cluster each row belongs to

    score = silhouette_score(X_scaled, km.labels_)
    print(f"k={k}: silhouette = {score:.3f}")

#===========================================================================



# %%
# Fit the final model with the best k=2 clusters
# Let's say healthy and at-risk groups of clients
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Per cluster, calculate mean of each feature
cluster_profile = df.groupby("cluster")[feature_cols].mean()

# Find n_clients for each cluster
cluster_profile["n_clients"] = df.groupby("cluster").size()
cluster_profile
# Observed: Cluster 1 is the at-risk group: 
# Roughly double the denial rate
# More than double the SLA breach rate
# Lower PMPM than cluster 0.
# Lower PMPM because denied claims pay $0, so a client denying more of its claims naturally shows less total spend, not because they're actually cheaper to cover.



# %%
# Does these segments match with the known "problem accounts"?
# is_problem_account is used to check whether the unsupervised method independently rediscovers it.
pd.crosstab(
    df["cluster"], 
    df["is_problem_account"]
)

#===========================================================================



# %%
# View all clients of cluster 1 and their features
df[df["cluster"] == 1][["client_id", "client_name", "is_problem_account"] + feature_cols]
# Observed: Client CL008 not flagged as is_problem_account but landed in cluster 1 (at-risk)
# It might be a real account worth a closer review

#===========================================================================



# %%
# Visualizing the split use PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df["pc1"] = components[:, 0]
df["pc2"] = components[:, 1]

print(f"Variance explained: PC1 = {pca.explained_variance_ratio_[0]:.1%}, "
      f"PC2 = {pca.explained_variance_ratio_[1]:.1%}, "
      f"total = {pca.explained_variance_ratio_.sum():.1%}")

#===========================================================================



# %%
# Plot client segmentation
fig, ax = plt.subplots(figsize=(7, 6))

colors = {0: "steelblue", 1: "firebrick"}
for cluster_id, group in df.groupby("cluster"):
    ax.scatter(
        group["pc1"], group["pc2"],
        c=colors[cluster_id], label=f"Cluster {cluster_id}",
        s=80, alpha=0.7, edgecolor="white",
    )

# ring around clients that were flagged as true problem accounts
true_problems = df[df["is_problem_account"]]
ax.scatter(
    true_problems["pc1"], true_problems["pc2"],
    facecolors="none", edgecolors="black", s=200, linewidths=1.5,
    label="Known problem account",
)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
ax.set_title("Client segmentation — PCA projection")
ax.legend()
plt.tight_layout()
plt.show()

#===========================================================================



# %%
# What does each principal component actually represent?
# PCA components are combinations of the original features. This shows what's driving the separation.
loadings = pd.DataFrame(
    pca.components_.T, index=feature_cols, columns=["PC1", "PC2"]
)
loadings

#===========================================================================



# %%
summary = pd.DataFrame({
    "metric": [
        "Clients with enough volume to cluster",
        "Chosen k (by silhouette score)",
        "Clients in at-risk cluster",
        "Known problem accounts recovered",
        "Additional client flagged (not in original answer key)",
        "PC1 + PC2 variance explained",
    ],
    "value": [
        f"{len(df)} of {len(client_features)}",
        "2",
        f"{(df['cluster'] == 1).sum()}",
        f"{df[df['cluster']==1]['is_problem_account'].sum()} of {df['is_problem_account'].sum()}",
        f"{(df['cluster']==1).sum() - df[df['cluster']==1]['is_problem_account'].sum()}",
        f"{pca.explained_variance_ratio_.sum():.1%}",
    ],
})
summary
# %%