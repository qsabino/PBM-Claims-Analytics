# GOAL: Predict awp (the drug cost of a claim) using only information available before adjudication 
#===========================================================================


# %%
# Global setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
RANDOM_STATE = 42


#===========================================================================


# %%
# Load and merge data
clients = pd.read_csv("../data/raw/clients.csv")
drugs = pd.read_csv("../data/raw/drugs.csv")
members = pd.read_csv("../data/raw/members.csv")
claims = pd.read_csv("../data/raw/claims.csv", parse_dates=["fill_date"])

df = (
    claims
    .merge(members[["member_id", "client_id"]], on="member_id")
    .merge(clients[["client_id", "plan_type", "pa_sla_hours"]], on="client_id")
    .merge(drugs[["ndc_code", "drug_tier", "requires_pa", "base_awp"]], on="ndc_code")
)
print(f"{len(df):,} claims, target (awp) range: ${df['awp'].min():.2f} - ${df['awp'].max():.2f}")


#===========================================================================


# %%
# Leakage check

train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE)
train = df.loc[train_idx].copy()
test = df.loc[test_idx].copy()
y_train, y_test = train["awp"], test["awp"]

print(f"Train: {len(train):,} claims  |  Test: {len(test):,} claims")


#===========================================================================


# %%
# Build the feature matrix
def build_features(d, num_cols, cat_cols, reference_columns=None):
    X = d[num_cols + ["requires_pa"]].copy()
    X["requires_pa"] = X["requires_pa"].astype(int)
    X = pd.concat([X, pd.get_dummies(d[cat_cols].astype(str), drop_first=True)], axis=1)
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0)
    return X

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"{name:35s}  R²={r2:.4f}   MAE=${mae:,.2f}   RMSE=${rmse:,.2f}")
    return {"model": name, "r2": r2, "mae": mae, "rmse": rmse}

results = []


#===========================================================================


# %%
# Model A: just drug_tier as a 4-category bucket, plus the other claim/client attributes from notebook 01's feature set.

num_A = ["days_supply", "pa_sla_hours"]
cat_A = ["drug_tier", "pharmacy_type", "plan_type"]

X_train_A = build_features(train, num_A, cat_A)
X_test_A = build_features(test, num_A, cat_A, reference_columns=X_train_A.columns)

lr_A = LinearRegression().fit(X_train_A, y_train)
results.append(evaluate("A: Linear (tier bucket)", y_test, lr_A.predict(X_test_A)))

rf_A = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
rf_A.fit(X_train_A, y_train)
results.append(evaluate("A: Random Forest (tier bucket)", y_test, rf_A.predict(X_test_A)))

# Observed: Random forest does noticeably better than linear regression with the same features
# showing that the relationship between these features isn't purely additive, 
# which is exactly the kind of thing a tree model can pick up on without being told.


#===========================================================================


# %%
# Model B: add base_awp — a continuous drug-level value
# Going from a categorical bucket drug-tier to the real underlying continuous variable should matter?

num_B = ["days_supply", "pa_sla_hours", "base_awp"]

X_train_B = build_features(train, num_B, cat_A)
X_test_B = build_features(test, num_B, cat_A, reference_columns=X_train_B.columns)

lr_B = LinearRegression().fit(X_train_B, y_train)
results.append(evaluate("B: Linear (+ base_awp)", y_test, lr_B.predict(X_test_B)))

rf_B = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
rf_B.fit(X_train_B, y_train)
results.append(evaluate("B: Random Forest (+ base_awp)", y_test, rf_B.predict(X_test_B)))

# Observed: Random forest jumps to ~0.99 R²; linear regression barely improves. 
# That gap is the interesting part. A tree model can approximate almost any relationship
# Plain linear regression only looks for relationships where each feature adds a fixed amount to the prediction.


#===========================================================================


# %%
# Model C: give linear regression the interaction it actually needs
# Tests whether linear regression was ever really "worse," or just missing the right input.

for d in [train, test]:
    d["base_awp_x_ds"] = d["base_awp"] * (d["days_supply"] / 30)

X_train_C = train[["base_awp_x_ds"]]
X_test_C = test[["base_awp_x_ds"]]

lr_C = LinearRegression().fit(X_train_C, y_train)
lr_C_pred = lr_C.predict(X_test_C)
results.append(evaluate("C: Linear (engineered interaction)", y_test, lr_C_pred))

# Observed: One engineered feature, and linear regression matches the random forest's ~0.99 R²
# A "worse" model type wasn't actually worse, it just needed the underlying mechanism handed to it explicitly


#===========================================================================


# %%
# Comparing all five models

results_df = pd.DataFrame(results).set_index("model")
results_df


#===========================================================================


# %%
# Model comparision visualization

fig, ax = plt.subplots(figsize=(8, 4.5))
results_df["r2"].plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("R² on test set")
ax.set_title("Claim cost regression — model comparison")
ax.set_xlim(0, 1)
plt.tight_layout()
plt.show()


#===========================================================================


# %%
# Residual check on the best model
# Confirms there's no leftover pattern the model is missing, and no systematic bias

residuals = y_test - lr_C_pred

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(y_test, lr_C_pred, alpha=0.15, s=8, color="#4C72B0")
lims = [0, max(y_test.max(), lr_C_pred.max())]
axes[0].plot(lims, lims, "k--", alpha=0.5, label="Perfect prediction")
axes[0].set_xlabel("Actual AWP ($)")
axes[0].set_ylabel("Predicted AWP ($)")
axes[0].set_title("Actual vs. Predicted")
axes[0].legend()

axes[1].scatter(lr_C_pred, residuals, alpha=0.15, s=8, color="#C44E52")
axes[1].axhline(0, color="k", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Predicted AWP ($)")
axes[1].set_ylabel("Residual (actual − predicted)")
axes[1].set_title("Residuals vs. Predicted")

plt.tight_layout()
plt.show()

# Observed: Residuals scatter evenly around zero with no funnel shape or curve
# No sign the model is missing a systematic pattern it should be capturing.


#===========================================================================


# %%
# Is this stable across different splits? Cross-validation

cv_scores = cross_val_score(
    LinearRegression(), train[["base_awp_x_ds"]], y_train, cv=5, scoring="r2"
)
print(f"CV R² per fold: {cv_scores.round(4)}")
print(f"Mean: {cv_scores.mean():.4f}  |  Std: {cv_scores.std():.4f}")

# Oserved: Std: 0.0002 --> stable


#===========================================================================


# %%
# Feature importance, random forest (Model B)

importances = pd.Series(rf_B.feature_importances_, index=X_train_B.columns).sort_values(ascending=False)
importances


#===========================================================================


# %%
# Summary

summary = pd.DataFrame({
    "metric": [
        "Best model",
        "Best model R²",
        "Best model MAE",
        "R² gain from tier bucket → base_awp (Random Forest)",
        "R² gain from adding base_awp alone (Linear, additive)",
        "R² after engineering the interaction (Linear)",
        "5-fold CV R² (mean ± std)",
    ],
    "value": [
        "C: Linear (engineered interaction)",
        f"{results_df.loc['C: Linear (engineered interaction)', 'r2']:.4f}",
        f"${results_df.loc['C: Linear (engineered interaction)', 'mae']:,.2f}",
        f"{results_df.loc['B: Random Forest (+ base_awp)', 'r2'] - results_df.loc['A: Random Forest (tier bucket)', 'r2']:.4f}",
        f"{results_df.loc['B: Linear (+ base_awp)', 'r2'] - results_df.loc['A: Linear (tier bucket)', 'r2']:.4f}",
        f"{results_df.loc['C: Linear (engineered interaction)', 'r2']:.4f}",
        f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
    ],
})
summary
# %%
