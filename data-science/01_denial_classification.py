# GOAL: Predict whether a claim will be denied, using only information available before adjudication.
# This is a genuinely imbalanced classification problem (denial rate ~7.5%)
#===========================================================================


# %%
# Global setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
RANDOM_STATE = 42


#===========================================================================


# %%
# Load and Merge data

clients = pd.read_csv("../data/raw/clients.csv")
drugs = pd.read_csv("../data/raw/drugs.csv")
members = pd.read_csv("../data/raw/members.csv")
claims = pd.read_csv("../data/raw/claims.csv", parse_dates=["fill_date"])

df = (
    claims
    .merge(members[["member_id", "client_id"]], on="member_id")
    .merge(clients[["client_id", "plan_type", "pa_sla_hours"]], on="client_id")
    .merge(drugs[["ndc_code", "drug_tier", "requires_pa"]], on="ndc_code")
)
print(f"{len(df):,} claims, denial rate {df['denied'].mean():.1%}")


#===========================================================================


# %%
# Train/test split

train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, stratify=df["denied"], random_state=RANDOM_STATE
)
train = df.loc[train_idx].copy()
test = df.loc[test_idx].copy()

print(f"Train: {len(train):,} claims, denial rate {train['denied'].mean():.1%}")
print(f"Test:  {len(test):,} claims, denial rate {test['denied'].mean():.1%}")


#===========================================================================


# %%
# Client historical denial rate

global_rate = train["denied"].mean()
client_rate_map = train.groupby("client_id")["denied"].mean()

train["client_hist_denial_rate"] = train["client_id"].map(client_rate_map)
test["client_hist_denial_rate"] = test["client_id"].map(client_rate_map).fillna(global_rate)

print(f"Clients not in train set: "
      f"{(~test['client_id'].isin(client_rate_map.index)).sum()} of {len(test)}")


#===========================================================================


# %%
# Build the feature matrix

num_cols = ["days_supply", "awp", "dispensing_fee", "copay", "pa_sla_hours", "client_hist_denial_rate"]
cat_cols = ["drug_tier", "pharmacy_type", "plan_type"]

def build_features(d, reference_columns=None):
    X = d[num_cols + ["requires_pa"]].copy()
    X["requires_pa"] = X["requires_pa"].astype(int)
    X = pd.concat([X, pd.get_dummies(d[cat_cols].astype(str), drop_first=True)], axis=1)
    if reference_columns is not None:
        # keep train/test columns aligned even if a category is missing from any side
        X = X.reindex(columns=reference_columns, fill_value=0)
    return X

X_train = build_features(train)
X_test = build_features(test, reference_columns=X_train.columns)
y_train = train["denied"].astype(int)
y_test = test["denied"].astype(int)

print(f"Feature matrix: {X_train.shape[1]} features")
X_train.columns.tolist()


#===========================================================================


# %%
# The majority-class baseline. Always predict "not denied"

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print(f"Baseline accuracy: {baseline.score(X_test, y_test):.1%}")
print(classification_report(y_test, baseline_pred, zero_division=0))

# Observation: 92.5% accuracy, and it's completely useless
# recall on the denied class is 0.0, because the baseline never predicts a denial at all.


#===========================================================================


# %%
# Model 1: Logistic Regression
# class_weight="balanced" tells the model to weight the minority (denied) class more heavily during training

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
logreg.fit(X_train_scaled, y_train)

logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
logreg_pred = logreg.predict(X_test_scaled)

print(f"ROC-AUC: {roc_auc_score(y_test, logreg_proba):.3f}")
print(classification_report(y_test, logreg_pred, zero_division=0))


#===========================================================================


# %%
# Model 2: Random Forest
# A tree-based model, for comparison.
# Doesn't need feature scaling, and can pick up non-linear or interaction effects

rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)

print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.3f}")
print(classification_report(y_test, rf_pred, zero_division=0))


#===========================================================================


# %%
# Model 3: XGBoost
# scale_pos_weight is XGBoost's version of class_weight="balanced": 
# it's set to the ratio of negative to positive cases, 
# so the model is penalized more for missing a denial than for misclassifying a routine claim.

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb.fit(X_train, y_train)

xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_pred = xgb.predict(X_test)

print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.3f}")
print(classification_report(y_test, xgb_pred, zero_division=0))

# Observation: XGBoost lands the same ROC-AUC as logistic regression and random forest, about 0.70
# A more sophisticated algorithm does not move the ceiling.
# This is the point where further model tuning stops being the right use of time, 
# and the conversation shifts to whether better features (not better models) could close the gap.


#===========================================================================


# %%
# Confusion matrices, side by side

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ConfusionMatrixDisplay.from_predictions(
    y_test, logreg_pred, ax=axes[0], colorbar=False,
    display_labels=["Not denied", "Denied"]
)
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_pred, ax=axes[1], colorbar=False,
    display_labels=["Not denied", "Denied"]
)
axes[1].set_title("Random Forest")

ConfusionMatrixDisplay.from_predictions(
    y_test, xgb_pred, ax=axes[2], colorbar=False,
    display_labels=["Not denied", "Denied"]
)
axes[2].set_title("XGBoost")

plt.tight_layout()
plt.show()


#===========================================================================


# %%
# ROC and Precision-Recall curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

models = [
    ("Logistic Regression", logreg_proba),
    ("Random Forest", rf_proba),
    ("XGBoost", xgb_proba),
]

for name, proba in models:
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, label=name)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

for name, proba in models:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(rec, prec, label=name)
axes[1].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.4, label="Random guess")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.tight_layout()
plt.show()


#===========================================================================


# %%
# 5-fold cross-validation on the training set, using ROC-AUC as the scoring metric
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
])
cv_scores_logreg = cross_val_score(cv_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

cv_scores_xgb = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="roc_auc")

print(f"Logistic Regression — CV ROC-AUC per fold: {cv_scores_logreg.round(3)}")
print(f"  Mean: {cv_scores_logreg.mean():.3f}  |  Std: {cv_scores_logreg.std():.3f}")
print()
print(f"XGBoost — CV ROC-AUC per fold: {cv_scores_xgb.round(3)}")
print(f"  Mean: {cv_scores_xgb.mean():.3f}  |  Std: {cv_scores_xgb.std():.3f}")

# Observation: Both models land within 0.01 standard deviation across folds
# --> a stable result for both, not a lucky split for either one.


#===========================================================================


# %%
# What's actually driving the predictions?

rf_importances = pd.Series(rf.feature_importances_, index=X_train.columns, name="Random Forest")
xgb_importances = pd.Series(xgb.feature_importances_, index=X_train.columns, name="XGBoost")

importances = pd.concat([rf_importances, xgb_importances], axis=1).sort_values("Random Forest", ascending=False)
importances


#===========================================================================


# %%
# Importance visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

importances["Random Forest"].sort_values().tail(8).plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("Random Forest")
axes[0].set_xlabel("Feature importance")

importances["XGBoost"].sort_values().tail(8).plot(kind="barh", ax=axes[1], color="#55A868")
axes[1].set_title("XGBoost")
axes[1].set_xlabel("Feature importance")

fig.suptitle("Top predictors of denial — Random Forest vs. XGBoost")
plt.tight_layout()
plt.show()

# Oservation: two structurally different algorithms agreeing on what matters, 
# on top of already agreeing on how well it can be predicted


#===========================================================================


# %%
# Summary

summary = pd.DataFrame({
    "metric": [
        "Book-wide denial rate",
        "Baseline (majority class) accuracy",
        "Baseline recall on denied class",
        "Logistic Regression ROC-AUC",
        "Random Forest ROC-AUC",
        "XGBoost ROC-AUC",
        "XGBoost 5-fold CV ROC-AUC (mean \u00b1 std)",
        "Top predictor (agrees across RF and XGBoost)",
    ],
    "value": [
        f"{df['denied'].mean():.1%}",
        f"{baseline.score(X_test, y_test):.1%}",
        "0.0%",
        f"{roc_auc_score(y_test, logreg_proba):.3f}",
        f"{roc_auc_score(y_test, rf_proba):.3f}",
        f"{roc_auc_score(y_test, xgb_proba):.3f}",
        f"{cv_scores_xgb.mean():.3f} \u00b1 {cv_scores_xgb.std():.3f}",
        importances["Random Forest"].idxmax(),
    ],
})
summary
# %%
