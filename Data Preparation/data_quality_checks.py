"""
data_quality_checks.py
Team 5, Customer Churn Analysis. Stage 2: Data Preparation (written by the Data Engineers).

This script repeats the first checks we made on the raw file, before any cleaning.
It gives the counts and scores that the Data Preparation documents quote:
  1. A profile of every column (type, empty cells, smallest and largest value).
  2. The counts: empty cells, outliers (IQR rule), repeated rows, customers with tenure 0,
     phone-line values that cannot be true, and the share of customers who left.
  3. The effect of each cleaning choice on a simple model (logistic regression, 5-fold cross-validation,
     repeated 3 times). It shows why we fixed the phone-line values and kept the repeated rows.

The script only reads the raw file. It saves nothing and changes nothing.
Compare what it prints with Data_Preparation/Code/data_quality_checks_output.txt.

How to run (from the main folder of the repository):
    python Data_Preparation/Code/data_quality_checks.py

Needs: Python 3.9 or newer, pandas and scikit-learn (SciPy comes with scikit-learn).
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
from scipy.optimize import OptimizeWarning
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------------------------
# Settings. They are the same as in data_preparation.py.
# ----------------------------------------------------------------------------------------
SCALE_COLS = ["tenure", "MonthlyCharges"]
BINARY_MAPS = {                # text value -> number
    "gender": {"Female": 0, "Male": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "Yes": 1},
    "InternetService": {"DSL": 0, "Fiber optic": 1},
    "Churn": {"No": 0, "Yes": 1},
}
FOLDS, REPEATS = 5, 3          # cross-validation: 5 folds, repeated 3 times (15 scores per version)

# Scikit-learn 1.6 with a newer SciPy prints this harmless warning for every model it fits. No number changes.
warnings.filterwarnings("ignore", message="Unknown solver options", category=OptimizeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
parser = argparse.ArgumentParser(description="Early quality checks on the raw ATS customer data (Stage 2).")
parser.add_argument("--raw", default=str(REPO / "Raw_Data" / "Dataset_ATS_v2.csv"), help="the raw data file")
args = parser.parse_args()
raw = pd.read_csv(args.raw)


def heading(text):
    print("\n" + text)
    print("-" * len(text))


def iqr_outliers(values):
    """Count values far outside the middle 50% (the usual IQR rule: beyond 1.5 x IQR from the quartiles)."""
    q1, q3 = values.quantile([0.25, 0.75])
    spread = q3 - q1
    return int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())


def make_clean(table, fix_phone_rows=True, drop_phone_rows=False, drop_repeated=False):
    """The cleaning choices we compared. The team chose: fix the phone rows, keep the repeated rows."""
    clean = table.copy()
    phone_rows = (clean["PhoneService"] == "No") & (clean["MultipleLines"] == "Yes")
    if drop_phone_rows:
        clean = clean.loc[~phone_rows].copy()
    elif fix_phone_rows:
        clean.loc[phone_rows, "MultipleLines"] = "No"
    if drop_repeated:
        clean = clean.drop_duplicates().copy()
    return clean


def encode(clean):
    """Text columns become numbers. Contract becomes three 0/1 columns. Same rules as data_preparation.py."""
    encoded = clean.copy()
    for col, mapping in BINARY_MAPS.items():
        encoded[col] = encoded[col].map(mapping)
    return pd.get_dummies(encoded, columns=["Contract"], dtype=int)


def cross_validated_auc(clean):
    """AUC of a logistic regression, from 5-fold cross-validation repeated 3 times.
    The scaler is fitted inside each fold, so nothing from the held-out rows leaks in."""
    encoded = encode(clean)
    features = [c for c in encoded.columns if c != "Churn"]
    model = make_pipeline(ColumnTransformer([("scale", MinMaxScaler(), SCALE_COLS)], remainder="passthrough"),
                          LogisticRegression(max_iter=1000))
    folds = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=0)   # fixed, so the folds are the same every time
    scores = cross_val_score(model, encoded[features], encoded["Churn"], cv=folds, scoring="roc_auc")
    return scores.mean(), scores.std()


# ----------------------------------------------------------------------------------------
# 1. Profile of the raw file
# ----------------------------------------------------------------------------------------
text_cols = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
print(f"Raw data: {len(raw):,} rows and {raw.shape[1]} columns")
heading("1. Profile of each column")
profile = pd.DataFrame({
    "Column": raw.columns,
    "Type": ["text" if c in text_cols else "number" for c in raw.columns],
    "Empty cells": raw.isna().sum().values,
    "Different values": raw.nunique().values,
    "Smallest": [raw[c].min() if c not in text_cols else "" for c in raw.columns],
    "Largest": [raw[c].max() if c not in text_cols else "" for c in raw.columns],
    "Values (text columns)": [" / ".join(sorted(raw[c].unique())) if c in text_cols else "" for c in raw.columns],
})
print(profile.to_string(index=False))

# ----------------------------------------------------------------------------------------
# 2. The counts
# ----------------------------------------------------------------------------------------
heading("2. Counts")
fixed = make_clean(raw)
no_phone = raw["PhoneService"] == "No"
phone_rows = no_phone & (raw["MultipleLines"] == "Yes")
tenure0 = raw["tenure"] == 0
left = int((raw["Churn"] == "Yes").sum())
share = left / len(raw) * 100
counts = [
    ("Empty cells", int(raw.isna().sum().sum()), "no empty cells"),
    ("Text values with spaces at the start or end", int(sum((raw[c] != raw[c].str.strip()).sum() for c in text_cols)), ""),
    ("tenure values that are not whole numbers", int((raw["tenure"] % 1 != 0).sum()), ""),
    ("MonthlyCharges values that are not whole numbers", int((raw["MonthlyCharges"] % 1 != 0).sum()), ""),
    ("Outliers in tenure (IQR rule)", iqr_outliers(raw["tenure"]), f"values run from {raw['tenure'].min()} to {raw['tenure'].max()}"),
    ("Outliers in MonthlyCharges (IQR rule)", iqr_outliers(raw["MonthlyCharges"]),
     f"values run from {raw['MonthlyCharges'].min():g} to {raw['MonthlyCharges'].max():g}"),
    ("Repeated rows in the raw file (all 10 columns the same)", int(raw.duplicated().sum()), "each extra copy counts once"),
    ("Repeated rows after the phone-line fix", int(fixed.duplicated().sum()), "the fix makes one more row a copy of an earlier row"),
    ("Customers without phone service", int(no_phone.sum()), ""),
    ("   of them with MultipleLines = Yes (cannot be true)", int(phone_rows.sum()), "changed to No in data_preparation.py"),
    ("Customers with tenure 0", int(tenure0.sum()), "probably new customers"),
    ("   of them who left", int((tenure0 & (raw["Churn"] == "Yes")).sum()), ""),
    ("Customers who left (Churn = Yes)", left, f"{share:.2f}% of {len(raw):,}"),
]
width = max(len(name) for name, _, _ in counts)
for name, value, note in counts:
    print(f"  {name:<{width}}  {value:>6,}   {note}")
print(f"  A model that always says No is right {100 - share:.2f}% of the time. Accuracy alone is not enough.")

# ----------------------------------------------------------------------------------------
# 3. Effect of each cleaning choice on a simple model
# ----------------------------------------------------------------------------------------
heading("3. Effect of each cleaning choice on a simple model")
print("Logistic regression, AUC from cross-validation. AUC runs from 0.5 (no skill) to 1 (perfect). Higher is better.\n")
versions = [
    ("Chosen: 260 changed to No, repeated rows kept", make_clean(raw)),
    ("No fix at all (original values)", make_clean(raw, fix_phone_rows=False)),
    ("260 phone-line rows removed", make_clean(raw, fix_phone_rows=False, drop_phone_rows=True)),
    ("Repeated rows removed", make_clean(raw, drop_repeated=True)),
]
rows = []
for label, table in versions:
    mean, spread = cross_validated_auc(table)
    rows.append({"Version of the data": label, "Rows": len(table), "AUC": round(mean, 3), "Spread between runs": round(spread, 3)})
result = pd.DataFrame(rows)
print(result.to_string(index=False))
best, worst = result["AUC"].max(), result["AUC"].min()
print(f"\nThe four AUC values lie within {best - worst:.3f} of each other. The spread between runs is about {result['Spread between runs'].mean():.3f}.")
print("A difference smaller than the spread between runs is too small to count. So no choice changes the model in a way we can measure.")
print("\nDone. This script saved nothing.")
