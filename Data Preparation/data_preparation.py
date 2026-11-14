"""
data_preparation.py
Team 5, Customer Churn Analysis. Stage 2: Data Preparation (written by the Data Engineers).

What this script does, in order:
  1. Loads the raw file Dataset_ATS_v2.csv.
  2. Checks for missing data and handles it (this file has none).
  3. Fixes one logic problem: MultipleLines = Yes for customers who have no phone service.
  4. Encodes the text columns as numbers.
  5. Splits the customers into training rows (80%) and test rows (20%).
  6. Scales tenure and MonthlyCharges to the range 0 to 1. The scaler learns from the training rows only.
  7. Saves every file, and prints a short quality report.

How to run (from the main folder of the repository):
    python Data_Preparation/Code/data_preparation.py

Needs: Python 3.9 or newer, pandas, numpy, scikit-learn, joblib.
"""

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------------------------
# Settings. These are the team decisions. Do not change them unless the team agrees.
# ----------------------------------------------------------------------------------------
SEED = 42                      # keeps the split the same every time
TEST_SHARE = 0.2               # 20% of the customers are kept for the final test
SCALE_COLS = ["tenure", "MonthlyCharges"]                                   # scaled to the range 0 to 1
CLUSTER_COLS = ["tenure", "MonthlyCharges", "Dependents", "SeniorCitizen"]  # the four columns K-Means uses
BINARY_MAPS = {                # text value -> number
    "gender": {"Female": 0, "Male": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "Yes": 1},
    "InternetService": {"DSL": 0, "Fiber optic": 1},
    "Churn": {"No": 0, "Yes": 1},
}
CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]

# ----------------------------------------------------------------------------------------
# Folders. The script finds them from its own place in the repository.
# ----------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
parser = argparse.ArgumentParser(description="Prepare the ATS customer data (Stage 2).")
parser.add_argument("--raw", default=str(REPO / "Raw_Data" / "Dataset_ATS_v2.csv"), help="the raw data file")
parser.add_argument("--out", default=str(REPO / "Data_Preparation"), help="the Data_Preparation folder")
args = parser.parse_args()
RAW_FILE = Path(args.raw)
OUT = Path(args.out)
DIR_DATA = OUT / "Preprocessed_Dataset"
DIR_SETS = OUT / "Training_Testing_Sets"
DIR_SCALE = OUT / "Scaling"
for folder in (DIR_DATA, DIR_SETS, DIR_SCALE):
    folder.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------------------
# Step 2. Missing data
# ----------------------------------------------------------------------------------------
def handle_missing(raw):
    """Count empty cells. If there are any, fill numbers with the median and text with the most common value.
    (This file has no empty cells, so nothing changes. The rule is here so the method is on record.)"""
    empty_cells = int(raw.isna().sum().sum())
    blank_text = int(sum(raw[c].astype(str).str.strip().eq("").sum() for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])))
    if empty_cells:
        for col in raw.columns:
            if raw[col].isna().any():
                raw[col] = raw[col].fillna(raw[col].median() if pd.api.types.is_numeric_dtype(raw[col]) else raw[col].mode()[0])
    return raw, empty_cells, blank_text


# ----------------------------------------------------------------------------------------
# Step 3. Fix the logic problem
# ----------------------------------------------------------------------------------------
def fix_phone_lines(raw):
    """A customer without phone service cannot have several phone lines, so MultipleLines Yes becomes No."""
    clean = raw.copy()
    wrong = (clean["PhoneService"] == "No") & (clean["MultipleLines"] == "Yes")
    clean.loc[wrong, "MultipleLines"] = "No"
    return clean, int(wrong.sum())


# ----------------------------------------------------------------------------------------
# Step 4. Encoding
# ----------------------------------------------------------------------------------------
def encode(clean):
    """Text columns become numbers. Contract becomes three 0/1 columns. Churn is the last column."""
    encoded = clean.copy()
    for col, mapping in BINARY_MAPS.items():
        encoded[col] = encoded[col].map(mapping)
    encoded = pd.get_dummies(encoded, columns=["Contract"], dtype=int)
    feature_cols = [c for c in encoded.columns if c != "Churn"]
    return encoded[feature_cols + ["Churn"]], feature_cols


def encoding_map():
    """The code book: every original value and the number that replaces it."""
    rows = [(c, k, v) for c, m in BINARY_MAPS.items() for k, v in m.items()]
    for k in CONTRACT_TYPES:
        rows.append(("Contract", k, f"column Contract_{k} = 1 (the other two Contract columns = 0)"))
    return pd.DataFrame(rows, columns=["Column", "Original value", "Coded as"])


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------------
raw = pd.read_csv(RAW_FILE)
print(f"Raw data: {len(raw):,} rows and {raw.shape[1]} columns")

raw, empty_cells, blank_text = handle_missing(raw)
print(f"Empty cells: {empty_cells}.  Blank text cells: {blank_text}.")

clean, n_fixed = fix_phone_lines(raw)
print(f"Phone-line values changed from Yes to No: {n_fixed}")

encoded, all_cols = encode(clean)

# Step 5. Split. Stratify on Churn so both parts keep the same share of customers who left.
X, y = encoded[all_cols], encoded["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SHARE, stratify=y, random_state=SEED)

# Step 6. Scale. The scaler learns the smallest and largest value from the TRAINING rows only.
scaler = MinMaxScaler().fit(X_train[SCALE_COLS])


def scaled(table):
    table = table.copy()
    table[SCALE_COLS] = scaler.transform(table[SCALE_COLS])
    return table


train = scaled(X_train).assign(Churn=y_train)
test = scaled(X_test).assign(Churn=y_test)
everyone = scaled(X).assign(Churn=y)
cluster = everyone[CLUSTER_COLS + ["Churn"]]

# ----------------------------------------------------------------------------------------
# Step 7. Save
# ----------------------------------------------------------------------------------------
files = {
    DIR_DATA / "ats_clean_final.csv": clean,                  # cleaned, values still readable text
    DIR_DATA / "ats_preprocessed_final.csv": encoded,         # cleaned and encoded, not scaled
    DIR_DATA / "ats_cluster_final.csv": cluster,              # the four clustering columns (scaled) and Churn
    DIR_SETS / "ats_ann_train_final.csv": train,              # scaled, 80% of the customers
    DIR_SETS / "ats_ann_test_final.csv": test,                # scaled, 20% of the customers
}
for path, table in files.items():
    table.to_csv(path, index=False)
encoding_map().to_csv(DIR_DATA / "encoding_map_final.csv", index=False)
joblib.dump(scaler, DIR_SCALE / "scaler_final.joblib")

# ---- tables that document the split and the scaling ----
def composition(name, rows):
    return {
        "Set": name,
        "Rows": len(rows),
        "Share of all rows (%)": round(len(rows) / len(encoded) * 100, 2),
        "Customers who left": int(rows["Churn"].sum()),
        "Churn rate (%)": round(rows["Churn"].mean() * 100, 2),
        "Senior citizens (%)": round(rows["SeniorCitizen"].mean() * 100, 2),
        "With dependents (%)": round(rows["Dependents"].mean() * 100, 2),
        "Fiber optic internet (%)": round(rows["InternetService"].mean() * 100, 2),
        "Average tenure": round(rows["tenure"].mean(), 2),
        "Average MonthlyCharges": round(rows["MonthlyCharges"].mean(), 2),
    }


unscaled_train = X_train.assign(Churn=y_train)
unscaled_test = X_test.assign(Churn=y_test)
pd.DataFrame([composition("All customers", encoded), composition("Training set", unscaled_train),
              composition("Test set", unscaled_test)]).to_csv(DIR_SETS / "split_composition.csv", index=False)

scale_table = pd.DataFrame({
    "Column": SCALE_COLS,
    "Smallest value in training rows": scaler.data_min_,
    "Largest value in training rows": scaler.data_max_,
    "Range": scaler.data_max_ - scaler.data_min_,
    "Training rows used": int(scaler.n_samples_seen_),
})
scale_table.to_csv(DIR_SCALE / "scaling_values.csv", index=False)

rows = []
for col in SCALE_COLS:
    for set_name, before, after in (("Training set", unscaled_train, train), ("Test set", unscaled_test, test)):
        for stage, table in (("Before scaling", before), ("After scaling", after)):
            rows.append({"Column": col, "Set": set_name, "Stage": stage, "Minimum": round(table[col].min(), 4),
                         "Maximum": round(table[col].max(), 4), "Mean": round(table[col].mean(), 4),
                         "Standard deviation": round(table[col].std(), 4)})
pd.DataFrame(rows).to_csv(DIR_SCALE / "scaling_before_after.csv", index=False)

# ----------------------------------------------------------------------------------------
# Quality report
# ----------------------------------------------------------------------------------------
# Test rows that have exactly the same details as a training row (for information only)
key = lambda table: table[all_cols].round(8).astype(str).agg("|".join, axis=1)
twins = int(key(test).isin(set(key(train))).sum())
model_files = [train, test, cluster]
both_parts = pd.concat([train, test])
same_customers = (len(cluster) == len(both_parts)
                  and np.allclose(np.sort(cluster["tenure"]), np.sort(both_parts["tenure"]))
                  and np.allclose(np.sort(cluster["MonthlyCharges"]), np.sort(both_parts["MonthlyCharges"])))
empty_total = sum(int(table.isna().sum().sum()) for table in model_files)
overlap = len(set(train.index) & set(test.index))
checks = [
    ("Training + test rows equal the cleaned rows", len(train) + len(test) == len(clean),
     f"{len(train)} + {len(test)} = {len(train) + len(test)}; cleaned rows = {len(clean)}"),
    ("Training and test files have the same columns", list(train.columns) == list(test.columns), f"{train.shape[1]} columns in both"),
    ("No row is in two parts", overlap == 0, f"{overlap} rows in two parts"),
    ("No empty cells in the model files", empty_total == 0, f"{empty_total} empty cells"),
    ("Every column is a number", all(pd.api.types.is_numeric_dtype(t) for table in model_files for t in table.dtypes), "no text columns left"),
    ("Churn holds only 0 and 1", all(set(table["Churn"].unique()) <= {0, 1} for table in model_files), "0 = stayed, 1 = left"),
    ("Churn share is the same in training and test", abs(train["Churn"].mean() - test["Churn"].mean()) < 0.001,
     f"{train['Churn'].mean() * 100:.2f}% and {test['Churn'].mean() * 100:.2f}%"),
    ("The scaler learned from the training rows only", int(scaler.n_samples_seen_) == len(train),
     f"scaler used {int(scaler.n_samples_seen_)} rows; training rows = {len(train)}"),
    ("Scaled columns run from 0 to 1 in training",
     bool(np.isclose(train[SCALE_COLS].min().max(), 0) and np.isclose(train[SCALE_COLS].max().min(), 1)),
     ", ".join(f"{c} {train[c].min():.2f} to {train[c].max():.2f}" for c in SCALE_COLS)),
    ("Churn is not one of the feature columns", "Churn" not in all_cols, f"{len(all_cols)} feature columns"),
    ("The clustering file and the ANN files hold the same customers", bool(same_customers), f"{len(cluster)} customers in both"),
]
rows_q = [(name, "PASS" if ok else "FAIL", detail) for name, ok, detail in checks]
rows_q.append(("Test rows with the same details as a training row", "INFO", f"{twins} of {len(test)} ({twins / len(test) * 100:.1f}%)"))
quality = pd.DataFrame(rows_q, columns=["Check", "Result", "Detail"])
quality.to_csv(DIR_DATA / "quality_checks.csv", index=False)

print("\nFiles saved:")
for path in list(files) + [DIR_DATA / "encoding_map_final.csv"]:
    shape = pd.read_csv(path).shape
    print(f"  {path.relative_to(REPO)}   {shape[0]:,} rows x {shape[1]} columns   SHA-256 {sha256_of(path)[:12]}")
print(f"  {(DIR_SCALE / 'scaler_final.joblib').relative_to(REPO)}")
print(f"\nTraining rows: {len(train):,}   Test rows: {len(test):,}")
print(f"Churn share: all {y.mean() * 100:.2f}%   training {y_train.mean() * 100:.2f}%   test {y_test.mean() * 100:.2f}%")
print("\nQuality checks:")
print(quality.to_string(index=False))
if (quality["Result"] == "FAIL").any():
    raise SystemExit("A quality check failed. Read the table above.")
