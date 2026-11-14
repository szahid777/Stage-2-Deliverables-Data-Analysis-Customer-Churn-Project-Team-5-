# Team 5: Customer Churn Analysis, Stage 2

Customer churn project for the ACS WIL program. This repository holds the Stage 2 deliverables: **data preparation** (Data Engineers) and **clustering analysis with K-Means** (Data Analyst).

## Team

| Name | Role |
|---|---|
| Salman Zahid | Project Manager |
| Md Abdur Rahman | Data Engineer (data preparation) |
| Mir Asif Hossain | Data Engineer (data preparation) |
| Alisha Khadka | Data Analyst (clustering analysis) |
| Walid Bin Manju | Predictive modelling (ANN) |
| Hridaya Bisural | Business Analyst |

## The data

`Raw_Data/Dataset_ATS_v2.csv` has 7,043 customers and 10 columns: gender, SeniorCitizen, Dependents, tenure, PhoneService, MultipleLines, InternetService, Contract, MonthlyCharges and Churn. In total, 1,869 customers (26.54%) left. The file has no customer ID, and it does not state the units of tenure and MonthlyCharges. Our first checks on the raw file found no empty cells and no outliers. They found 302 repeated rows, 11 customers with tenure 0 and 260 phone-line values that cannot be true.

## What is in this repository

| Folder | What it holds |
|---|---|
| `Raw_Data/` | The raw dataset |
| `Data_Preparation/Code/` | `data_preparation.py` builds every data file from the raw data. `data_quality_checks.py` repeats our first checks on the raw file and prints the counts. `data_preparation_output.txt` and `data_quality_checks_output.txt` are what the two scripts print when they run. |
| `Data_Preparation/Preprocessed_Dataset/` | `ats_clean_final.csv` (readable text, after the logic fix), `ats_preprocessed_final.csv` (all customers, encoded, not scaled), `ats_cluster_final.csv` (the clustering file), `encoding_map_final.csv` (the code book) and `quality_checks.csv` |
| `Data_Preparation/Training_Testing_Sets/` | `ats_ann_train_final.csv`, `ats_ann_test_final.csv`, `split_composition.csv`, and the documentation of their size and composition (Word) |
| `Data_Preparation/Scaling/` | `scaler_final.joblib`, `scaling_values.csv`, `scaling_before_after.csv`, and the Scaling Techniques Documentation (Word, with code snippets) |
| `Data_Preparation/Data_Quality_Documents/` | `Data_Preparation_Documents.docx`: data preparation note, data dictionary, issues log, change log and quality checks |
| `Clustering_Analysis/` | `ACS_WIL_Clustering_Report.docx`: the clustering report (method, number of clusters, results, insights and limits) |
| `Clustering_Analysis/Code/` | The clustering notebook and the file it reads, `ats_cluster_v1.csv` (the same content as `ats_cluster_final.csv`) |
| `Clustering_Analysis/Optimal_Number_of_Clusters/` | `elbow_method.png`: the elbow chart for K = 1 to 10 |
| `Clustering_Analysis/Trained_KMeans_Model/` | `kmeans_model.pkl`: the trained K-Means model (K = 3) |
| `Clustering_Analysis/Cluster_Visualisation_and_Labels/` | The two cluster charts and `cluster_analysis_summary.csv` |

## What the data scripts do

`data_preparation.py` reads `Raw_Data/Dataset_ATS_v2.csv` and does these steps:

1. **Load** the raw file (7,043 rows, 10 columns).
2. **Missing data.** It counts empty cells (there are none). If there were any, numbers would be filled with the median and text with the most common value.
3. **Logic fix.** 260 customers have no phone service but MultipleLines = Yes. These values are changed to No.
4. **Encoding.** Two-value columns become 0 and 1. Contract becomes three 0/1 columns. `encoding_map_final.csv` is the code book.
5. **Split** into training (5,634 rows) and test (1,409 rows). Stratified on Churn, `random_state` 42.
6. **Scale** tenure and MonthlyCharges to 0 to 1 (Min-Max). The scaler learns from the training rows only.
7. **Save** the files and run the quality checks. The script stops with an error if a check fails.

`data_quality_checks.py` is a second script. It only reads the raw file and saves nothing. It prints a profile of each column and the counts: empty cells, outliers, repeated rows, customers with tenure 0 and phone-line values that cannot be true. It also shows the effect of each cleaning choice on a simple model. These are the numbers in the Data Preparation Documents.

## Who made what

- **Data_Preparation:** Md Abdur Rahman and Mir Asif Hossain (Data Engineers).
- **Clustering_Analysis:** Alisha Khadka (Data Analyst). The notebook, report, model file, charts and summary table are her own work, not changed. Her notebook reads `ats_cluster_v1.csv`, which the Data Engineers prepared. It sits next to the notebook and has the same content as `Data_Preparation/Preprocessed_Dataset/ats_cluster_final.csv`.

## Clustering method

- K-Means from scikit-learn on four columns: tenure and MonthlyCharges (scaled 0 to 1), Dependents and SeniorCitizen. Churn is not used to make the groups. It is used afterwards to check each group.
- `KMeans(n_clusters=3, random_state=42, n_init=10)`, trained on all 7,043 customers.
- Number of clusters: elbow method for K = 1 to 10, then silhouette score for K = 3 and K = 4 on a fixed sample of 2,000 customers. K = 3 scores 0.5175 and K = 4 scores 0.4692, so K = 3 was chosen. This is a first choice from the options tested.

## Key results

| Item | Result |
|---|---|
| Preprocessed dataset | 7,043 rows x 12 columns, no empty cells, every column a number |
| Training and test sets | 5,634 and 1,409 rows (80% and 20%), stratified on Churn, churn 26.54% in both |
| Scaling | Min-Max (0 to 1) on tenure and MonthlyCharges, fitted on the training rows only |
| Cluster 0 | 2,023 customers (28.72%). All have dependents. 15.03% left |
| Cluster 1 | 1,138 customers (16.16%). All are senior citizens. 41.83% left |
| Cluster 2 | 3,882 customers (55.12%). No dependents, not senior. 28.05% left |

The clusters are patterns in the data. They do not show that age or dependents make customers leave.

## How to run

```
pip install pandas numpy scikit-learn joblib
python Data_Preparation/Code/data_preparation.py
python Data_Preparation/Code/data_quality_checks.py
```

Run the commands from the main folder of the repository. They need Python 3.9 or newer. The first script rebuilds every data file, the scaler and the tables in a few seconds, with the same content each time. The second script only prints. The Word documents are not changed. Compare what each script prints with its output file in `Data_Preparation/Code/`.

To see the clustering, open `Clustering_Analysis/Code/ATS_Customer_Churn_Clustering.ipynb` in Jupyter or Colab and choose *Run all*. It reads `ats_cluster_v1.csv`, which sits in the same folder. The notebook also needs matplotlib (Colab has it already). It saves its two charts, its summary table and its model file next to itself.

## Notes

- Every random step has a fixed random state, so the results repeat exactly. It is 42 everywhere, except for the cross-validation folds in `data_quality_checks.py`, which use 0.
- The saved model files (`kmeans_model.pkl` and `scaler_final.joblib`) were made with scikit-learn 1.6.1. A newer version loads them with a warning. Run the notebook or the data script again to rebuild them with your version.
- The `.pkl` and `.joblib` files are saved Python objects. Open them only from a source you trust. These ones were made by the team.
- The notebook shows the elbow curve on screen (cell 1) but does not save it. `elbow_method.png` is the saved picture. Its vertical axis is called WCSS. The notebook calls the same number inertia.
- The two data scripts and the clustering notebook were tested with two sets of library versions (pandas 2.2.3, numpy 2.1.3, scikit-learn 1.6.1 and pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1). The data files were identical, both scripts printed the same text, and the clustering numbers were the same.
- The video is uploaded separately to the course site.
