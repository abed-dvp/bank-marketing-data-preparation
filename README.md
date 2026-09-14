# Bank Marketing — Data Preparation

## Business Problem

Predict whether a bank customer will subscribe to a term deposit (`y = yes / no`) **before making the marketing call**, enabling efficient lead qualification and targeted outbound marketing.

---

## Dataset

- **Source**: [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Observations**: 45,211 rows × 17 original columns
- **Target Variable**: `y` (binary classification: `yes` / `no`)
- **Class Distribution**: Skewed target with **88.30% non-subscribers** (`no`) and **11.70% subscribers** (`yes`) in the training data

---

## Why This Project

The focus of this portfolio project is **rigorous Data Preparation and pipeline design**, not model complexity or hyperparameter tuning.

In real-world data science, model performance is fundamentally bounded by the quality, realism, and integrity of the preprocessing pipeline. This project demonstrates how to address common real-world data challenges—such as missing values masked as strings, heavy outliers, target imbalance, feature engineering, and insidious data leakage—while maintaining strict data isolation between training and evaluation partitions.

---

## Data Preparation Workflow

```
Raw Data
   ↓
Data Quality Audit (deduplication check, missingness scan, sentinel value identification)
   ↓
Data Leakage Detection (removal of call 'duration' from pre-call prediction matrix)
   ↓
Stratified Train/Test Split (80% train / 20% test, preserving 88.3% / 11.7% class ratio)
   ↓
Missing / Unknown Handling (SimpleImputer for job/education; preserved unknown categories for contact/poutcome)
   ↓
Outlier Inspection & Distribution Analysis (IQR detection on skewed financial variables)
   ↓
Feature Scaling (StandardScaler for Gaussian-like features; RobustScaler for skewed/outlier-heavy features)
   ↓
Categorical Encoding (OneHotEncoder with drop='if_binary' and handle_unknown='ignore')
   ↓
Feature Engineering & Discretization (decoupled pdays sentinel, aggregated debt exposure, demographic age bins)
   ↓
Class Balancing with SMOTE (resampled training set to 50/50 while keeping test set untouched)
   ↓
Feature Selection (training-only validation split, permutation importance, top 20 selection)
   ↓
Data Leakage Demonstration (empirical proof of artificial metric inflation via post-event features)
   ↓
Evaluation & Business Interpretation
```

---

## Key Data Decisions

1. **`duration` Dropped to Prevent Data Leakage**: The call duration is unknown before contacting the client. Including it would create an invalid model that cannot be deployed prior to placing calls.
2. **Context-Specific Treatment of `"unknown"` Values**:
   - `job` (0.64%) and `education` (4.11%) represent true missing client profile data and were imputed using the training mode via `SimpleImputer(strategy='most_frequent')`.
   - `contact` (28.8%) and `poutcome` (81.8%) reflect explicit institutional statuses (e.g., uncontacted clients) and were retained as informative categorical levels.
3. **Outlier Preservation with `RobustScaler`**: Extreme balances and campaign counts reflect real affluent customers and aggressive outreach rather than data corruption. No rows were discarded; instead, median/IQR scaling bounded their influence.
4. **Leakage-Free Preprocessing**: All scalers, imputers, and encoders were `fit()` exclusively on the training partition and applied to the test set via `transform()`.
5. **Semantic Categorical Encoding**: Avoided `OrdinalEncoder` to prevent imposing arbitrary Euclidean distances on nominal variables; employed `OneHotEncoder` with `drop='if_binary'` and `handle_unknown='ignore'`.
6. **Sentinel Value Decoupling**: The artificial `pdays = -1` sentinel (uncontacted clients) was cleanly decoupled into a binary contact indicator (`was_previously_contacted`) and non-negative elapsed days (`pdays_clean` scaled to $[0, 1]$).
7. **Hypothesis-Driven Feature Engineering**: Synthesized `has_any_loan` (combining housing and personal loans) and demographic age cohorts (`age_group`).

---

## Model Results

All models were evaluated on the **exact same untouched real-world test set** (9,043 rows, 88.30% `"no"` / 11.70% `"yes"`) using `LogisticRegression(max_iter=2000, random_state=42)`:

| Model | Feature Count | Accuracy | Precision (`yes`) | Recall (`yes`) | F1-Score (`yes`) | False Positives | False Negatives |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy Majority Baseline** | 0 | `0.8830` | `0.0000` | `0.0000` | `0.0000` | `0` | `1,058` |
| **Baseline Logistic Regression** | 51 | `0.8922` | `0.6379` | `0.1815` | `0.2826` | `109` | `866` |
| **SMOTE Logistic Regression** | 51 | `0.7594` | `0.2690` | **`0.6153`** | **`0.3744`** | `1,769` | **`407`** |
| **Selected-Feature Logistic Regression** | **20** | **`0.8927`** | **`0.6528`** | `0.1777` | `0.2793` | **`100`** | `870` |

*Note: The positive class is `"yes"` (deposit subscribers).*

---

## Data Leakage Demonstration

To demonstrate why data leakage is misleading in practice, a controlled experiment was run where scaled `duration` was reintroduced into the feature matrix:

| Model Configuration | Feature Count | Accuracy | Precision (`yes`) | Recall (`yes`) | F1-Score (`yes`) | False Positives | False Negatives |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Realistic Baseline** | 51 | `0.8922` | `0.6379` | `0.1815` | `0.2826` | `109` | `866` |
| **Baseline + duration leakage** *(INVALID)* | 52 | `0.9012` | `0.6401` | `0.3563` | `0.4578` | `212` | `681` |

Including `duration` artificially inflates F1-Score by **+62.0%** and nearly doubles recall. However, because call duration is known **only during or after** the phone call, this model is **completely invalid for a pre-call prediction system**. It serves as a reminder that:
> **High offline validation performance $\ne$ valid real-world model.**

---

## Feature Importance

Permutation feature importance was evaluated on an isolated 25% training validation split using minority-class F1-Score (`make_scorer(f1_score, pos_label="yes")`):

1. **`poutcome_success`** ranked #1 overall (`Importance = +0.1511`), serving as the single strongest predictor of re-subscription.
2. **`has_any_loan`** (engineered loan indicator) ranked **#4 overall** (`+0.0323`), outperforming the individual raw loan attributes.
3. Demographic cohorts **`age_group_60+`** (#6) and **`age_group_31-45`** (#11) contributed strong non-linear demographic signal.
4. **`was_previously_contacted`** ranked #15 (`+0.0079`).
5. **Model Parsimony**: Selecting the top 20 features (a **60.8% reduction in feature space**) preserved **98.8% of baseline F1-Score** while improving precision from 63.79% to 65.28% and reducing false alarms.

*(Note: Permutation importance measures empirical model reliance on this specific split and architecture; it does not claim physical causality.)*

---

## Key Learning

> **Good machine learning performance depends not only on the algorithm, but on whether the data preparation process is realistic, leakage-free, reproducible, and aligned with the prediction moment.**

---

## Limitations

- **Single Model Family**: Evaluated strictly with transparent `LogisticRegression` to isolate data preparation effects without algorithmic confounders.
- **No Hyperparameter Tuning**: Default regularization parameters (`C=1.0`) were retained to focus on data changes.
- **Fixed Decision Threshold**: Default 0.5 classification threshold was used throughout; threshold tuning would be necessary in a real deployment to optimize specific business utility functions.
- **SMOTE on One-Hot Features**: Standard SMOTE was used pedagogically; in mixed-type production pipelines, `SMOTENC` should be employed to preserve discrete dummy levels.
- **No Explicit Cost Matrix**: Exact institutional dollar costs for False Negatives vs. False Positives were unavailable, so models are presented as operational trade-offs rather than a single definitive winner.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/abed-dvp/bank-marketing-data-preparation.git
cd bank-marketing-data-preparation

# Install required dependencies
pip install -r requirements.txt

# Run or view the complete executed notebook
jupyter notebook notebook.ipynb
```
