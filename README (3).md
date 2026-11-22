# Loan Approval Prediction using Support Vector Machine (SVM)

A machine learning project that uses a **Support Vector Machine (SVM)** to predict whether a loan application will be **approved or rejected**, based on the applicant's financial and employment details.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Data Leakage Note](#data-leakage-note)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Author](#author)

## Project Overview

Banks must decide quickly and consistently whether to approve a loan. This project trains a binary classifier that learns from past applications and predicts the outcome for new applicants.

- **Problem type:** Binary classification
- **Algorithm:** Support Vector Machine (SVM) with hyperparameter tuning
- **Target:** `loan_approved` (1 = Approved, 0 = Rejected)

## Dataset

The dataset (`loan_approval.csv`) contains **2,000 applicants** and **8 columns**. There are no missing values and no duplicate rows.

| Column | Description | Used as |
|---|---|---|
| `name` | Applicant name | Dropped (identifier) |
| `city` | Applicant city | Dropped (identifier) |
| `income` | Annual income | Feature |
| `credit_score` | Credit score (300 to 850) | Feature |
| `loan_amount` | Requested loan amount | Feature |
| `years_employed` | Years of employment | Feature |
| `points` | Internal score (10 to 100) | Excluded (see [Data Leakage Note](#data-leakage-note)) |
| `loan_approved` | Loan decision | Target |

**Class balance:** 56% Rejected and 44% Approved, so the data is reasonably balanced.

## Methodology

1. **Data loading:** read the CSV file with pandas.
2. **Exploratory data analysis:** check shape, data types, missing values, summary statistics, class balance and correlations.
3. **Data cleaning:** drop `name` and `city` (identifiers that carry no predictive information) and convert the target from True/False to 1/0.
4. **Visualization:** class distribution, credit score boxplot and a correlation heatmap.
5. **Train/test split:** 80% training and 20% testing, stratified so both sets keep the same class ratio.
6. **Pipeline:** `StandardScaler` followed by `SVC`. Scaling is required because SVM is distance-based and the features have very different ranges.
7. **Hyperparameter tuning:** `GridSearchCV` with 5-fold stratified cross-validation over:
   - `kernel`: linear, rbf
   - `C`: 0.1, 1, 10, 100
   - `gamma`: scale, 0.1, 0.01, 0.001 (rbf only)
8. **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix and ROC curve on the unseen test set.

## Results

Best parameters: `kernel = rbf`, `C = 100`, `gamma = scale`

| Metric | Score |
|---|---|
| Cross-validation accuracy | 95.7% |
| Test accuracy | 97.0% |
| Precision | 96.6% |
| Recall | 96.6% |
| F1-score | 96.6% |
| ROC-AUC | 0.996 |

![EDA plots](eda_plots.png)

![SVM results](svm_results.png)

The model correctly classifies about 97 out of every 100 applicants in the test set. `credit_score` is the strongest real predictor (correlation of about 0.72 with the target).

## Data Leakage Note

The `points` column was **not** used for training. In this dataset, every applicant with `points >= 60` was approved and every applicant below 60 was rejected, so the column is effectively the approval rule itself.

| Setup | Test accuracy |
|---|---|
| Without `points` (used in this project) | 97.0% |
| With `points` | 100% (trivial, due to leakage) |

Including it would give a perfect score that says nothing about how well the model learns from real applicant information. To try it anyway, set `USE_POINTS = True` in `svm_loan_approval.py`.

## Project Structure

```
svm-loan-approval/
├── svm_loan_approval.py   # Complete SVM code
├── loan_approval.csv      # Dataset
├── eda_plots.png          # EDA graphs (generated)
├── svm_results.png        # Confusion matrix and ROC curve (generated)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/svm-loan-approval.git
   cd svm-loan-approval
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python svm_loan_approval.py
   ```

The script prints the metrics in the terminal and saves the plots as `eda_plots.png` and `svm_results.png`.

## Technologies Used

- Python 3
- pandas and NumPy
- scikit-learn
- Matplotlib

## Author

**Your Name**
Roll No: your-roll-number
Course: your-course-name
