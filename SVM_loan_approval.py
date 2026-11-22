# =====================================================
#  SVM Model on Loan Approval Dataset
# =====================================================
# Install (agar zaroorat ho):  pip install pandas numpy matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)

# -----------------------------------------------------
# SETTING: 'points' column use karni hai ya nahi?
# Data mein dekha gaya: points >= 60  <=>  loan approved (100% match).
# Yani 'points' khud approval ka rule hai (answer leak). Isliye model ko asal
# kaam (income, credit score, etc. se faisla seekhna) karwane ke liye False rakhein.
# True karne se accuracy 100% aa jati hai, lekin wo model ki kamyabi nahi.
# -----------------------------------------------------
USE_POINTS = False

# =====================================================
# STEP 1: Data load karna
# =====================================================
df = pd.read_csv("loan_approval.csv")   # apni file ka path

# =====================================================
# STEP 2: Data ko samajhna (EDA - Exploratory Data Analysis)
# =====================================================
print("Shape (rows, columns):", df.shape)
print("\nColumns aur unke data types:\n", df.dtypes)
print("\nPehli 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nStatistical summary:\n", df.describe().round(2))
print("\nTarget (loan_approved) ka ratio:\n",
      df["loan_approved"].value_counts(normalize=True).round(3))

# =====================================================
# STEP 3: Data cleaning / preparation
# =====================================================
# 'name' aur 'city' sirf pehchan (identifier) hain - har banday ka alag naam,
# aur 2000 mein se ~1882 alag cities. Inse model kuch seekh nahi sakta, isliye hata dein.
df = df.drop(columns=["name", "city"])

# Target ko True/False se 1/0 mein badalna (1 = Approved, 0 = Rejected)
df["loan_approved"] = df["loan_approved"].astype(int)

# =====================================================
# STEP 4: Visualization (graphs)
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# (a) Class balance
df["loan_approved"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color=["tomato", "seagreen"])
axes[0].set_xticklabels(["Rejected (0)", "Approved (1)"], rotation=0)
axes[0].set_title("Loan Approved - Class Balance")
axes[0].set_ylabel("Number of applicants")

# (b) Credit score: approved vs rejected
df.boxplot(column="credit_score", by="loan_approved", ax=axes[1])
axes[1].set_title("Credit Score vs Loan Approved")
axes[1].set_xlabel("loan_approved (0 = No, 1 = Yes)")
plt.sca(axes[1]); plt.suptitle("")

# (c) Correlation heatmap
corr = df.corr()
im = axes[2].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[2].set_xticks(range(len(corr))); axes[2].set_xticklabels(corr.columns, rotation=45, ha="right")
axes[2].set_yticks(range(len(corr))); axes[2].set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        axes[2].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
axes[2].set_title("Correlation Matrix")
plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.show()

print("\nTarget ke saath correlation:\n",
      corr["loan_approved"].drop("loan_approved").sort_values(ascending=False).round(3))

# =====================================================
# STEP 5: Features (X) aur Target (y) alag karna
# =====================================================
features = ["income", "credit_score", "loan_amount", "years_employed"]
if USE_POINTS:
    features.append("points")

X = df[features]
y = df["loan_approved"]
print("\nUse hone wale features:", features)

# =====================================================
# STEP 6: Train / Test split (80% / 20%)
# =====================================================
# stratify=y  -> train aur test dono mein Approved/Rejected ka ratio same rahe
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

# =====================================================
# STEP 7: Pipeline = Scaling + SVM
# =====================================================
# SVM distance par kaam karta hai. income (30,000-150,000) aur credit_score (300-850)
# ki scale bohat alag hai, isliye StandardScaler zaroori hai.
# Pipeline ka faida: scaler sirf train data se seekhta hai (data leakage nahi hota).
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC()),
])

# =====================================================
# STEP 8: Hyperparameter tuning (GridSearchCV)
# =====================================================
# kernel : linear = seedhi line se alag karna, rbf = curved boundary
# C      : chhota C = simple model (zyada galtiyan manzoor), bada C = sakht model (overfit ka khatra)
# gamma  : (sirf rbf) ek training point ka asar kitni door tak jaye
param_grid = [
    {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
    {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10, 100],
     "svm__gamma": ["scale", 0.1, 0.01, 0.001]},
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
gs.fit(X_train, y_train)

print("\nBest parameters:", gs.best_params_)
print("Best CV accuracy: %.4f" % gs.best_score_)

# =====================================================
# STEP 9: Test data par evaluation
# =====================================================
model = gs.best_estimator_
y_pred = model.predict(X_test)
y_score = model.decision_function(X_test)   # ROC-AUC ke liye

print("\n----- TEST RESULTS -----")
print("Accuracy : %.4f" % accuracy_score(y_test, y_pred))
print("Precision: %.4f" % precision_score(y_test, y_pred))
print("Recall   : %.4f" % recall_score(y_test, y_pred))
print("F1-score : %.4f" % f1_score(y_test, y_pred))
print("ROC-AUC  : %.4f" % roc_auc_score(y_test, y_score))
print("\nClassification report:\n",
      classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# Confusion matrix + ROC curve
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Rejected", "Approved"], cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix")
RocCurveDisplay.from_predictions(y_test, y_score, ax=axes[1], name="SVM")
axes[1].plot([0, 1], [0, 1], "k--", label="Random guess")
axes[1].set_title("ROC Curve")
axes[1].legend()
plt.tight_layout()
plt.savefig("svm_results.png", dpi=150)
plt.show()

# =====================================================
# STEP 10: Nayi application par prediction
# =====================================================
new_applicant = pd.DataFrame([{
    "income": 85000, "credit_score": 720, "loan_amount": 20000,
    "years_employed": 10, "points": 70,
}])[features]

pred = model.predict(new_applicant)[0]
print("\nNew applicant ka result:", "APPROVED" if pred == 1 else "REJECTED")

# =====================================================
# STEP 11 (Bonus): 'points' ka asar dekhna (leakage check)
# =====================================================
# Dono versions alag-alag train karke compare karte hain.
def run(feats):
    Xtr, Xte = df.loc[X_train.index, feats], df.loc[X_test.index, feats]
    g = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1).fit(Xtr, y_train)
    return accuracy_score(y_test, g.best_estimator_.predict(Xte)), g.best_params_

base_feats = ["income", "credit_score", "loan_amount", "years_employed"]
acc_without, bp_without = run(base_feats)
acc_with, bp_with = run(base_feats + ["points"])
print("\nWITHOUT points -> Test accuracy: %.4f | %s" % (acc_without, bp_without))
print("WITH points    -> Test accuracy: %.4f | %s" % (acc_with, bp_with))