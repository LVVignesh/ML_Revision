# ==================================================
# STEP 0: IMPORT LIBRARIES
# ==================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt


# ==================================================
# STEP 1: LOAD DATA
# ==================================================

df = pd.read_csv("online_shoppers_intention.csv")

print("\nRAW DATA SHAPE:", df.shape)
print(df.head())


# ==================================================
# STEP 2: TARGET VARIABLE
# ==================================================
# Revenue is already boolean → convert to 0/1

df["Revenue"] = df["Revenue"].astype(int)

print("\nTARGET DISTRIBUTION")
print(df["Revenue"].value_counts())


# ==================================================
# STEP 3: FEATURE / TARGET SPLIT
# ==================================================

X = df.drop("Revenue", axis=1)
y = df["Revenue"]


# ==================================================
# STEP 4: TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTRAIN SIZE:", X_train.shape)
print("TEST SIZE:", X_test.shape)


# ==================================================
# STEP 5: HANDLE CATEGORICAL FEATURES
# ==================================================
# Month, VisitorType, Weekend

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# ==================================================
# STEP 6: FEATURE SCALING
# ==================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==================================================
# STEP 7: TRAIN LOGISTIC REGRESSION
# ==================================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ==================================================
# STEP 8: PREDICTION
# ==================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ==================================================
# STEP 9: EVALUATION
# ==================================================

print("\nACCURACY:", accuracy_score(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\nROC-AUC SCORE:", roc_auc_score(y_test, y_prob))


# ==================================================
# STEP 10: ROC CURVE PLOT
# ==================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Online Shoppers")
plt.legend()
plt.show()
