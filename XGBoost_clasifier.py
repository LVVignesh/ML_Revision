# ==================================================
# STEP 0: IMPORT LIBRARIES
# ==================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# ==================================================
# STEP 1: LOAD DATA
# ==================================================

df = pd.read_csv("online_shoppers_intention.csv")

print("\nRAW DATA SHAPE:", df.shape)
print(df.head())


# ==================================================
# STEP 2: TARGET VARIABLE
# ==================================================

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


# ==================================================
# STEP 5: ONE-HOT ENCODING
# ==================================================

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# ==================================================
# STEP 6: TRAIN XGBOOST MODEL
# ==================================================

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)


# ==================================================
# STEP 7: PREDICTION
# ==================================================

y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]


# ==================================================
# STEP 8: EVALUATION
# ==================================================

print("\nACCURACY:", accuracy_score(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC SCORE:", roc_auc)


# ==================================================
# STEP 9: ROC CURVE
# ==================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.show()
