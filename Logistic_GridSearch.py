# ==================================================
# HYPERPARAMETER TUNING - GRID SEARCH (LOGISTIC)
# ==================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


# ==================================================
# LOAD & PREPARE DATA
# ==================================================

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


# ==================================================
# ROBUST SCALING
# ==================================================

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==================================================
# GRID SEARCH
# ==================================================

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "class_weight": [None, "balanced"]
}

model = LogisticRegression(max_iter=1000)

grid = GridSearchCV(
    model,
    param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)


# ==================================================
# RESULTS
# ==================================================

print("BEST PARAMETERS:")
print(grid.best_params_)

best_model = grid.best_estimator_

y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))
