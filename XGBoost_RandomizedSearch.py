# ==================================================
# RANDOMIZED SEARCH + XGBOOST TUNING
# ==================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import RobustScaler

from xgboost import XGBClassifier


# ==================================================
# LOAD DATA
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
# SCALING (OPTIONAL FOR XGB, KEPT FOR CONSISTENCY)
# ==================================================

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==================================================
# PARAMETER DISTRIBUTION
# ==================================================

param_dist = {
    "n_estimators": [200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 6],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3]
}


# ==================================================
# RANDOMIZED SEARCH
# ==================================================

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,          # number of random configs
    scoring="roc_auc",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)


# ==================================================
# BEST MODEL EVALUATION
# ==================================================

best_xgb = random_search.best_estimator_

print("\nBEST PARAMETERS:")
print(random_search.best_params_)

y_prob = best_xgb.predict_proba(X_test)[:, 1]
y_pred = best_xgb.predict(X_test)

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))
