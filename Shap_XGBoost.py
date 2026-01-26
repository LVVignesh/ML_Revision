# ==================================================
# SHAP EXPLAINABILITY WITH XGBOOST
# ==================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier


# ==================================================
# STEP 1: LOAD DATA
# ==================================================

df = pd.read_csv("online_shoppers_intention.csv")

print("RAW DATA SHAPE:", df.shape)


# ==================================================
# STEP 2: TARGET ENCODING
# ==================================================

df["Revenue"] = df["Revenue"].astype(int)


# ==================================================
# STEP 3: ONE-HOT ENCODING (CATEGORICAL)
# ==================================================

df = pd.get_dummies(df, drop_first=True)


# ==================================================
# STEP 4: SPLIT FEATURES / TARGET
# ==================================================

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==================================================
# STEP 5: TRAIN XGBOOST MODEL
# ==================================================

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)


# ==================================================
# STEP 6: MODEL PERFORMANCE CHECK
# ==================================================

y_prob = xgb_model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# ==================================================
# STEP 7: SHAP EXPLAINER
# ==================================================

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)


# ==================================================
# STEP 8: SHAP SUMMARY PLOT (GLOBAL IMPORTANCE)
# ==================================================

shap.summary_plot(shap_values, X_test)


# ==================================================
# STEP 9: SHAP BAR PLOT (AVERAGE IMPACT)
# ==================================================

shap.summary_plot(shap_values, X_test, plot_type="bar")


# ==================================================
# STEP 10: SHAP DEPENDENCE PLOT (FEATURE EFFECT)
# ==================================================

shap.dependence_plot("PageValues", shap_values, X_test)


# ==================================================
# STEP 11: SHAP FORCE PLOT (SINGLE PREDICTION)
# ==================================================

sample_index = 0

shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index],
    X_test.iloc[sample_index],
    matplotlib=True
)
