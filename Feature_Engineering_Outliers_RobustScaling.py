# ==================================================
# FEATURE ENGINEERING: OUTLIER CAPPING + ROBUST SCALING
# Online Shoppers Intention Dataset
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


# ==================================================
# STEP 1: LOAD DATA
# ==================================================

df = pd.read_csv("online_shoppers_intention.csv")
print("RAW DATA SHAPE:", df.shape)

df["Revenue"] = df["Revenue"].astype(int)


# ==================================================
# STEP 2: ONE-HOT ENCODING
# ==================================================

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1)
y = df["Revenue"]


# ==================================================
# STEP 3: TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==================================================
# STEP 4: OUTLIER CAPPING (PERCENTILE-BASED)
# ==================================================
# We cap extreme values instead of removing rows

def cap_outliers(df, lower=0.01, upper=0.99):
    capped_df = df.copy()

    for col in capped_df.columns:
        # Apply only to continuous numerical columns
        if capped_df[col].dtype in ["int64", "float64"] and capped_df[col].nunique() > 10:
            lower_bound = capped_df[col].quantile(lower)
            upper_bound = capped_df[col].quantile(upper)
            capped_df[col] = capped_df[col].clip(lower_bound, upper_bound)

    return capped_df


X_train_capped = cap_outliers(X_train)
X_test_capped = cap_outliers(X_test)


# ==================================================
# STEP 5: ROBUST SCALING
# ==================================================
# Uses median & IQR → resistant to outliers

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train_capped)
X_test_scaled = scaler.transform(X_test_capped)


# ==================================================
# STEP 6: LOGISTIC REGRESSION (CLEAR EFFECT DEMO)
# ==================================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# ==================================================
# STEP 7: EVALUATION
# ==================================================

y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

print("\nROC-AUC SCORE:", roc_auc_score(y_test, y_prob))
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))
