# ==================================================
# STEP 0: IMPORT LIBRARIES
# ==================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
# Revenue = True → 1 (Purchase)
# Revenue = False → 0 (No Purchase)

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
# STEP 5: ENCODE CATEGORICAL FEATURES
# ==================================================

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns (important!)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# ==================================================
# STEP 6: FEATURE SCALING (OPTIONAL FOR RF)
# ==================================================
# Random Forest does NOT require scaling
# But we keep it for consistency with other models



# ==================================================
# STEP 7: TRAIN RANDOM FOREST MODEL
# ==================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)


# ==================================================
# STEP 8: PREDICTION
# ==================================================

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]


# ==================================================
# STEP 9: EVALUATION
# ==================================================

print("\nACCURACY:", accuracy_score(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC SCORE:", roc_auc)


# ==================================================
# STEP 10: ROC CURVE
# ==================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.show()


# ==================================================
# STEP 11: FEATURE IMPORTANCE (TOP 10)
# ==================================================

importances = rf_model.feature_importances_
feature_names = X_train.columns


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance_df.head(10))
