# ==================================================
# XGBOOST WITH KMEANS CLUSTER AS FEATURE
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, classification_report

from xgboost import XGBClassifier


# ==================================================
# STEP 1: LOAD DATA
# ==================================================

df = pd.read_csv("online_shoppers_intention.csv")
print("RAW DATA SHAPE:", df.shape)

df["Revenue"] = df["Revenue"].astype(int)


# ==================================================
# STEP 2: ONE-HOT ENCODING
# ==================================================

df_encoded = pd.get_dummies(df, drop_first=True)
print("ENCODED DATA SHAPE:", df_encoded.shape)

X = df_encoded.drop("Revenue", axis=1)
y = df_encoded["Revenue"]


# ==================================================
# STEP 3: SCALE FEATURES
# ==================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==================================================
# STEP 4: PCA (FOR CLUSTERING)
# ==================================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", pca.explained_variance_ratio_.sum())


# ==================================================
# STEP 5: KMEANS CLUSTERING
# ==================================================

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

df_encoded["Cluster"] = clusters

print("\nCLUSTER DISTRIBUTION:")
print(df_encoded["Cluster"].value_counts())


# ==================================================
# STEP 6: TRAIN / TEST SPLIT
# ==================================================

X_final = df_encoded.drop("Revenue", axis=1)
y_final = df_encoded["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final,
    test_size=0.25,
    random_state=42,
    stratify=y_final
)


# ==================================================
# STEP 7: TRAIN XGBOOST (WITH CLUSTER FEATURE)
# ==================================================

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)


# ==================================================
# STEP 8: EVALUATION
# ==================================================

y_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))
