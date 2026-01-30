# =========================================================
# ONLINE SHOPPERS ML PROJECT - END TO END PIPELINE
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,roc_curve, auc,
    precision_recall_curve    
)

from xgboost import XGBClassifier

# =========================================================
# STEP 1: LOAD DATA
# =========================================================

df = pd.read_csv("online_shoppers_intention.csv")

print("RAW DATA SHAPE:", df.shape)

# Convert target to numeric
df["Revenue"] = df["Revenue"].astype(int)


# =========================================================
# STEP 2: ONE-HOT ENCODING
# =========================================================
# ML models cannot understand text

df = pd.get_dummies(df, drop_first=True)


# =========================================================
# STEP 3: TRAIN / TEST SPLIT
# =========================================================

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# =========================================================
# STEP 4: FEATURE SCALING
# =========================================================
# RobustScaler handles outliers better than StandardScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# STEP 5: UNSUPERVISED LEARNING (PCA + KMEANS)
# =========================================================

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("PCA Variance Explained:", pca.explained_variance_ratio_)

kmeans = KMeans(n_clusters=3, random_state=42)
train_clusters = kmeans.fit_predict(X_train_pca)
test_clusters = kmeans.predict(X_test_pca)

# Add clusters as a feature
X_train_hybrid = np.column_stack([X_train_scaled, train_clusters])
X_test_hybrid = np.column_stack([X_test_scaled, test_clusters])


# =========================================================
# STEP 6: SUPERVISED MODEL (XGBOOST)
# =========================================================

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_hybrid, y_train)


# =========================================================
# STEP 7: EVALUATION
# =========================================================

y_pred = model.predict(X_test_hybrid)
y_prob = model.predict_proba(X_test_hybrid)[:, 1]

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("ROC-AUC SCORE:", roc_auc_score(y_test, y_prob))


# =========================================================
# STEP 8: CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =========================================================
# STEP 9: ROC CURVE
# =========================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================================================
# STEP 10: PRECISION-RECALL CURVE       
# =========================================================

precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


# =========================================================
# STEP 11: SHAP EXPLAINABILITY
# =========================================================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_hybrid)

shap.summary_plot(shap_values, X_test_hybrid)
