# =========================================================
# ONLINE SHOPPERS ML PROJECT 
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score
)
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve
)

from xgboost import XGBClassifier


# =========================================================
# STEP 1: LOAD DATA
# =========================================================

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)

print("RAW DATA SHAPE:", df.shape)


# =========================================================
# STEP 2: ONE-HOT ENCODING
# =========================================================

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1)
y = df["Revenue"]


# =========================================================
# STEP 3: TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)


# =========================================================
# STEP 4: ROBUST SCALING
# =========================================================

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

X_train_hybrid = np.column_stack([X_train_scaled, train_clusters])
X_test_hybrid = np.column_stack([X_test_scaled, test_clusters])


# =========================================================
# STEP 6: HYPERPARAMETER TUNING (XGBOOST ONLY)
# =========================================================

param_dist = {
    "n_estimators": [150, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

random_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_hybrid, y_train)

best_model = random_search.best_estimator_

print("\nBEST PARAMETERS:")
print(random_search.best_params_)


# =========================================================
# STEP 7: CROSS-VALIDATION (STABILITY CHECK)
# =========================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    best_model,
    X_train_hybrid,
    y_train,
    cv=skf,
    scoring="roc_auc"
)

print("\nCROSS-VALIDATION ROC-AUC SCORES:", cv_scores)
print("MEAN ROC-AUC:", cv_scores.mean())
print("STD DEV:", cv_scores.std())


# =========================================================
# STEP 8: FINAL EVALUATION
# =========================================================

y_pred = best_model.predict(X_test_hybrid)
y_prob = best_model.predict_proba(X_test_hybrid)[:, 1]

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("TEST ROC-AUC:", roc_auc_score(y_test, y_prob))


# =========================================================
# STEP 9: VISUALIZATIONS
# =========================================================

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


# =========================================================
# STEP 10: SHAP EXPLAINABILITY
# =========================================================

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_hybrid)

shap.summary_plot(shap_values, X_test_hybrid)
