# ==================================================
# STEP 0: IMPORT LIBRARIES
# ==================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import matplotlib.pyplot as plt


# ==================================================
# STEP 1: LOAD DATA
# ==================================================

df = pd.read_csv("online_shoppers_intention.csv")

# Convert target to 0/1
df["Revenue"] = df["Revenue"].astype(int)

X = df.drop("Revenue", axis=1)
y = df["Revenue"]


# ==================================================
# STEP 2: HANDLE CATEGORICAL FEATURES
# ==================================================
# Decision Trees DO NOT need scaling

X = pd.get_dummies(X, drop_first=True)


# ==================================================
# STEP 3: TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==================================================
# STEP 4: TRAIN DECISION TREE (INTENTIONALLY UNRESTRICTED)
# ==================================================

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)


# ==================================================
# STEP 5: EVALUATION
# ==================================================

y_pred = tree.predict(X_test)
y_prob = tree.predict_proba(X_test)[:, 1]

print("\nACCURACY:", accuracy_score(y_test, y_pred))
print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\nROC-AUC SCORE:", roc_auc_score(y_test, y_prob))


# ==================================================
# STEP 6: VISUALIZE TREE (FIRST LEVELS ONLY)
# ==================================================

plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["No Purchase", "Purchase"],
    max_depth=3,
    filled=True
)
plt.title("Decision Tree (Top 3 Levels)")
plt.show()
