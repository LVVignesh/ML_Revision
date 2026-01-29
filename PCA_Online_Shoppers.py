# ==================================================
# PCA (PRINCIPAL COMPONENT ANALYSIS)
# Unsupervised Learning - Online Shoppers Dataset
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==================================================
# STEP 1: LOAD DATA (NO TARGET)
# ==================================================

df = pd.read_csv("online_shoppers_intention.csv")

# Drop target for PCA (unsupervised!)
df_features = df.drop("Revenue", axis=1)

# One-hot encode categorical features
df_features = pd.get_dummies(df_features, drop_first=True)

print("FEATURE SHAPE BEFORE PCA:", df_features.shape)


# ==================================================
# STEP 2: SCALE DATA (MANDATORY FOR PCA)
# ==================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)


# ==================================================
# STEP 3: APPLY PCA (2 COMPONENTS FOR VISUALIZATION)
# ==================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("Total Variance Explained:", np.sum(pca.explained_variance_ratio_))


# ==================================================
# STEP 4: VISUALIZE PCA
# ==================================================

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Online Shoppers Data")
plt.show()
