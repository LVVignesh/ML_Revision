# ==================================================
# KMEANS CLUSTERING ON PCA DATA
# Online Shoppers Intention Dataset
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ==================================================
# STEP 1: LOAD DATA
# ==================================================

df = pd.read_csv("online_shoppers_intention.csv")
print("RAW DATA SHAPE:", df.shape)


# ==================================================
# STEP 2: TARGET ENCODING (ONLY FOR ANALYSIS LATER)
# ==================================================

df["Revenue"] = df["Revenue"].astype(int)


# ==================================================
# STEP 3: ONE-HOT ENCODING (CATEGORICAL FEATURES)
# ==================================================

df_encoded = pd.get_dummies(df, drop_first=True)
print("ENCODED DATA SHAPE:", df_encoded.shape)


# ==================================================
# STEP 4: REMOVE TARGET FOR UNSUPERVISED LEARNING
# ==================================================

X = df_encoded.drop("Revenue", axis=1)


# ==================================================
# STEP 5: STANDARD SCALING
# ==================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==================================================
# STEP 6: PCA (2 COMPONENTS)
# ==================================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", pca.explained_variance_ratio_.sum())


# ==================================================
# STEP 7: ELBOW METHOD TO FIND OPTIMAL K
# ==================================================

inertia = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()


# ==================================================
# STEP 8: TRAIN KMEANS (CHOOSE K = 3)
# ==================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_pca)


# ==================================================
# STEP 9: VISUALIZE CLUSTERS IN PCA SPACE
# ==================================================

plt.figure(figsize=(7, 5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KMeans Clustering on PCA Data")
plt.colorbar(label="Cluster")
plt.show()


# ==================================================
# STEP 10: ADD CLUSTERS BACK TO ORIGINAL DATA
# ==================================================

df_clusters = df.copy()
df_clusters["Cluster"] = clusters


# ==================================================
# STEP 11: CLUSTER INTERPRETATION
# ==================================================

print("\nCLUSTER DISTRIBUTION:")
print(df_clusters["Cluster"].value_counts())

print("\nMEAN FEATURE VALUES PER CLUSTER:")
print(df_clusters.groupby("Cluster").mean(numeric_only=True))


# ==================================================
# STEP 12: REVENUE RATE PER CLUSTER (INSIGHT)
# ==================================================

print("\nREVENUE RATE PER CLUSTER:")
print(df_clusters.groupby("Cluster")["Revenue"].mean())
