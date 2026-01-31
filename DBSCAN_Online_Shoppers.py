# ==================================================
# DBSCAN CLUSTERING - ONLINE SHOPPERS
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# --------------------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------------------

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1)

print("DATA SHAPE:", X.shape)

# --------------------------------------------------
# STEP 2: SCALE DATA (VERY IMPORTANT FOR DBSCAN)
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# STEP 3: PCA (FOR VISUALIZATION)
# --------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --------------------------------------------------
# STEP 4: APPLY DBSCAN
# --------------------------------------------------

dbscan = DBSCAN(
    eps=0.5,
    min_samples=10
)

clusters = dbscan.fit_predict(X_pca)

# --------------------------------------------------
# STEP 5: CLUSTER DISTRIBUTION
# --------------------------------------------------

cluster_counts = pd.Series(clusters).value_counts()
print("\nCLUSTER DISTRIBUTION:")
print(cluster_counts)

# --------------------------------------------------
# STEP 6: VISUALIZE CLUSTERS
# --------------------------------------------------

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="tab10",
    s=10
)

plt.title("DBSCAN Clustering (Noise = -1)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
