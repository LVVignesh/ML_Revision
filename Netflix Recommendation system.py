# =========================================================
# NETFLIX RECOMMENDATION SYSTEM
# Collaborative Filtering + SVD + Evaluation
# =========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# STEP 1: LOAD DATA
# =========================================================

df = pd.read_csv("netflix.csv")
print("RAW DATA SHAPE:", df.shape)

# FIX 1: Use "MOVIE" (All caps) to match your dataset
df = df[df["type"] == "MOVIE"].reset_index(drop=True)

# FIX 2: Use "runtime" instead of "duration"
# FIX 3: Removed .str.replace() because your runtime is already numeric
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
df["runtime"] = df["runtime"].fillna(df["runtime"].median())

# =========================================================
# STEP 2: FEATURE ENGINEERING (IMPLICIT FEEDBACK)
# =========================================================

# Using runtime for popularity calculation
df["popularity"] = (
    df["release_year"].rank(pct=True) +
    df["runtime"].rank(pct=True)
)

scaler = MinMaxScaler()
df["popularity"] = scaler.fit_transform(df[["popularity"]])

# =========================================================
# STEP 3: SIMULATE USER INTERACTIONS
# =========================================================

NUM_USERS = 100
np.random.seed(42)

users = [f"user_{i}" for i in range(NUM_USERS)]
interaction_data = []

for user in users:
    watched = df.sample(120)
    for _, row in watched.iterrows():
        interaction_data.append([
            user,
            row["title"],
            row["popularity"] + np.random.normal(0, 0.05)
        ])

interactions = pd.DataFrame(
    interaction_data,
    columns=["user", "title", "rating"]
)

interactions["rating"] = interactions["rating"].clip(0, 1)
print("INTERACTIONS SHAPE:", interactions.shape)

# =========================================================
# STEP 4: USER–ITEM MATRIX
# =========================================================

user_item = interactions.pivot_table(
    index="user",
    columns="title",
    values="rating",
    fill_value=0
)

print("USER–ITEM MATRIX SHAPE:", user_item.shape)

# =========================================================
# STEP 5: MATRIX FACTORIZATION (SVD)
# =========================================================

svd = TruncatedSVD(n_components=30, random_state=42)
user_latent = svd.fit_transform(user_item)
movie_latent = svd.components_.T

print("VARIANCE EXPLAINED:", svd.explained_variance_ratio_.sum())

# =========================================================
# STEP 6: SIMILARITY MATRIX
# =========================================================

similarity_matrix = cosine_similarity(movie_latent)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_item.columns,
    columns=user_item.columns
)

# =========================================================
# STEP 7: RECOMMENDATION FUNCTION
# =========================================================

def recommend_movies(movie_title, top_k=10):
    if movie_title not in similarity_df:
        return []

    return (
        similarity_df[movie_title]
        .sort_values(ascending=False)
        .iloc[1:top_k+1]
        .index
        .tolist()
    )

# =========================================================
# STEP 8: EVALUATION METRICS
# =========================================================

def precision_recall_at_k(user_item_matrix, similarity_df, k=10):
    precisions = []
    recalls = []

    for user in user_item_matrix.index:
        watched = user_item_matrix.loc[user]
        relevant_items = watched[watched > 0].index.tolist()

        if len(relevant_items) < 2:
            continue

        seed_item = relevant_items[0]
        recommended = recommend_movies(seed_item, k)

        hits = len(set(recommended) & set(relevant_items))

        precisions.append(hits / k)
        recalls.append(hits / len(relevant_items))

    return np.mean(precisions), np.mean(recalls)

precision, recall = precision_recall_at_k(user_item, similarity_df)

print("\nEVALUATION METRICS")
print("Precision@10:", round(precision, 3))
print("Recall@10:", round(recall, 3))

# =========================================================
# STEP 9: SAMPLE RECOMMENDATION
# =========================================================

sample_movie = user_item.columns[0]
print("\nRECOMMENDATIONS FOR:", sample_movie)
print(recommend_movies(sample_movie, top_k=5))