# =========================================================
# PURCHASE PROBABILITY RANKING
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

# =========================================================
# STEP 1: LOAD DATA
# =========================================================

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)

print("RAW DATA SHAPE:", df.shape)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values.reshape(-1, 1)

# =========================================================
# STEP 2: TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================================================
# STEP 3: FEATURE SCALING
# =========================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# =========================================================
# STEP 4: DEFINE TABULAR MLP MODEL
# =========================================================

class TabularMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

model = TabularMLP(X_train.shape[1])

# =========================================================
# STEP 5: WEIGHTED LOSS (CLASS IMBALANCE)
# =========================================================

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================================================
# STEP 6: TRAINING LOOP
# =========================================================

epochs = 50
losses = []

for epoch in range(epochs):
    model.train()

    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

# =========================================================
# STEP 7: EVALUATION (RANKING)
# =========================================================

model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_probs = torch.sigmoid(test_logits).numpy().flatten()

# ROC-AUC
roc_auc = roc_auc_score(y_test.numpy(), test_probs)
print("\nROC-AUC SCORE:", roc_auc)

# =========================================================
# STEP 8: PRECISION@K (BUSINESS METRIC)
# =========================================================

def precision_at_k(y_true, y_scores, k=10):
    idx = np.argsort(y_scores)[::-1][:k]
    return y_true[idx].sum() / k

precision_10 = precision_at_k(y_test.numpy().flatten(), test_probs, k=10)
print("Precision@10:", precision_10)

# =========================================================
# STEP 9: PLOTS
# =========================================================

# Loss curve
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Probability distribution
plt.hist(test_probs[y_test.numpy().flatten() == 0], bins=50, alpha=0.6, label="No Purchase")
plt.hist(test_probs[y_test.numpy().flatten() == 1], bins=50, alpha=0.6, label="Purchase")
plt.legend()
plt.title("Predicted Purchase Probability Distribution")
plt.show()

# =========================================================
# STEP 10: TOP-K RANKING OUTPUT
# =========================================================

ranking_df = pd.DataFrame({
    "Purchase_Probability": test_probs,
    "Actual": y_test.numpy().flatten()
}).sort_values(by="Purchase_Probability", ascending=False)

print("\nTOP 5 USERS BY PURCHASE PROBABILITY")
print(ranking_df.head())
