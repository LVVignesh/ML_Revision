# =========================================================
# SELF-ATTENTION NEURAL NETWORK (TABULAR DATA)
# ONLINE SHOPPERS DATASET
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =========================================================
# STEP 1: LOAD & PREPARE DATA
# =========================================================

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

input_dim = X_train.shape[1]


# =========================================================
# STEP 2: SELF-ATTENTION MODULE
# =========================================================

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, features)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = torch.matmul(Q, K.T) / np.sqrt(x.shape[1])
        weights = self.softmax(scores)

        return torch.matmul(weights, V)


# =========================================================
# STEP 3: ATTENTION-BASED CLASSIFIER
# =========================================================

class AttentionClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.attention = SelfAttention(input_dim)
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.attention(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)


# =========================================================
# STEP 4: TRAINING SETUP
# =========================================================

model = AttentionClassifier(input_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40


# =========================================================
# STEP 5: TRAINING LOOP
# =========================================================

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


# =========================================================
# STEP 6: EVALUATION
# =========================================================

model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds = (preds > 0.5).int()
    acc = accuracy_score(y_test, preds)

print("\nSELF-ATTENTION TEST ACCURACY:", acc)
