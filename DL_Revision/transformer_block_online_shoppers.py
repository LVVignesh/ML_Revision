# =========================================================
# TRANSFORMER BLOCK 
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
# STEP 1: DATA
# =========================================================

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)
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
embed_dim = 64   # 🔥 FIX: Transformer embedding size
num_heads = 4


# =========================================================
# STEP 2: MULTI-HEAD ATTENTION
# =========================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B = x.size(0)

        Q = self.q(x).view(B, self.num_heads, self.head_dim)
        K = self.k(x).view(B, self.num_heads, self.head_dim)
        V = self.v(x).view(B, self.num_heads, self.head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        weights = self.softmax(scores)

        attention = torch.matmul(weights, V)
        attention = attention.reshape(B, -1)

        return self.out(attention)


# =========================================================
# STEP 3: TRANSFORMER BLOCK
# =========================================================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=128, dropout=0.3):
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return self.dropout(x)


# =========================================================
# STEP 4: CLASSIFIER
# =========================================================

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return self.sigmoid(x)


# =========================================================
# STEP 5: TRAINING
# =========================================================

model = TransformerClassifier(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
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

print("\nTRANSFORMER BLOCK TEST ACCURACY:", acc)
