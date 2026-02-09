# =========================================================
# TRANSFORMER WITH POSITIONAL ENCODING 
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =========================================================
# STEP 1: LOAD DATA
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

batch_size, num_features = X_train.shape
embed_dim = 32
num_heads = 4


# =========================================================
# STEP 2: POSITIONAL ENCODING (FEATURE-WISE)
# =========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()

        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe


# =========================================================
# STEP 3: MULTI-HEAD SELF ATTENTION
# =========================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, E = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        Q, K, V = qkv.permute(2, 0, 3, 1, 4)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        weights = self.softmax(scores)

        attention = torch.matmul(weights, V)
        attention = attention.transpose(1, 2).reshape(B, N, E)

        return self.out(attention)


# =========================================================
# STEP 4: TRANSFORMER BLOCK
# =========================================================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x


# =========================================================
# STEP 5: TRANSFORMER CLASSIFIER
# =========================================================

class TransformerClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.embedding = nn.Linear(1, embed_dim)
        self.positional = PositionalEncoding(num_features, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(-1)              # (B, F, 1)
        x = self.embedding(x)            # (B, F, E)
        x = self.positional(x)
        x = self.transformer(x)
        x = x.mean(dim=1)                # Pool over features
        x = self.fc(x)
        return self.sigmoid(x)


# =========================================================
# STEP 6: TRAINING
# =========================================================

model = TransformerClassifier(num_features)
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
# STEP 7: EVALUATION
# =========================================================

model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds = (preds > 0.5).int()
    acc = accuracy_score(y_test, preds)

print("\nTRANSFORMER + POSITIONAL ENCODING ACCURACY:", acc)
