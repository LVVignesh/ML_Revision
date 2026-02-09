# =========================================================
# MINI DL PROJECT: MLP vs CNN vs TRANSFORMER
# ONLINE SHOPPERS DATASET
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


# =========================================================
# STEP 1: LOAD & PREPARE DATA
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


# =========================================================
# STEP 2: MODELS
# =========================================================

# -------- MLP --------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# -------- CNN (1D) --------
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=3)
        self.fc = nn.Sequential(
            nn.Linear((input_dim - 2) * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------- TRANSFORMER --------
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 32
        self.embed = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x)
        x, _ = self.attn(x, x, x)
        x = x.mean(dim=1)
        return self.fc(x)


# =========================================================
# STEP 3: TRAIN & EVALUATE FUNCTION
# =========================================================

def train_and_evaluate(model, name):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(40):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_test)
        binary = (preds > 0.5).int()
        acc = accuracy_score(y_test, binary)
        auc = roc_auc_score(y_test, preds)

    print(f"\n{name} RESULTS")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}")


# =========================================================
# STEP 4: RUN COMPARISON
# =========================================================

train_and_evaluate(MLP(), "MLP")
train_and_evaluate(CNN1D(), "CNN")
train_and_evaluate(Transformer(), "TRANSFORMER")
