# =========================================================
# 1D CNN FOR ONLINE SHOPPERS DATASET
# =========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# CNN expects (batch, channels, features)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test), batch_size=64
)

# =========================================================
# STEP 2: DEFINE 1D CNN MODEL
# =========================================================

class CNN1D(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        conv_output_size = ((input_features - 2) // 2 - 2) // 2
        self.fc1 = nn.Linear(32 * conv_output_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))

model = CNN1D(X_train.shape[2])

# =========================================================
# STEP 3: TRAINING SETUP
# =========================================================

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================================================
# STEP 4: TRAINING LOOP
# =========================================================

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Train Loss: {total_loss:.4f}")

# =========================================================
# STEP 5: EVALUATION
# =========================================================

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        predicted = (preds > 0.5).float()
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

print("\nCNN TEST ACCURACY:", correct / total)
