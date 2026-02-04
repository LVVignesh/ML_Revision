# =========================================================
# PYTORCH NEURAL NETWORK - ONLINE SHOPPERS DATASET
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
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

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# =========================================================
# STEP 2: DEFINE NEURAL NETWORK
# =========================================================

class OnlineShopperNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

# =========================================================
# STEP 3: INITIALIZE MODEL, LOSS, OPTIMIZER
# =========================================================

input_size = X_train.shape[1]

model = OnlineShopperNN(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================================================
# STEP 4: TRAINING LOOP
# =========================================================

epochs = 50

for epoch in range(epochs):

    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# =========================================================
# STEP 5: EVALUATION
# =========================================================

model.eval()

with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = (predictions >= 0.5).float()

accuracy = (predicted_classes.eq(y_test)).sum() / y_test.shape[0]

print("\nTEST ACCURACY:", accuracy.item())
