# =========================================================
# OPTIMIZER COMPARISON: SGD vs ADAM
# ONLINE SHOPPERS DATASET (PyTorch)
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
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

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# =========================================================
# STEP 2: MODEL DEFINITION
# =========================================================

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# STEP 3: TRAINING FUNCTION
# =========================================================

def train_model(optimizer_name="adam"):
    model = SimpleNN(X_train.shape[1])
    criterion = nn.BCELoss()

    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining with {optimizer_name.upper()} optimizer")

    for epoch in range(50):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # Evaluation
    with torch.no_grad():
        preds = model(X_test)
        preds = (preds > 0.5).float()
        accuracy = (preds == y_test).float().mean()

    print(f"{optimizer_name.upper()} TEST ACCURACY: {accuracy.item():.4f}")

# =========================================================
# STEP 4: RUN BOTH OPTIMIZERS
# =========================================================

train_model("sgd")
train_model("adam")
