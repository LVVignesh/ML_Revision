# =========================================================
# LEARNING RATE SCHEDULER + VALIDATION LOOP
# ONLINE SHOPPERS (PyTorch)
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# STEP 1: DATA PREPARATION
# =========================================================

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values

# Train / Validation / Test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# =========================================================
# STEP 2: MODEL
# =========================================================

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

model = Net(X_train.shape[1])

# =========================================================
# STEP 3: LOSS, OPTIMIZER, SCHEDULER
# =========================================================

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,   # reduce LR every 10 epochs
    gamma=0.5       # LR = LR * 0.5
)

# =========================================================
# STEP 4: TRAINING + VALIDATION LOOP
# =========================================================

epochs = 50

for epoch in range(epochs):

    # ---- TRAIN ----
    model.train()
    optimizer.zero_grad()

    train_preds = model(X_train)
    train_loss = criterion(train_preds, y_train)

    train_loss.backward()
    optimizer.step()

    # ---- VALIDATE ----
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val)

    scheduler.step()

    if epoch % 5 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f} | "
            f"LR: {current_lr:.5f}"
        )

# =========================================================
# STEP 5: FINAL TEST EVALUATION
# =========================================================

with torch.no_grad():
    test_preds = model(X_test)
    test_preds = (test_preds > 0.5).float()
    test_acc = (test_preds == y_test).float().mean()

print("\nFINAL TEST ACCURACY:", test_acc.item())
