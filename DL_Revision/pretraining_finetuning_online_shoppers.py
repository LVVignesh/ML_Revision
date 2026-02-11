# =========================================================
# SELF-SUPERVISED PRETRAINING + FINETUNING
# ONLINE SHOPPERS DATASET
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# =========================================================
# STEP 1: LOAD & PREPROCESS DATA
# =========================================================

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

input_dim = X_train.shape[1]

# =========================================================
# STEP 2: AUTOENCODER (SELF-SUPERVISED PRETRAINING)
# =========================================================

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


autoencoder = AutoEncoder(input_dim)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\n=== PRETRAINING AUTOENCODER ===")

pretrain_losses = []

for epoch in range(50):

    optimizer.zero_grad()

    reconstructed = autoencoder(X_train)
    loss = criterion(reconstructed, X_train)

    loss.backward()
    optimizer.step()

    pretrain_losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Reconstruction Loss: {loss.item():.4f}")

# Plot pretraining loss
plt.plot(pretrain_losses)
plt.title("Autoencoder Pretraining Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()


# =========================================================
# STEP 3: FINETUNING CLASSIFIER USING PRETRAINED ENCODER
# =========================================================

class FinetuneClassifier(nn.Module):
    def __init__(self, pretrained_encoder):
        super().__init__()

        self.encoder = pretrained_encoder
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.classifier(latent)


model = FinetuneClassifier(autoencoder.encoder)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

print("\n=== FINETUNING CLASSIFIER ===")

finetune_losses = []

for epoch in range(50):

    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    finetune_losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Classification Loss: {loss.item():.4f}")

# Plot finetuning loss
plt.plot(finetune_losses)
plt.title("Finetuning Loss")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.show()


# =========================================================
# STEP 4: EVALUATION
# =========================================================

with torch.no_grad():
    preds = model(X_test)
    roc_auc = roc_auc_score(y_test.numpy(), preds.numpy())

print("\nFINAL ROC-AUC SCORE:", roc_auc)
