# =========================================================
# OVERFITTING vs UNDERFITTING vs REGULARIZATION (FROM SCRATCH)
# ONLINE SHOPPERS DATASET
# =========================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# STEP 1: LOAD & PREPARE DATA
# =========================================================

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# STEP 2: ACTIVATION FUNCTIONS
# =========================================================

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =========================================================
# STEP 3: LOSS FUNCTION
# =========================================================

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8
    return np.mean(
        -(y_true * np.log(y_pred + eps) +
          (1 - y_true) * np.log(1 - y_pred + eps))
    )

# =========================================================
# STEP 4: MODEL CONFIGURATIONS
# =========================================================

EXPERIMENT = "regularized"
# Options:
# "underfit"
# "overfit"
# "regularized"

np.random.seed(42)

input_size = X_train.shape[1]

if EXPERIMENT == "underfit":
    hidden_size = 2          # too small
    lambda_reg = 0.0
    dropout_rate = 0.0

elif EXPERIMENT == "overfit":
    hidden_size = 64         # too large
    lambda_reg = 0.0
    dropout_rate = 0.0

else:  # regularized
    hidden_size = 16
    lambda_reg = 0.01        # L2 regularization
    dropout_rate = 0.5       # Dropout

output_size = 1
learning_rate = 0.01
epochs = 200

# =========================================================
# STEP 5: INITIALIZE WEIGHTS
# =========================================================

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

m = X_train.shape[0]

# =========================================================
# STEP 6: TRAINING LOOP
# =========================================================

for epoch in range(epochs):

    # ---------- FORWARD PROP ----------
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)

    # Dropout (training only)
    if dropout_rate > 0:
        dropout_mask = (np.random.rand(*A1.shape) > dropout_rate)
        A1 = (A1 * dropout_mask) / (1 - dropout_rate)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # ---------- LOSS ----------
    data_loss = binary_cross_entropy(y_train, A2)
    l2_loss = (lambda_reg / (2 * m)) * (np.sum(W1**2) + np.sum(W2**2))
    loss = data_loss + l2_loss

    # ---------- BACKPROP ----------
    dZ2 = A2 - y_train
    dW2 = (np.dot(A1.T, dZ2) / m) + lambda_reg * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)

    dW1 = (np.dot(X_train.T, dZ1) / m) + lambda_reg * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # ---------- UPDATE ----------
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # ---------- LOG ----------
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# =========================================================
# STEP 7: EVALUATION
# =========================================================

Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)

predictions = (A2_test > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)

print("\nFINAL TEST ACCURACY:", accuracy)
