# =========================================================
# OPTIMIZERS FROM SCRATCH - ADAM
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

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

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
# STEP 4: INITIALIZE NETWORK
# =========================================================

np.random.seed(42)

input_size = X_train.shape[1]
hidden_size = 8
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# =========================================================
# STEP 5: ADAM OPTIMIZER SETUP
# =========================================================

learning_rate = 0.001
epochs = 200
m = X_train.shape[0]

beta1 = 0.9     # momentum decay
beta2 = 0.999   # RMS decay
epsilon = 1e-8

# Initialize Adam variables
vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)

sW1 = np.zeros_like(W1)
sb1 = np.zeros_like(b1)
sW2 = np.zeros_like(W2)
sb2 = np.zeros_like(b2)

# =========================================================
# STEP 6: TRAINING LOOP WITH ADAM
# =========================================================

for epoch in range(epochs):

    # -------- FORWARD PROP --------
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    loss = binary_cross_entropy(y_train, A2)

    # -------- BACKPROP --------
    dZ2 = A2 - y_train
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X_train.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # -------- ADAM UPDATE --------
    # Momentum
    vW1 = beta1 * vW1 + (1 - beta1) * dW1
    vb1 = beta1 * vb1 + (1 - beta1) * db1
    vW2 = beta1 * vW2 + (1 - beta1) * dW2
    vb2 = beta1 * vb2 + (1 - beta1) * db2

    # RMS
    sW1 = beta2 * sW1 + (1 - beta2) * (dW1 ** 2)
    sb1 = beta2 * sb1 + (1 - beta2) * (db1 ** 2)
    sW2 = beta2 * sW2 + (1 - beta2) * (dW2 ** 2)
    sb2 = beta2 * sb2 + (1 - beta2) * (db2 ** 2)

    # Bias correction
    vW1_corr = vW1 / (1 - beta1 ** (epoch + 1))
    vb1_corr = vb1 / (1 - beta1 ** (epoch + 1))
    vW2_corr = vW2 / (1 - beta1 ** (epoch + 1))
    vb2_corr = vb2 / (1 - beta1 ** (epoch + 1))

    sW1_corr = sW1 / (1 - beta2 ** (epoch + 1))
    sb1_corr = sb1 / (1 - beta2 ** (epoch + 1))
    sW2_corr = sW2 / (1 - beta2 ** (epoch + 1))
    sb2_corr = sb2 / (1 - beta2 ** (epoch + 1))

    # Update weights
    W1 -= learning_rate * vW1_corr / (np.sqrt(sW1_corr) + epsilon)
    b1 -= learning_rate * vb1_corr / (np.sqrt(sb1_corr) + epsilon)
    W2 -= learning_rate * vW2_corr / (np.sqrt(sW2_corr) + epsilon)
    b2 -= learning_rate * vb2_corr / (np.sqrt(sb2_corr) + epsilon)

    # -------- LOG --------
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")
