# =========================================================
# LOSS FUNCTION - BINARY CROSS ENTROPY
# Dataset: Online Shoppers Intention
# =========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# STEP 1: LOAD DATA
# =========================================================

df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)


# =========================================================
# STEP 2: ACTIVATION FUNCTIONS
# =========================================================

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =========================================================
# STEP 3: LOSS FUNCTION (BINARY CROSS-ENTROPY)
# =========================================================

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8  # prevent log(0)
    loss = -(
        y_true * np.log(y_pred + epsilon) +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )
    return np.mean(loss)


# =========================================================
# STEP 4: INITIALIZE WEIGHTS
# =========================================================

np.random.seed(42)

input_size = X_test.shape[1]
hidden_size = 8
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


# =========================================================
# STEP 5: FORWARD PROPAGATION
# =========================================================

Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)


# =========================================================
# STEP 6: COMPUTE LOSS
# =========================================================

loss = binary_cross_entropy(y_test, A2)

print("LOSS VALUE:", loss)
print("\nINTERPRETATION:")
print("Higher loss = worse predictions")
print("Lower loss = better predictions")
