# =========================================================
# BACKPROPAGATION FROM SCRATCH (NUMPY)
# ONLINE SHOPPERS DATASET
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

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features
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
    epsilon = 1e-8
    loss = -(
        y_true * np.log(y_pred + epsilon) +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )
    return np.mean(loss)

# =========================================================
# STEP 4: INITIALIZE WEIGHTS
# =========================================================

np.random.seed(42)

input_size = X_train.shape[1]
hidden_size = 8
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

learning_rate = 0.01
m = X_train.shape[0]

# =========================================================
# STEP 5: FORWARD PROPAGATION
# =========================================================

Z1 = np.dot(X_train, W1) + b1
A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)

loss_before = binary_cross_entropy(y_train, A2)
print("LOSS BEFORE BACKPROP:", loss_before)

# =========================================================
# STEP 6: BACKPROPAGATION
# =========================================================

# Output layer error
dZ2 = A2 - y_train
dW2 = np.dot(A1.T, dZ2) / m
db2 = np.sum(dZ2, axis=0, keepdims=True) / m

# Hidden layer error
dA1 = np.dot(dZ2, W2.T)
dZ1 = dA1 * (Z1 > 0)  # ReLU derivative

dW1 = np.dot(X_train.T, dZ1) / m
db1 = np.sum(dZ1, axis=0, keepdims=True) / m

# =========================================================
# STEP 7: UPDATE WEIGHTS
# =========================================================

W2 -= learning_rate * dW2
b2 -= learning_rate * db2

W1 -= learning_rate * dW1
b1 -= learning_rate * db1

# =========================================================
# STEP 8: FORWARD PASS AGAIN (AFTER UPDATE)
# =========================================================

Z1 = np.dot(X_train, W1) + b1
A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)

loss_after = binary_cross_entropy(y_train, A2)
print("LOSS AFTER ONE BACKPROP STEP:", loss_after)
