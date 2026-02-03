# =========================================================
# FORWARD PROPAGATION - NEURAL NETWORK (NO TRAINING)
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

# Convert target to numeric
df["Revenue"] = df["Revenue"].astype(int)

# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)

print("DATA SHAPE:", df.shape)


# =========================================================
# STEP 2: SPLIT FEATURES & TARGET
# =========================================================

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values.reshape(-1, 1)


# =========================================================
# STEP 3: TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# =========================================================
# STEP 4: FEATURE SCALING
# =========================================================

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)


# =========================================================
# STEP 5: DEFINE ACTIVATION FUNCTIONS
# =========================================================

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =========================================================
# STEP 6: INITIALIZE WEIGHTS (RANDOM)
# =========================================================
# This is IMPORTANT: weights are NOT trained yet

np.random.seed(42)

input_size = X_test.shape[1]      # number of features
hidden_size = 8                   # small hidden layer
output_size = 1                   # binary output

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


# =========================================================
# STEP 7: FORWARD PROPAGATION
# =========================================================

# Input → Hidden layer
Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)

# Hidden → Output layer
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)


# =========================================================
# STEP 8: OUTPUT PREDICTIONS
# =========================================================

print("\nFORWARD PROPAGATION OUTPUT (FIRST 10 SAMPLES)")
print(A2[:10])

print("\nINTERPRETATION:")
print("Each value = probability that user will generate Revenue")
