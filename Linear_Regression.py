# ==================================================
# STEP 0: IMPORT LIBRARIES
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==================================================
# STEP 1: LOAD RAW NETFLIX DATA
# ==================================================

df = pd.read_csv("netflix.csv")

# Select simple features for learning
features = ["runtime", "release_year", "type"]
target = "imdb_score"

df = df[features + [target]]
df = df.dropna(subset=[target])


# ==================================================
# STEP 2: BASIC EDA PLOTS
# ==================================================

plt.hist(df["imdb_score"], bins=20)
plt.title("Distribution of IMDb Scores")
plt.xlabel("IMDb Score")
plt.ylabel("Count")
plt.show()

plt.scatter(df["runtime"], df["imdb_score"], alpha=0.3)
plt.xlabel("Runtime")
plt.ylabel("IMDb Score")
plt.title("Runtime vs IMDb Score (EDA)")
plt.show()


# ==================================================
# STEP 3: TRAIN / TEST SPLIT
# ==================================================

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ==================================================
# STEP 4: PREPROCESS TRAIN DATA ONLY
# ==================================================

imputer = SimpleImputer(strategy="median")
X_train[["runtime", "release_year"]] = imputer.fit_transform(
    X_train[["runtime", "release_year"]]
)

X_train = pd.get_dummies(X_train, columns=["type"], drop_first=True)

scaler = StandardScaler()
X_train[["runtime", "release_year"]] = scaler.fit_transform(
    X_train[["runtime", "release_year"]]
)


# ==================================================
# STEP 5: APPLY SAME PREPROCESSING TO TEST
# ==================================================

X_test[["runtime", "release_year"]] = imputer.transform(
    X_test[["runtime", "release_year"]]
)

X_test = pd.get_dummies(X_test, columns=["type"], drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_test[["runtime", "release_year"]] = scaler.transform(
    X_test[["runtime", "release_year"]]
)


# ==================================================
# STEP 6: TRAIN LINEAR REGRESSION
# ==================================================

model = LinearRegression()
model.fit(X_train, y_train)


# ==================================================
# STEP 7: PREDICTION & EVALUATION
# ==================================================

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# ==================================================
# STEP 8: LINEAR REGRESSION PLOTS
# ==================================================

# Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual IMDb Score")
plt.ylabel("Predicted IMDb Score")
plt.title("Actual vs Predicted IMDb Scores")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.show()

# Residuals
residuals = y_test - y_pred

plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0)
plt.xlabel("Predicted IMDb Score")
plt.ylabel("Residual (Error)")
plt.title("Residual Plot (Linear Regression)")
plt.show()
