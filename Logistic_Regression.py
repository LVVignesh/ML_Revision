# ==================================================
# STEP 0: IMPORT LIBRARIES
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ==================================================
# STEP 1: LOAD DATA
# ==================================================

df = pd.read_csv("netflix.csv")

features = ["runtime", "release_year", "type"]
df = df[features + ["imdb_score"]]
df = df.dropna(subset=["imdb_score"])


# ==================================================
# STEP 2: CREATE CLASS LABEL
# ==================================================

df["high_rated"] = (df["imdb_score"] >= 7.5).astype(int)


# ==================================================
# STEP 3: EDA FOR CLASSIFICATION
# ==================================================

df["high_rated"].value_counts().plot(kind="bar")
plt.title("High Rated vs Not High Rated")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

plt.scatter(df["runtime"], df["imdb_score"], c=df["high_rated"], alpha=0.4)
plt.xlabel("Runtime")
plt.ylabel("IMDb Score")
plt.title("Runtime vs IMDb (Colored by Class)")
plt.show()


# ==================================================
# STEP 4: TRAIN / TEST SPLIT
# ==================================================

X = df.drop(["imdb_score", "high_rated"], axis=1)
y = df["high_rated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ==================================================
# STEP 5: PREPROCESS
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

X_test[["runtime", "release_year"]] = imputer.transform(
    X_test[["runtime", "release_year"]]
)

X_test = pd.get_dummies(X_test, columns=["type"], drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_test[["runtime", "release_year"]] = scaler.transform(
    X_test[["runtime", "release_year"]]
)


# ==================================================
# STEP 6: TRAIN LOGISTIC REGRESSION
# ==================================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ==================================================
# STEP 7: EVALUATION
# ==================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ==================================================
# STEP 8: LOGISTIC REGRESSION PLOTS
# ==================================================

# Probability distribution
plt.hist(y_prob, bins=20)
plt.xlabel("Predicted Probability (High Rated)")
plt.ylabel("Count")
plt.title("Predicted Probabilities")
plt.show()

# Confusion matrix heat-style plot
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
