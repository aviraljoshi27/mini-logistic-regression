"""Train my LogisticRegression on the breast cancer data, from start to finish."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mini_lr.logistic_regression import LogisticRegression
from mini_lr.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "breast_cancer.csv"
RANDOM_STATE = 42

# 1. Load: Pandas reads the CSV, then switch to NumPy for the maths.
df = pd.read_csv(DATA_PATH)
X = df.drop(columns="target").to_numpy()
y = df["target"].to_numpy()

# 2. Stratified split: keep the same malignant/benign mix in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 3. Scale: learn mean and std from train only, then apply them to both.
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train my model on the scaled training data.
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# 5. Report what happened.
print(f"Train shape: {X_train.shape}, test shape: {X_test.shape}")
print(
    f"Malignant share - train: {np.mean(y_train == 1):.3f}, test: {np.mean(y_test == 1):.3f}"
)
print(f"Loss: first {model.loss_history_[0]:.4f}, last {model.loss_history_[-1]:.4f}")
print(f"Test accuracy: {np.mean(model.predict(X_test_scaled) == y_test):.3f}")
