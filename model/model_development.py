import pandas as pd
import joblib
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["diagnosis"] = data.target  # 0 = malignant, 1 = benign

# Pick 5 features (from the allowed list equivalents in this dataset)
# (The sklearn dataset uses "mean ..." names)
features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
X = df[features]
y = df["diagnosis"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (mandatory)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# Save model + scaler to model directory
model_dir = "."
model_path = os.path.join(model_dir, "breast_cancer_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Saved: {model_path} and {scaler_path}")

# Reload test (no retraining)
loaded_model = joblib.load("breast_cancer_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

sample = X_test.iloc[:1]
sample_scaled = loaded_scaler.transform(sample)
pred = loaded_model.predict(sample_scaled)[0]

print("Reload test prediction:", "Benign" if pred == 1 else "Malignant")
