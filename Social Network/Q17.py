import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")

print("First five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

# Select features and target
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train RBF Kernel SVM
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

model.fit(X_train, y_train)

# Predict test-set results
y_pred = model.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nError Analysis:")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Identify incorrectly classified customers
results = X_test.copy()
results["Actual_Purchased"] = y_test
results["Predicted_Purchased"] = y_pred

false_positives = results[
    (results["Actual_Purchased"] == 0) &
    (results["Predicted_Purchased"] == 1)
]

false_negatives = results[
    (results["Actual_Purchased"] == 1) &
    (results["Predicted_Purchased"] == 0)
]

print("\nFalse Positive Customers:")
print(false_positives)

print("\nFalse Negative Customers:")
print(false_negatives)