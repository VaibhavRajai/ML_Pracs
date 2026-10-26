import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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

# Split dataset (same partition for both models)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Logistic Regression with standardisation
lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# RBF Kernel SVM with standardisation
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf"))
])

svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Function to evaluate models
def evaluate_model(name, y_test, y_pred):
    print("\n", name)
    print("-" * 40)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

# Evaluate both models
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("RBF Kernel SVM", y_test, svm_pred)

# Compare results in a table
results = pd.DataFrame({
    "Model": ["Logistic Regression", "RBF Kernel SVM"],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, svm_pred)
    ],
    "Precision": [
        precision_score(y_test, lr_pred, zero_division=0),
        precision_score(y_test, svm_pred, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, lr_pred, zero_division=0),
        recall_score(y_test, svm_pred, zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, lr_pred, zero_division=0),
        f1_score(y_test, svm_pred, zero_division=0)
    ]
})

print("\nModel Comparison:")
print(results.to_string(index=False))

# Select the model with the higher F1-score
best_model = results.loc[results["F1 Score"].idxmax()]

print("\nPreferred Model based on F1 Score:")
print(best_model["Model"])
print("F1 Score:", best_model["F1 Score"])