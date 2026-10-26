import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")

print(df.head())
print(df.isnull().sum())

df = df.dropna()

# Target variable
y = df["Purchased"]

# Define three feature combinations
X_age = df[["Age"]]
X_salary = df[["EstimatedSalary"]]
X_both = df[["Age", "EstimatedSalary"]]

# Split the data using the same indices for all models
X_train_both, X_test_both, y_train, y_test = train_test_split(
    X_both,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Get the same training and testing rows for each feature combination
X_train_age = X_train_both[["Age"]]
X_test_age = X_test_both[["Age"]]

X_train_salary = X_train_both[["EstimatedSalary"]]
X_test_salary = X_test_both[["EstimatedSalary"]]


# Function to train and evaluate an SVM model
def evaluate_svm(X_train, X_test, y_train, y_test, model_name):

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    return {
        "Feature Combination": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# Model 1: Age Only
result1 = evaluate_svm(
    X_train_age, X_test_age, y_train, y_test, "Age Only"
)

# Model 2: Salary Only
result2 = evaluate_svm(
    X_train_salary, X_test_salary, y_train, y_test, "Salary Only"
)

# Model 3: Age + Salary
result3 = evaluate_svm(
    X_train_both, X_test_both, y_train, y_test, "Age + Salary"
)


# Compare results
results = pd.DataFrame([result1, result2, result3])

print("\nModel Comparison")
print(results.to_string(index=False))

# Identify the best feature combination based on F1 Score
best_model = results.loc[results["F1 Score"].idxmax()]

print("\nMost Effective Feature Combination:")
print(best_model["Feature Combination"])

print("Accuracy:", best_model["Accuracy"])
print("Precision:", best_model["Precision"])
print("Recall:", best_model["Recall"])
print("F1 Score:", best_model["F1 Score"])