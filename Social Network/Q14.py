import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")

print(df.head())
print(df.isnull().sum())

df = df.dropna()

# Select features and target
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Default SVM model with standardisation
default_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# Cross-validation scores for default model
default_cv = cross_val_score(
    default_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nDefault Model")
print("Cross-validation scores:", default_cv)
print("Mean CV Accuracy:", default_cv.mean())

# Train default model
default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)

default_accuracy = accuracy_score(y_test, default_pred)
default_f1 = f1_score(y_test, default_pred, zero_division=0)

print("Test Accuracy:", default_accuracy)
print("Test F1 Score:", default_f1)


# Hyperparameter tuning
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(
    Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ]),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)
print("Best Cross-validation Accuracy:", grid.best_score_)

# Evaluate tuned model
tuned_pred = best_model.predict(X_test)

tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_f1 = f1_score(y_test, tuned_pred, zero_division=0)

print("Tuned Test Accuracy:", tuned_accuracy)
print("Tuned Test F1 Score:", tuned_f1)


# Compare models
comparison = pd.DataFrame({
    "Model": ["Default SVM", "Tuned SVM"],
    "Mean CV Accuracy": [default_cv.mean(), grid.best_score_],
    "Test Accuracy": [default_accuracy, tuned_accuracy],
    "F1 Score": [default_f1, tuned_f1]
})

print("\nModel Comparison")
print(comparison)

if tuned_f1 > default_f1:
    print("\nHyperparameter tuning improved the F1 Score.")
elif tuned_f1 < default_f1:
    print("\nDefault SVM achieved a better F1 Score.")
else:
    print("\nBoth models achieved the same F1 Score.")