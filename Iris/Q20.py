import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris dataset
df = pd.read_csv("Iris.csv")

print("First five rows:")
print(df.head())

# Separate features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Decision Tree using Gini criterion
gini_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)
gini_model.fit(X_train, y_train)

# Train Decision Tree using Entropy criterion
entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
entropy_model.fit(X_train, y_train)

# Predictions
gini_pred = gini_model.predict(X_test)
entropy_pred = entropy_model.predict(X_test)

# Evaluate Gini model
print("\nGini Criterion")
print("-------------------------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, gini_pred))
print("Accuracy:", accuracy_score(y_test, gini_pred))

# Evaluate Entropy model
print("\nEntropy Criterion")
print("-------------------------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, entropy_pred))
print("Accuracy:", accuracy_score(y_test, entropy_pred))

# Compare accuracy
results = pd.DataFrame({
    "Criterion": ["Gini", "Entropy"],
    "Accuracy": [
        accuracy_score(y_test, gini_pred),
        accuracy_score(y_test, entropy_pred)
    ]
})

print("\nModel Comparison:")
print(results.to_string(index=False))

# Visualize both decision trees
plt.figure(figsize=(16, 6))
plot_tree(
    gini_model,
    feature_names=X.columns,
    class_names=gini_model.classes_,
    filled=True,
    rounded=True
)
plt.title("Decision Tree Using Gini Criterion")
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 6))
plot_tree(
    entropy_model,
    feature_names=X.columns,
    class_names=entropy_model.classes_,
    filled=True,
    rounded=True
)
plt.title("Decision Tree Using Entropy Criterion")
plt.tight_layout()
plt.show()