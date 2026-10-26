import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# Test different tree depths
depths = [2, 3, 4, 5]
train_accuracies = []
test_accuracies = []

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print("\nMaximum Depth:", depth)
    print("Training Accuracy:", round(train_acc * 100, 2), "%")
    print("Testing Accuracy:", round(test_acc * 100, 2), "%")

# Plot training and testing accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    depths, train_accuracies,
    marker="o", label="Training Accuracy"
)

plt.plot(
    depths, test_accuracies,
    marker="s", label="Testing Accuracy"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Depth vs Accuracy - Iris Dataset")
plt.xticks(depths)
plt.ylim(0.8, 1.02)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Identify depth with the highest testing accuracy
best_index = test_accuracies.index(max(test_accuracies))
print("\nBest Tree Depth:", depths[best_index])
print("Highest Testing Accuracy:",
      round(test_accuracies[best_index] * 100, 2), "%")