import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# Standardise features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear SVM
model1 = SVC(kernel="linear")
model1.fit(X_train_scaled, y_train)

pred1 = model1.predict(X_test_scaled)

# Train RBF SVM
model2 = SVC(kernel="rbf")
model2.fit(X_train_scaled, y_train)

pred2 = model2.predict(X_test_scaled)

# Compare test performance
print("\nLinear Kernel SVM")
print("Accuracy:", accuracy_score(y_test, pred1))
print("F1 Score:", f1_score(y_test, pred1, zero_division=0))

print("\nRBF Kernel SVM")
print("Accuracy:", accuracy_score(y_test, pred2))
print("F1 Score:", f1_score(y_test, pred2, zero_division=0))


# Function to plot decision boundaries
def plot_boundary(model, X_data, y_data, title):

    x_min = X_data[:, 0].min() - 1
    x_max = X_data[:, 0].max() + 1
    y_min = X_data[:, 1].min() - 1
    y_max = X_data[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(
        X_data[:, 0],
        X_data[:, 1],
        c=y_data,
        edgecolors="black"
    )

    plt.xlabel("Age (Standardised)")
    plt.ylabel("Estimated Salary (Standardised)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Plot Linear SVM decision boundary
plot_boundary(
    model1,
    X_train_scaled,
    y_train.to_numpy(),
    "Linear Kernel SVM Decision Boundary"
)

# Plot RBF SVM decision boundary
plot_boundary(
    model2,
    X_train_scaled,
    y_train.to_numpy(),
    "RBF Kernel SVM Decision Boundary"
)