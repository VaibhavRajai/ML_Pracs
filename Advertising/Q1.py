import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Advertising.csv")

# Inspect dataset
print("First Five Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# Preprocess dataset
df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
df.dropna(inplace=True)

# Features and target
features = ["TV", "Radio", "Newspaper"]
X = df[features]
y = df["Sales"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

# Train models using different feature combinations
for r in range(1, len(features) + 1):
    for combo in combinations(features, r):

        X_train_combo = X_train[list(combo)]
        X_test_combo = X_test[list(combo)]

        model = LinearRegression()
        model.fit(X_train_combo, y_train)

        y_pred = model.predict(X_test_combo)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Features": " + ".join(combo),
            "RMSE": round(rmse, 4),
            "R2 Score": round(r2, 4)
        })

# Display results
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Best models
best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
best_r2 = results_df.loc[results_df["R2 Score"].idxmax()]

print("\nBest Model Based on Lowest RMSE:")
print(best_rmse)

print("\nBest Model Based on Highest R2 Score:")
print(best_r2)