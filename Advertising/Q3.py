import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Advertising.csv")

print(df.head())
print(df.columns)
print(df.isnull().sum())

df = df.dropna()

# Target variable
y = df["sales"]

# Model 1: TV + Radio + Newspaper
X_train, X_test, y_train, y_test = train_test_split(
    df[["TV", "radio", "newspaper"]],
    y,
    test_size=0.2,
    random_state=42
)

model1 = LinearRegression()
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

rmse1 = np.sqrt(mean_squared_error(y_test, pred1))
r2_1 = r2_score(y_test, pred1)

n = len(y_test)
p1 = 3

adj_r2_1 = 1 - ((1 - r2_1) * (n - 1) / (n - p1 - 1))

# Model 2: TV + Radio Only
X_train2 = X_train[["TV", "radio"]]
X_test2 = X_test[["TV", "radio"]]

model2 = LinearRegression()
model2.fit(X_train2, y_train)

pred2 = model2.predict(X_test2)

rmse2 = np.sqrt(mean_squared_error(y_test, pred2))
r2_2 = r2_score(y_test, pred2)

p2 = 2

adj_r2_2 = 1 - ((1 - r2_2) * (n - 1) / (n - p2 - 1))

# Compare models
results = pd.DataFrame({
    "Model": ["TV + Radio + Newspaper", "TV + Radio"],
    "RMSE": [rmse1, rmse2],
    "R2 Score": [r2_1, r2_2],
    "Adjusted R2 Score": [adj_r2_1, adj_r2_2]
})

print("\nModel Comparison")
print(results)

if rmse2 < rmse1 and adj_r2_2 >= adj_r2_1:
    print("\nNewspaper may not provide useful additional information.")
else:
    print("\nCompare the scores to determine the contribution of Newspaper.")