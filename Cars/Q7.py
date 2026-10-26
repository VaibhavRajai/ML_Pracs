import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Cars.csv")

df = pd.get_dummies(df, drop_first=True, dtype=int)

df = df[["Year", "Kms_Driven", "Selling_Price"]]
df = df.dropna()

df["Vehicle_age"] = 2026 - df["Year"]

X1 = df[["Kms_Driven"]]
X2 = df[["Kms_Driven", "Vehicle_age"]]
y = df["Selling_Price"]

# Split the data using the same indices for both models
X1_train, X1_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42
)

X2_train, X2_test, _, _ = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

# Train Model 1: Kilometres Driven Only
model1 = LinearRegression()
model1.fit(X1_train, y_train)

pred1 = model1.predict(X1_test)

# Train Model 2: Kilometres Driven + Vehicle Age
model2 = LinearRegression()
model2.fit(X2_train, y_train)

pred2 = model2.predict(X2_test)

# Evaluate Model 1
rmse1 = np.sqrt(mean_squared_error(y_test, pred1))
r2_1 = r2_score(y_test, pred1)

# Evaluate Model 2
rmse2 = np.sqrt(mean_squared_error(y_test, pred2))
r2_2 = r2_score(y_test, pred2)

# Display results
print("Model 1: Kilometres Driven Only")
print("RMSE:", rmse1)
print("R2 Score:", r2_1)

print("\nModel 2: Kilometres Driven + Vehicle Age")
print("RMSE:", rmse2)
print("R2 Score:", r2_2)

# Compare models
print("\nComparison:")

if rmse2 < rmse1 and r2_2 > r2_1:
    print("Vehicle age improves prediction.")
elif rmse2 > rmse1 and r2_2 < r2_1:
    print("Vehicle age does not improve prediction on this test set.")
else:
    print("The results are mixed. Compare both metrics.")