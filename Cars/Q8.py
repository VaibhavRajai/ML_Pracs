import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("Cars.csv")
df = pd.get_dummies(df, drop_first=True, dtype=int)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Basic Model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

rmse1 = np.sqrt(
    -cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    ).mean()
)

print("Basic RMSE:", rmse1)
print("Basic R2:", r2_score(y_test, model.predict(X_test)))

# Hyperparameter Tuning
params = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    params,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train, y_train)

best = grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print("Tuned RMSE:", np.sqrt(-grid.best_score_))
print("Tuned R2:", r2_score(y_test, best.predict(X_test)))