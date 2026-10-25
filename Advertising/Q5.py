import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Advertising.csv")
df = df.dropna()

# Select features and target
X = df[["TV", "radio", "newspaper"]]
y = df["sales"]

# Split dataset: 80% training pool and 20% test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Try 50%, 70%, and 80% of the full dataset
for size in [0.625, 0.875, 1.0]:

    # Select training data
    X_part = X_train.sample(frac=size, random_state=42)
    y_part = y_train.loc[X_part.index]

    # Train model
    model = LinearRegression()
    model.fit(X_part, y_part)

    # Predict sales using the same test set
    pred = model.predict(X_test)

    # Calculate RMSE and R2 Score
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    # Display results
    print("\nTraining data:", int(size * 80), "% of full dataset")
    print("RMSE:", rmse)
    print("R2 Score:", r2)
