import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("Cars.csv")

df=pd.get_dummies(df,drop_first=True,dtype=int)
df=df.dropna()

X=df.drop("Selling_Price",axis=1)
y=df["Selling_Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Model 1 : All Features

model1=RandomForestRegressor(n_estimators=100,random_state=42)
model1.fit(X_train,y_train)

pred1=model1.predict(X_test)

rmse1 = np.sqrt(mean_squared_error(y_test, pred1))
r2_1 = r2_score(y_test, pred1)

# Feature Importance
importance=pd.DataFrame({
    "Feature":X.columns,
    "Importance":model1.feature_importances_
})

importance=importance.sort_values("Importance",ascending=False)

print("\nFeature Importance Ranking")
print(importance)
# Select Top 3 Features
top3 = importance["Feature"].head(3).tolist()
print("\nTop 3 Features:", top3)
X_train3 = X_train[top3]
X_test3 = X_test[top3]

model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(X_train3, y_train)

pred2 = model2.predict(X_test3)

rmse2 = np.sqrt(mean_squared_error(y_test, pred2))
r2_2 = r2_score(y_test, pred2)

# Compare Models
results = pd.DataFrame({
    "Model": ["All Features", "Top 3 Features"],
    "RMSE": [rmse1, rmse2],
    "R2 Score": [r2_1, r2_2]
})

print("\nModel Comparison:")
print(results)