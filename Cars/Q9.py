import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score


df=pd.read_csv("Cars.csv")

y=df["Selling_Price"]

#Numerical only
X1=df.select_dtypes(include=np.number).drop(columns=["Selling_Price"])

X2=pd.get_dummies(df.drop(columns=["Selling_Price"]),drop_first=True,dtype=int)

X1_train,X1_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,random_state=42)
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y,test_size=0.2,random_state=42)

model1=DecisionTreeRegressor()
model1.fit(X1_train,y_train)

pred1=model1.predict(X1_test)

model2 = DecisionTreeRegressor(random_state=42)
model2.fit(X2_train, y2_train)
pred2 = model2.predict(X2_test)

print("Numerical Features Only")
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred1)))
print("R2 Score:", r2_score(y_test, pred1))

print("\nWith Categorical Encoding")
print("RMSE:", np.sqrt(mean_squared_error(y2_test, pred2)))
print("R2 Score:", r2_score(y2_test, pred2))


