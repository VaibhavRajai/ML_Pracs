import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv("Advertising.csv")
print(df.head())
print(df.columns)
print(df.isnull().sum())

df=df.dropna()

X=df[["TV","radio"]]
y=df["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model=LinearRegression()
model.fit(X_train,y_train)

pred=model.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,pred))
r2=r2_score(y_test,pred)

#Polynomial

poly=PolynomialFeatures(degree=2)

X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.fit_transform(X_test)

model2=LinearRegression()
model2.fit(X_train_poly,y_train)

pred2=model2.predict(X_test_poly)

rmse2=np.sqrt(mean_squared_error(y_test,pred2))
r2_2=r2_score(y_test,pred2)

print("\n Linear Regression: ")
print("RMSE: ",rmse)
print("R2 Score: ",r2)

print("\nPolynomial Regression: ")
print("RMSE: ",rmse2)
print("R2 Score: ",r2_2)



#Finding the better model

if rmse2 < rmse:
    print("\nPolynomial Regression Performs Better")
elif rmse < rmse2:
    print("\nLinear Regression Performs Better")
else:
    print("\nBoth models have the same RMSE")
