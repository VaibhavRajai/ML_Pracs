import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("Cars.csv")
df=pd.get_dummies(df,drop_first=True,dtype=int)

X=df.drop("Selling_Price",axis=1)
y=df["Selling_Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Find the Outliers

Q1=y_train.quantile(0.25)
Q3=y_train.quantile(0.75)
IQR=Q3-Q1

mask=(y_train>=Q1-1.5*IQR) & (y_train<=Q3+1.5*IQR)

print("Outliers: ",len(y_train)-mask.sum())


m1=DecisionTreeRegressor()
m1.fit(X_train,y_train)

m2=DecisionTreeRegressor()
m2.fit(X_train[mask],y_train[mask])


for name, model in [("Original", m1), ("Without Outliers", m2)]:
    pred = model.predict(X_test)
    print("\n", name)
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R2 Score:", r2_score(y_test, pred))
