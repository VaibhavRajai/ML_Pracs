import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("Cars.csv")

df=df[["Year","Kms_Driven","Transmission","Owner","Selling_Price"]]

df=df.dropna()

#Encode

df=pd.get_dummies(
    df,
    columns=["Transmission","Owner"],
    drop_first=True,
    dtype=int
)

#Select input and target

X=df.drop("Selling_Price",axis=1)
y=df["Selling_Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Create Models

models={
    "Linear Regression":LinearRegression(),
    "Decision Tree":DecisionTreeRegressor(random_state=42),
    "Random Forest":RandomForestRegressor(n_estimators=100,random_state=42)
}

#Train and compare

for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)

    print("\n",name)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

