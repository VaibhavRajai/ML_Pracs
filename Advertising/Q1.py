import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("Advertising.csv")
print(df.head())
print(df.columns)
print(df.isnull().sum())

df.dropna()

features_set=[
    ["TV"],
    ["radio"],
    ["newspaper"],
    ["TV","radio"],
    ["TV","newspaper"],
    ["radio","newspaper"],
    ["TV","radio","newspaper"]
]

results=[]

for features in features_set:
    X=df[features]
    y=df["sales"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    model=LinearRegression()
    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)

    results.append({
        "Features":"+".join(features),
        "RMSE":rmse,
        "R2":r2
    })

    results_df=pd.DataFrame(results)
    print(results_df)

    print("Best Model: ")
    print(results_df.iloc[results_df["RMSE"].argmin()])

