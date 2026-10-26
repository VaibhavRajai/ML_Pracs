import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix,accuracy_score,precision_score,f1_score,recall_score)

df=pd.read_csv("Social_Network_Ads.csv")

print(df.head())
print(df.isnull().sum())

df=df.dropna()

#Select features and target
X=df[["Age","EstimatedSalary"]]
y=df["Purchased"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Standardise numerical features
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Model 1:Linear SVM

model1=SVC(kernel="linear")
model1.fit(X_train,y_train)

pred1=model1.predict(X_test)

model2 = SVC(kernel="rbf")
model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

#Evalauation Function

def evaluate_model(name,actual,predicted):
    print("\n",name)
    print("--------------")
    print("Confusion Matrix: ")
    print(confusion_matrix(actual,predicted))
    print("Accuracy:", accuracy_score(actual, predicted))
    print("Precision:", precision_score(actual, predicted, zero_division=0))
    print("Recall:", recall_score(actual, predicted, zero_division=0))
    print("F1 Score:", f1_score(actual, predicted, zero_division=0))

evaluate_model("Linear Kernel SVM",y_test,pred1)
evaluate_model("RBF Kernel SVM",y_test,pred2)


accuracy1 = accuracy_score(y_test, pred1)
accuracy2 = accuracy_score(y_test, pred2)

f1_1 = f1_score(y_test, pred1, zero_division=0)
f1_2 = f1_score(y_test, pred2, zero_division=0)

print("\nModel Comparison")
print("------------------------------")

if f1_2 > f1_1:
    print("RBF Kernel SVM performs better based on F1 Score.")
elif f1_1 > f1_2:
    print("Linear Kernel SVM performs better based on F1 Score.")
else:
    print("Both models have the same F1 Score.")

print("Linear SVM Accuracy:", accuracy1)
print("RBF SVM Accuracy:", accuracy2)