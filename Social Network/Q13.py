import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score

df=pd.read_csv("Social_Network_Ads.csv")

print(df.head())
print(df.isnull().sum())
df=df.dropna()

#Select Features and target
X=df[["Age","EstimatedSalary"]]
y=df["Purchased"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Model 1 : RBF without feature scaling

model1=SVC(kernel="rbf")
model1.fit(X_train,y_train)

pred1=model1.predict(X_test)

print("\nRBF SVM Without Scaling")
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred1))
print("Accuracy:", accuracy_score(y_test, pred1))
print("F1 Score:", f1_score(y_test, pred1, zero_division=0))

#Standardised Features

scaler=StandardScaler()

X_train_sclaed=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

model2=SVC(kernel="rbf")
model2.fit(X_train_sclaed,y_train)

pred2=model2.predict(X_test_scaled)

print("\nRBF SVM With Scaling")
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred2))
print("Accuracy:", accuracy_score(y_test, pred2))
print("F1 Score:", f1_score(y_test, pred2, zero_division=0))
accuracy1 = accuracy_score(y_test, pred1)
accuracy2 = accuracy_score(y_test, pred2)

f1_1 = f1_score(y_test, pred1, zero_division=0)
f1_2 = f1_score(y_test, pred2, zero_division=0)

print("\nComparison")

if accuracy2 > accuracy1 and f1_2 > f1_1:
    print("Feature scaling improves the model's performance.")
elif accuracy2 < accuracy1 and f1_2 < f1_1:
    print("Feature scaling reduces the model's performance on this test set.")
else:
    print("The results are mixed. Compare accuracy and F1 Score.")