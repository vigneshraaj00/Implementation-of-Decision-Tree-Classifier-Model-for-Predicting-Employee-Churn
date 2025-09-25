# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vignesh Raaj
RegisterNumber: 212223230239

import pandas as pd
df=pd.read_csv("Employee.csv")

df.head()

df.info()

df.isnull().sum()

df['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['salary'] = le.fit_transform(df['salary'])
df.head()

x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=df['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

print("Name : Vignesh Raaj")
print("Register No.: 212223230239")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix: ",confusion)

from sklearn.metrics import classification_report
report=classification_report(y_test,y_pred)
print("Classification Report: ",report)

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
*/
```

## Output:
<img width="1217" height="216" alt="image" src="https://github.com/user-attachments/assets/066438f9-c144-429d-9e15-01636995b0ee" />
<img width="1240" height="377" alt="image" src="https://github.com/user-attachments/assets/a38146ae-ba16-4675-964c-7d6492f2b668" />
<img width="1263" height="241" alt="image" src="https://github.com/user-attachments/assets/f5aedfed-8e4f-44ca-b790-a1e5de0d76af" />
<img width="1235" height="100" alt="image" src="https://github.com/user-attachments/assets/a359db22-0495-4790-a6f6-39e57ad0af6f" />
<img width="1214" height="213" alt="image" src="https://github.com/user-attachments/assets/6a48ec4d-2465-4912-b639-da3d1f343cce" />
<img width="1240" height="221" alt="image" src="https://github.com/user-attachments/assets/bfb2ef90-30b6-4a83-ba6a-a20045e8b401" />
<img width="1240" height="50" alt="image" src="https://github.com/user-attachments/assets/fc0f0d39-2b95-43e3-949c-ed727b1a6f08" />
<img width="1246" height="91" alt="image" src="<img width="809" height="103" alt="Screenshot 2025-09-25 160551" src="https://github.com/user-attachments/assets/dbfe64a5-db1a-4511-b24d-d41c71902d4b" />
<img width="1247" height="72" alt="image" src="https://github.com/user-attachments/assets/fd562032-6124-49ff-9ae3-7511387490c5" />
<img width="1250" height="212" alt="image" src="https://github.com/user-attachments/assets/3044333f-d1c2-4d78-810b-ba17a2b81f14" />





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
