
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv(r'C:\Users\DELL\Desktop\Projects\Customer_Churn Prediction\Cleaned_Customer_Churn.csv')

X = df.drop(columns=['Churn'],axis=1)
y = df['Churn']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model1 = LogisticRegression()
model1.fit(X_train,y_train)

y_pred1 = model1.predict(X_test)
print(f'y_pred1: {y_pred1}')

accuracy1 = accuracy_score(y_test,y_pred1)
print(f'Accuracy1: {accuracy1}')

classification_report1 = classification_report(y_test,y_pred1)
print(f'{classification_report1}  classification_report1')

cm1 = confusion_matrix(y_test,y_pred1)
print(f'confusion_matrix1:{cm1}')

# model2
model2 = RandomForestClassifier()
model2.fit(X_train,y_train)

y_pred2 = model2.predict(X_test)
print(f'y_pred2: {y_pred2}')

accuracy2 = accuracy_score(y_test,y_pred2)
print(f'Accuracy2: {accuracy2}')

classification_report2 = classification_report(y_test,y_pred2)
print(f'{classification_report2} Classification_report2')

cm2 = confusion_matrix(y_test,y_pred2)
print(f'confusion_matrix2:{cm2}')

# model3
model3 = DecisionTreeClassifier()
model3.fit(X_train,y_train)

y_pred3 = model3.predict(X_test)
print(f'y_pred3: {y_pred3}')

accuracy3 = accuracy_score(y_test,y_pred3)
print(f'Accuracy3: {accuracy3}')

classification_report3 = classification_report(y_test,y_pred3)
print(f'{classification_report3} Classification_report3')

cm3 = confusion_matrix(y_test,y_pred3)
print(f'confusion_matrix3:{cm3}')

model4 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
model4.fit(X_train,y_train)

y_pred4 = model4.predict(X_test)
print(f'y_pred4: {y_pred4}')

accuracy4 = accuracy_score(y_test,y_pred4)
print(f'Accuracy4: {accuracy4}')

classification_report4 = classification_report(y_test,y_pred4)
print(f'{classification_report4} Classification_report4')

cm4 = confusion_matrix(y_test,y_pred4)
print(f'confusion_matrix4:{cm4}')

# compare the models

model1_result = pd.DataFrame([['LogisticRegression',y_pred1,accuracy1,classification_report1,cm1]],
                             columns=['Method','y_pred1','Accuracy1','classification_report1','confusion_matrix1'])
model2_result = pd.DataFrame([['RandomForestClassifier',y_pred2,accuracy2,classification_report2,cm2]],
                             columns=['Method','y_pred2','Accuracy2','classification_report2','confusion_matrix2'])
model3_result = pd.DataFrame([['DecisionTreeClassifier',y_pred3,accuracy3,classification_report3,cm3]],
                             columns=['Method','y_pred3','Accuracy3','classification_report3','confusion_matrix3'])
model4_result = pd.DataFrame([['XGBClassifier',y_pred4,accuracy4,classification_report4,cm4]],
                             columns=['Method','y_pred4','Accuracy4','classification_report4','confusion_matrix4'])

df_models = pd.concat([model1_result,model2_result,model3_result,model4_result],axis=0)
df_models.reset_index()
# the end