import pandas as pd
import numpy as np
import random
import re
import string
from datetime import datetime,timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.utils import resample

# create synthetic churn dataset
np.random.seed(42)
n_rows = 1000

data = {
    'CustomerID':range(1,n_rows+1),
    'Name':[f'Customer_{i}' for i in range(1,n_rows+1)],
    'Age':np.random.randint(18,80,size=n_rows),
    'Gender':np.random.choice(['Male','Fmale','Female']),
    'MonthlyCharges':np.random.uniform(20,120,size=n_rows),
    'TenureMonths':np.random.randint(0,72,size=n_rows),
    'ContractType':np.random.choice(['Month-to-Month','One Year','Two Year'],size=n_rows),
    'SignupDate':[datetime(2020,1,1)+timedelta(days=int(x)) for x in np.random.randint(0,365*3,size=n_rows)],
    'IsActive':np.random.choice([1,0],size=n_rows),
    'Churn':np.random.choice([1,0],size=n_rows,p=[0.3,0.7])
}
df = pd.DataFrame(data)
print(df)
print(df.columns)
# introduce some missing values
for col in ['Age', 'Gender', 'MonthlyCharges']:
    df.loc[df.sample(frac=0.05).index,col] = np.nan
# Duplicate some rows
df = pd.concat([df,df.sample(10)],ignore_index=True)
print(df)
print(df.isnull().sum())

# Data Cleaning steps
# Handle missing values
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Gender'].fillna('Unknown',inplace=True)
df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean(),inplace=True)
print(df.isnull().sum())

# Remove Duplicates Rows
df.drop_duplicates(inplace=True)
# Fix Inconsistent DataTypes
print(df.info())
df['CustomerID'] = df['CustomerID'].astype(int)
df['Age']=df['Age'].astype(int)
df['MonthlyCharges']=df['MonthlyCharges'].astype(float)
df['IsActive']=df['IsActive'].astype(int)
df['Churn'] = df['Churn'].astype(int)

print(df.info())

# strip whitespace
df['Name'] = df['Name'].str.strip()
print(df['Name'])

# Correct typos in categorical data
obj = df.select_dtypes(include='object').columns
print(obj)
df['Gender'] = df['Gender'].replace({'Fmale':'Female'})
print(df['Gender'])

# Remove Outliers(ex Age>100 or Monthlycharges>200)
df = df[(df['Age']<=100)&(df['MonthlyCharges']<=200)]
print(df)

# Normalize or scale Numeric values
scaler = StandardScaler()
df[['Age','MonthlyCharges','TenureMonths']] = scaler.fit_transform(df[['Age','MonthlyCharges','TenureMonths']])

# Encode Categorical variables
label_enc = LabelEncoder()
df['Gender'] = label_enc.fit_transform(df['Gender'])
df['ContractType'] = label_enc.fit_transform(df['ContractType'])

# feature extraction (ex extract year from signupdate)
df['SignupYear'] = df['SignupDate'].dt.year
print(df['SignupYear'])

# Handle Date / Time fields(already extracted features)
# convert signupdate to datetime (already is)
df['SignupDate'] = pd.to_datetime(df['SignupDate'])
print(df['SignupDate'])

# Fill or drop zeros values(ex:drop f tenure months is 0)
df = df[df['TenureMonths']!=0]
print(df)
# Reorder columns
df = df[['CustomerID', 'Name', 'Age', 'Gender', 'MonthlyCharges', 'TenureMonths',
       'ContractType', 'SignupDate', 'SignupYear','IsActive', 'Churn']]
print(df)

# Rename columns for clarity
df.rename(columns={'IsActive':'CurrentlyActive'},inplace=True)
print(df.columns)

# Drop Uncessary COlumns(ex Name,CustomerID)
filtered_df = df.drop(columns=['CustomerID','Name','SignupDate'])
print(filtered_df )

# Detect Data IMbalance
print('Class distribution:\n',filtered_df['Churn'].value_counts())
print(df['Churn'])

# Remove Non printable characters(if any)
filtered_df = filtered_df.applymap(lambda x:"".join(c for c in x if c.isprintable()) if isinstance(x,str) else x)
# Handle Negative values(age and charges should not be negative)
filtered_df = filtered_df[(filtered_df['Age']>0)&(filtered_df['MonthlyCharges']>=0)]
print(filtered_df)

# Check for Unique values
for col in filtered_df.columns:
    print(f'{col} has {filtered_df[col].nunique()} unique values')
    
# Backup cleaned Data
filtered_df.to_csv('Cleaned_Customer_Churn.csv',index=False)