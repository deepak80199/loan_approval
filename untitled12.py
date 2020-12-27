# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OL7AFvkzw1Rlr8xA0h3gPCNQUyZNN-zW
"""

import pandas as pd
import numpy as np
import pickle

loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )

loan_data.head()


loan_data.info()

len(loan_data[loan_data['Loan_Status']==1])

loan_data['Gender']=loan_data['Gender'].apply(lambda x:1 if x=='Male' else 0)
loan_data['Gender']

loan_data['Married']=loan_data['Married'].apply(lambda x:1 if x=='Yes' else 0)
loan_data['Married']

loan_data['Education']=loan_data['Education'].apply(lambda x:1 if x=='Graduate' else 0)
loan_data['Education']

loan_data['Self_Employed']=loan_data['Self_Employed'].apply(lambda x:1 if x=='Yes' else 0)
loan_data['Self_Employed']

loan_data['Property_Area']=loan_data['Property_Area'].apply(lambda x:1 if x=='Rural' else (2 if x=='Semiurban' else 3))
loan_data['Property_Area']

loan_data['Dependents']=loan_data['Dependents'].apply(lambda x:1 if x=='1' else (2 if x=='2' else (3 if x=="3+" else 0)))
loan_data['Dependents']



loan_data['LoanAmount']=loan_data['LoanAmount'].fillna(np.mean(loan_data['LoanAmount']))
loan_data['Loan_Amount_Term']=loan_data['Loan_Amount_Term'].fillna(np.mean(loan_data['Loan_Amount_Term']))

loan_data['Credit_History']=loan_data['Credit_History'].fillna((loan_data['Credit_History'].mode().loc[0]))

loan_data['Dependents']=loan_data['Dependents'].fillna(np.mean(loan_data['Dependents']))

loan_data['Dependents']

loan_data.columns

loan_data=loan_data[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']]

loan_data.isnull().sum()

X = loan_data.drop(columns = ['Loan_Status'])       
Y = loan_data.Loan_Status

from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(X)
scaled_features = pd.DataFrame(data=scaled_features)
scaled_features.columns= X.columns

scaled_features.head()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

X_train.head()

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, Y_train)

pickle.dump(log_model,open('model.pkl','wb'))

predictions = log_model.predict(X_test)
print("predictions :",predictions)
model=pickle.load(open('model.pkl','rb'))
print("model pridiction",model.predict(X_test))

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))

'''

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')





test_data['Gender']=test_data['Gender'].apply(lambda x:1 if x=='Male' else 0)
test_data['Gender']

test_data['Married']=test_data['Married'].apply(lambda x:1 if x=='Yes' else 0)
test_data['Married']

test_data['Education']=test_data['Education'].apply(lambda x:1 if x=='Graduate' else 0)
test_data['Education']

test_data['Self_Employed']=test_data['Self_Employed'].apply(lambda x:1 if x=='Yes' else 0)
test_data['Self_Employed']

test_data['Property_Area']=test_data['Property_Area'].apply(lambda x:1 if x=='Rural' else (2 if x=='Semiurban' else 3))
test_data['Property_Area']

test_data['Dependents']=test_data['Dependents'].apply(lambda x:1 if x=='1' else (2 if x=='2' else (3 if x=="3+" else 0)))
test_data['Dependents']

test_data['LoanAmount']=test_data['LoanAmount'].fillna(np.mean(test_data['LoanAmount']))
test_data['Loan_Amount_Term']=test_data['Loan_Amount_Term'].fillna(np.mean(test_data['Loan_Amount_Term']))

test_data['Credit_History']=test_data['Credit_History'].fillna((test_data['Credit_History'].mode().loc[0]))

test_data['Dependents']=test_data['Dependents'].fillna(np.mean(loan_data['Dependents']))

test_data=test_data[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']]

from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(test_data)
scaled_features = pd.DataFrame(data=scaled_features)
scaled_features.columns= X.columns

predictions_final=log_model.predict(scaled_features)

predictions_final

'''