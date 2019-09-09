#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:04:48 2019

@author: partha
"""
########## Customer Lifetime Value Implemention #################################
#import modules
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime 
import numpy as np
from datetime import datetime, timedelta
#from datetime import datetime
from pymongo import MongoClient
#import json
start = datetime.now()

## connect with database 
client = MongoClient('mongodb://xx.xx.xx.xx:27017',
                   username='xxxxxxxx',
                   password='xxxxxxxxxx',
                   authSource='xxxxxxxxxxxx',
                   authMechanism='SCRAM-SHA-1')

db = client['spectaData']
invoiceCollection = db.lku_invoice_list ## collection of the input data

##---------------- Data ---------------------------------------------------------
##----------------Invoice Data --------------------------------------------------
df = invoiceCollection.find({})
## df convert into list
df = list(df)
## df convert into data Frame
df = pd.DataFrame.from_dict(df, orient='columns')
 
## Data wrangling
df.isnull().sum()
## select/subset only useful attribute
data = df[['subscriberid','area','invoiceid','gross_sale','description','reqdate','plan']] 
data.isnull().sum()
data.dtypes

########## Customer Lifetime Value Implemention #################################
data.head()
## Order datetime
data['reqdate']  = pd.to_datetime(data['reqdate'])
data = data.sort_values(by='reqdate')
## Removing Duplicates
filtered_data=data[['area','subscriberid']].drop_duplicates()

#Top ten area's customer
#filtered_data.area.value_counts()[:10].plot(kind='bar')

KHULNA_KDA_AREA_OFFICE_data = data[data['area'] =='KHULNA - KDA AREA OFFICE']
KHULNA_KDA_AREA_OFFICE_data.info()
KHULNA_KDA_AREA_OFFICE_data.describe()
list(KHULNA_KDA_AREA_OFFICE_data)

final_data = data[['subscriberid','invoiceid','reqdate','gross_sale']]
final_data['reqdate'] = final_data['reqdate'].astype(str) ## datetime convert into str
final_data = final_data[(final_data['reqdate'] >= '2016-01-00') & (final_data['reqdate'] <= str(max(final_data.reqdate)))]

###### Check number of transaction every months
## convert date-time to month year
final_data['reqdate']  = pd.to_datetime(final_data['reqdate'])
final_data['month_yr'] = final_data['reqdate'].apply(lambda x: x.strftime('%b-%Y'))
#final_data['num_transactions'] = final_data.groupby(['subscriberid','reqdate'])['invoiceid'].agg(['count']).reset_index()
#transactionCountData = final_data.groupby(['subscriberid'])['invoiceid'].transform('count')

final_data_group=final_data.groupby(['subscriberid']).agg({
                                        'reqdate': lambda date: (date.max() - date.min()).days,
                                        'invoiceid': lambda num: len(num),
#                                        'num_transactions': lambda quant: quant.sum(),
                                        'gross_sale': lambda price: price.sum()}).reset_index()

# Change the name of columns
final_data_group.columns=['subscriberid','num_days','num_transactions','spent_money']
final_data_group.head()

#### Calculate CLTV using following formula:
'''CLTV = ((Average Order Value x Purchase Frequency)/Churn Rate) x Profit margin.
Customer Value = Average Order Value * Purchase Frequency
'''
##1. Calculate average order value
# Average Order Value
final_data_group['avg_order_value']=final_data_group['spent_money']/final_data_group['num_transactions']
final_data_group.head()

##2. Calculate Purchase Frequency
purchase_frequency=sum(final_data_group['num_transactions'])/final_data_group.shape[0]

##3. Calculate Repeat Rate and Churn Rate
# Repeat Rate
repeat_rate = final_data_group[final_data_group.num_transactions > 1].shape[0]/final_data_group.shape[0]

#Churn Rate
churn_rate=1-repeat_rate

purchase_frequency,repeat_rate,churn_rate

##4. Calculate Profit Margin
'''Profit margin is the commonly used profitability ratio. 
It represents how much percentage of total sales has earned as the gain. 
Let's assume our business has approx 5% profit on the total sale.'''

# Profit Margin
final_data_group['profit_margin']=final_data_group['spent_money']*0.05
final_data_group.head()

##5. Calcualte Customer Lifetime Value
# Customer Value
final_data_group['CLV']=(final_data_group['avg_order_value']*purchase_frequency)/churn_rate
#Customer Lifetime Value
final_data_group['cust_lifetime_value']=final_data_group['CLV']*final_data_group['profit_margin']
final_data_group.head()

###############Prediction Model for CLTV##########################################
'''Here, I am going to predict CLTV using Linear Regression Model.
Let's first use the data loaded and filtered above.'''

KHULNA_KDA_AREA_OFFICE_data.head()
real_data = data[['subscriberid','invoiceid','reqdate','gross_sale']]
real_data['reqdate'] = data['reqdate'].astype(str) ## datetime convert into str
real_data = real_data[(real_data['reqdate'] >= '2016-01-00') & (real_data['reqdate'] <= str(max(real_data.reqdate)))]
real_data.head()

## Extract month and year from InvoiceDate/reqdate.
real_data['reqdate']  = pd.to_datetime(real_data['reqdate'])
real_data['month_yr'] = real_data['reqdate'].apply(lambda x: x.strftime('%m-%Y')) ##%b for first three letter of months

'''The pivot table takes the columns as input, 
and groups the entries into a two-dimensional table in such a way 
that provides a multidimensional summarization of the data.'''

sale=real_data.pivot_table(index=['subscriberid'],columns=['month_yr'],values='gross_sale',aggfunc='sum',fill_value=0)
sale.columns = pd.to_datetime(sale.columns, format='%m-%Y').to_period('m')
sale = sale.sort_index(axis=1).reset_index()
## Let's the sum all the months
sale['CLV']=sale.iloc[:,1:].sum(axis=1)

##Selecting Feature
'''Here, you need to divide the given columns into two types of variables dependent(or target variable) and 
independent variable(or feature variables). Select latest 6 month as independent variable.'''
sale = sale[sale.columns[-7:]].copy()
## Drop zero's if more than three zero's in a row
#sale1 = sale1.loc[~sale1.apply(lambda row: (row==0).any(), axis=1)]
sale = sale.loc[~sale.apply(lambda row: (row==0).sum() > 5, axis=1)]
#sale.columns = sale.columns.map( lambda x : sale.columns.mean() if x == 0 else x)
#sale.replace(0,sale.mean(axis=0),inplace=True)
## Feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#dataset_scaled = sc_X.fit_transform(sale)
#y = dataset_scaled[:,6]
#X = dataset_scaled[:,0:6]

y=sale[['CLV']]  ## dependent variable
sale.drop(['CLV'],axis=1, inplace=True)
X = sale[sale.columns[-6:]].copy() ## independent variable

#split training set and test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

################### Modelling Part ##############################################
# import model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

## Linear regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg_pred = linreg.predict(X_test)
#print("R-Square:",metrics.r2_score(y_test, linreg_pred))
metrics.r2_score(y_test, linreg_pred)
#accuracy = linreg.score(X_test,y_test)
#print(accuracy*100,'%')

scores=cross_val_score(linreg,X_test,y_test,cv=5,scoring='neg_mean_absolute_error')
print(r2_score(y_test,linreg_pred,multioutput='variance_weighted'))

## Polynomial regression
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
poly_reg_pred=lin_reg_2.predict(poly_reg.fit_transform(X_test))
metrics.r2_score(y_test, poly_reg_pred)

scores=cross_val_score(lin_reg_2,X_test,y_test,cv=5,scoring='neg_mean_absolute_error')
print(r2_score(y_test,poly_reg_pred,multioutput='variance_weighted'))

## Ridge regression
rr = Ridge(alpha=0.01) #''' higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely
# restricted and in this case linear and ridge regression resembles '''
rr.fit(X_train, y_train)
rr_pred = rr.predict(X_test)
metrics.r2_score(y_test, rr_pred)

Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
scores=cross_val_score(rr,X_test,y_test,cv=5,scoring='neg_mean_absolute_error')
print(r2_score(y_test,rr_pred,multioutput='variance_weighted'))

## LASSO regression
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)
metrics.r2_score(y_test, lasso_pred)

train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)
scores=cross_val_score(lasso,X_test,y_test,cv=5,scoring='neg_mean_absolute_error')
print(r2_score(y_test,lasso_pred,multioutput='variance_weighted'))

## Decision Tree regression
DTregressor = DecisionTreeRegressor(random_state = 0)
DTregressor.fit(X_train, y_train)
DTregressor_pred = DTregressor.predict(X_test)
metrics.r2_score(y_test, DTregressor_pred)

scores=cross_val_score(DTregressor,X_test,y_test,cv=5,scoring='neg_mean_absolute_error')
print(r2_score(y_test,DTregressor_pred,multioutput='variance_weighted'))

## Randomforest Regression
RandomForestregressor = RandomForestRegressor(n_estimators = 500, random_state = 0)                   
RandomForestregressor.fit(X_train, y_train)
RandomForestregressor_pred = RandomForestregressor.predict(X_test) # Predicting a new result
metrics.r2_score(y_test,RandomForestregressor_pred)

scores=cross_val_score(RandomForestregressor,X_test,y_test,cv=5,scoring='neg_mean_absolute_error')
print(r2_score(y_test,RandomForestregressor_pred,multioutput='variance_weighted'))   

## SV Regression 
SVregressor = SVR(kernel = 'rbf')
SVregressor.fit(X_train, y_train)
SVregressor_pred = SVregressor.predict(X_test)
metrics.r2_score(y_test, SVregressor_pred)

scores=cross_val_score(SVregressor,X_test,y_test,cv=5,scoring='neg_mean_absolute_error')
print(r2_score(y_test,SVregressor_pred,multioutput='variance_weighted'))   


## Model comparison consolidate function
dic_data={}
list1=[]
max_clf_output=[]
tuple_l=()
def data_modeling(X_train,y_train,model,X_test,y_test):
    for i in range(len(model)):
        ml=model[i]
        ml.fit(X_train,y_train)
        pred=ml.predict(X_test)
#        acc_score=metrics.r2_score(pd.DataFrame(ml.predict(X_test)),y_test)
        acc_score = metrics.r2_score(y_test, pred)
        tuple_l=(ml.__class__.__name__,acc_score)
        dic_data[ml.__class__.__name__]=[acc_score,ml]
        list1.append(tuple_l)
        print(dic_data)
    for name,val in dic_data.items():
        if val==max(dic_data.values()):
            max_lis=[name,val]
            print('Maximum regressor',name,val)

    return list1,max_lis

list1,max_lis=data_modeling(X_train,y_train,[LinearRegression(),
#                                             PolynomialFeatures(degree = 2),
                                             Ridge(alpha=0.01),
                                             Lasso(),
                                             DecisionTreeRegressor(random_state = 0),
                                             RandomForestRegressor(n_estimators = 500, random_state = 0),
                                             SVR(kernel = 'rbf')],X_test,y_test)
    
model=max_lis[1][1]

## Model score Visualization
modelscore_df=pd.DataFrame(list1,columns=['Regression',"r-square score"])
modelscore_df
modelscore_df['regression code']=np.arange(6)
print(modelscore_df)
modelscore_df.shape[0]
















