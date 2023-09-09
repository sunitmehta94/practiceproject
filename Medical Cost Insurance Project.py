#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as pt
import warnings
warnings.filterwarnings("ignore")


# In[8]:


df=pd.read_csv("https://github.com/dsrscientist/dataset4/blob/main/medical_cost_insurance.csv")
df


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[12]:


features = ['sex', 'smoker', 'region']
 
plt.subplots(figsize=(20, 10))
for j, col in enumerate(features):
    plt.subplot(1, 3, j + 1)
 
    z = df[col].value_counts()
    plt.pie(z.values,
            labels=z.index,
            autopct='%1.1f%%')
 
plt.show()


# In[13]:


features = ['sex', 'children', 'smoker', 'region']
 
plt.subplots(figsize=(20, 10))
for j, col in enumerate(features):
    plt.subplot(2, 2, j + 1)
    df.groupby(col).mean()['charges'].plot.bar()
plt.show()


# In[14]:


features = ['age', 'bmi']
 
plt.subplots(figsize=(17, 7))
for j, col in enumerate(features):
    plt.subplot(1, 2, j + 1)
    sb.scatterplot(data=df, z=col,
                   h='charges',
                   hue='smoker')
plt.show()


# In[15]:


df.drop_duplicates(inplace=True)
sns.boxplot(df['age'])


# In[16]:


sns.boxplot(df['bmi'])


# In[17]:


A1=df['bmi'].quantile(0.25)
A2=df['bmi'].quantile(0.5)
A3=df['bmi'].quantile(0.75)
iqr=A3-A1
lowlim=A1-1.5*iqr
upplim=A3+1.5*iqr
print(lowlim)
print(upplim)


# In[18]:


from feature_engine.outliers import ArbitraryOutlierCapper
ar=ArbitraryOutlierCapper(min_capping_dict={'bmi':13.6749},max_capping_dict={'bmi':47.315})
df[['bmi']]=ar.fit_transform(df[['bmi']])
sns.boxplot(df['bmi'])


# In[19]:


df['bmi'].skew()
df['age'].skew()


# In[20]:


df['sex']=df['sex'].map({'male':0,'female':1})
df['smoker']=df['smoker'].map({'yes':1,'no':0})
df['region']=df['region'].map({'northwest':0, 'northeast':1,'southeast':2,'southwest':3})


# In[21]:


df.corr()


# In[22]:


C=df.drop(['charges'],axis=1)
D=df[['charges']]
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
n1=[]
n2=[]
n3=[]
cvs=0
for i in range(40,50):
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=i)
lrmodel=LinearRegression()
lrmodel.fit(xtrain,ytrain)
n1.append(lrmodel.score(xtrain,ytrain))
n2.append(lrmodel.score(xtest,ytest))
cvs=(cross_val_score(lrmodel,X,Y,cv=5,)).mean()
n3.append(cvs)
df1=pd.DataFrame({'train acc':l1,'test acc':l2,'cvs':l3})
df1


# In[23]:


xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)
lrmodel=LinearRegression()
lrmodel.fit(xtrain,ytrain)
print(lrmodel.score(xtrain,ytrain))
print(lrmodel.score(xtest,ytest))
print(cross_val_score(lrmodel,X,Y,cv=5,).mean())


# In[24]:


from sklearn.metrics import r2_score
svrmodel=SVR()
svrmodel.fit(xtrain,ytrain)
ypredtrain1=svrmodel.predict(xtrain)
ypredtest1=svrmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain1))
print(r2_score(ytest,ypredtest1))
print(cross_val_score(svrmodel,X,Y,cv=5,).mean())


# In[25]:


rfmodel=RandomForestRegressor(random_state=42)
rfmodel.fit(xtrain,ytrain)
ypredtrain2=rfmodel.predict(xtrain)
ypredtest2=rfmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain2))
print(r2_score(ytest,ypredtest2))
print(cross_val_score(rfmodel,X,Y,cv=5,).mean())
from sklearn.model_selection import GridSearchCV
estimator=RandomForestRegressor(random_state=42)
param_grid={'n_estimators':[10,40,50,98,100,120,150]}
grid=GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
rfmodel=RandomForestRegressor(random_state=42,n_estimators=120)
rfmodel.fit(xtrain,ytrain)
ypredtrain2=rfmodel.predict(xtrain)
ypredtest2=rfmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain2))
print(r2_score(ytest,ypredtest2))
print(cross_val_score(rfmodel,X,Y,cv=5,).mean())


# In[26]:


gbmodel=GradientBoostingRegressor()
gbmodel.fit(xtrain,ytrain)
ypredtrain3=gbmodel.predict(xtrain)
ypredtest3=gbmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain3))
print(r2_score(ytest,ypredtest3))
print(cross_val_score(gbmodel,X,Y,cv=5,).mean())
from sklearn.model_selection import GridSearchCV
estimator=GradientBoostingRegressor()
param_grid={'n_estimators':[10,15,19,20,21,50],'learning_rate':[0.1,0.19,0.2,0.21,0.8,1]}
grid=GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
gbmodel=GradientBoostingRegressor(n_estimators=19,learning_rate=0.2)
gbmodel.fit(xtrain,ytrain)
ypredtrain3=gbmodel.predict(xtrain)
ypredtest3=gbmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain3))
print(r2_score(ytest,ypredtest3))
print(cross_val_score(gbmodel,X,Y,cv=5,).mean())


# In[27]:


xgmodel=XGBRegressor()
xgmodel.fit(xtrain,ytrain)
ypredtrain4=xgmodel.predict(xtrain)
ypredtest4=xgmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain4))
print(r2_score(ytest,ypredtest4))
print(cross_val_score(xgmodel,X,Y,cv=5,).mean())
from sklearn.model_selection import GridSearchCV
estimator=XGBRegressor()
param_grid={'n_estimators':[10,15,20,40,50],'max_depth':[3,4,5],'gamma':[0,0.15,0.3,0.5,1]}
grid=GridSearchCV(estimator,param_grid,scoring="r2",cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_params_)
xgmodel=XGBRegressor(n_estimators=15,max_depth=3,gamma=0)
xgmodel.fit(xtrain,ytrain)
ypredtrain4=xgmodel.predict(xtrain)
ypredtest4=xgmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain4))
print(r2_score(ytest,ypredtest4))
print(cross_val_score(xgmodel,X,Y,cv=5,).mean())


# In[28]:


fts=pd.DataFrame(data=grid.best_estimator_.feature_importances_,index=X.columns,columns=['Importance'])
fts


# In[29]:


important_feature=fts[fts['Importance']>0.01]
important_feature


# In[30]:


df.drop(df[['sex','region']],axis=1,inplace=True)
Xf=df.drop(df[['charges']],axis=1)
X=df.drop(df[['charges']],axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xf,Y,test_size=0.2,random_state=42)
finalmodel=XGBRegressor(n_estimators=15,max_depth=3,gamma=0)
finalmodel.fit(xtrain,ytrain)
ypredtrain4=finalmodel.predict(xtrain)
ypredtest4=finalmodel.predict(xtest)
print(r2_score(ytrain,ypredtrain4))
print(r2_score(ytest,ypredtest4))
print(cross_val_score(finalmodel,X,Y,cv=5,).mean())


# In[31]:


from pickle import dump
dump(finalmodel,open('insurancemodelf.pkl','wb'))


# In[ ]:




