#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix


# In[2]:


#importing the dataset
df = pd.read_csv('https://github.com/dsrscientist/DSData/blob/master/winequality-red.csv')


# In[3]:


#EDA - Explory Data Analysis
df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df['quality'].value_counts()


# In[11]:


#assigning features
k = df.drop(['quality'],axis=1)
l = df['quality']


# In[12]:


#splitting the dataset
k_train , k_test , l_train , l_test = train_test_split(x,y,random_state =23)


# In[13]:


k_train.shape , k_test.shape , l_train.shape , l_test.shape


# In[14]:


#model selection
model = RandomForestRegressor()
model.fit(k_train , l_train)


# In[15]:


#metrics
l_pred = model.predict(k_test).astype('int32')


# In[16]:


l_pred.shape , k_test.shape


# In[17]:


accuracy_score(l_pred,l_test)


# In[18]:


confusion_matrix(l_test,l_pred)


# In[19]:


print(classification_report(l_test , l_pred))


# In[20]:


#model selection by svc
k = df.drop(['quality'],axis=1)
l = df['quality']


# In[21]:


k.shape


# In[22]:


sc = StandardScaler()
k = sc.fit_transform(k)


# In[23]:


k


# In[24]:


k_train , k_test , l_train , l_test = train_test_split(k,l,test_size=0.3,stratify = l,random_state=23)


# In[25]:


k_train.shape , k_test.shape , l_train.shape , l_test.shape


# In[26]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(k_train , l_train)


# In[27]:


l_pred = svc.predict(k_test)


# In[28]:


l_pred.shape


# In[29]:


l_pred


# In[30]:


print(classification_report(l_pred,l_test))


# In[31]:


#for accuracy purpose retrain the model
df['quality'].unique()


# In[32]:


# we have the quality values start from the 3 to 8
#arrange this as follows
#3,4,5, - 0
#6,7,8, -1
df['quality'].replace(to_replace = [3,4,5],value=0,inplace=True)
df['quality'].replace(to_replace = [6,7,8],value=1,inplace=True)


# In[33]:


J = df['quality']


# In[34]:


k_train , k_test , J_train ,J_test = train_test_split(k,J,test_size=0.3,random_state=23)


# In[35]:


k_train.shape , k_test.shape ,J_train.shape ,J_test.shape


# In[36]:


svc = SVC()


# In[37]:


svc.fit(k_train , J_train)


# In[38]:


l_pred1 = svc.predict(k_test)


# In[39]:


l_pred1


# In[40]:


print(classification_report(l_pred1 , J_test))


# In[41]:


#visualization
sns.pairplot(df)


# In[42]:


fd = df.head(10)


# In[43]:


plt.figure(figsize=(7,7))
sns.heatmap(fd,annot=True)


# In[44]:


plt.figure(figsize=(4,4))
sns.countplot(x='quality',data=df)


# In[45]:


#future prediction
df_data =  df.sample(1)


# In[46]:


df_data


# In[47]:


df_data.shape


# In[48]:


k_data = df_data.drop(['quality'],axis=1)


# In[49]:


k_data = sc.fit_transform(x_data)


# In[50]:


l_pred_data = svc.predict(x_data)


# In[51]:


l_pred_data

