#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import seaborn
import matplotlib
import sklearn
import pandas

print(sys.version)
print(seaborn.__version__)
print(matplotlib.__version__)
print(sklearn.__version__)
print(pandas.__version__)


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd


# In[3]:


#load the dataset
games = pd.read_csv("games.csv")


# In[4]:


#print the names of columns and overall shape
print(games.columns)
print(games.shape)


# In[5]:


#Make a histogram of all the ratings in the average_rating column
plt.hist(games['average_rating'])
plt.show


# In[6]:


#print the row which has the first zero among all the zeros in the average_rating column
print(games[games['average_rating']==0].iloc[0])
#print the row which has the first non-zero value among all the zeros in the average_rating column
print(games[games['average_rating']>0].iloc[0])


# In[7]:


#Remove any row without user reviews
games= games[games['users_rated']>0]


#Remove rows with missing values
games  = games.dropna(axis=0)


# In[8]:


print(games.columns)


# In[9]:


#correlation matrix of all columns/attributes
corrmat = games.corr()
fig = plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax = 0.8,square = True)
plt.show()


# In[14]:


#Get all the columns from the dataframe in the form of a list
columns = games.columns.tolist()

#Filter the columns to remove the columns that we do not want
columns = [c for c in columns if c not in['bayes_average_rating','average_rating','type','name','id']]

#Store the variable that we are making the predictions on
target = "average_rating"
print(target)


# In[15]:


#Generating training and test datasets
from sklearn.model_selection import train_test_split

#Generating training set
train = games.sample(frac = 0.8,random_state = 1)

#Generating test set by selecting anything that is not in training set
test = games.loc[~games.index.isin(train.index)]


# In[16]:


print(train.shape)
print(test.shape)


# In[17]:


#Import Linear Regression Model
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

#Initialize the model class
LR = LinearRegression()

#Fit the model on the training data
LR.fit(train[columns],train[target])


# In[19]:


#Generate predictions for test set
predictions = LR.predict(test[columns])

#Finding out the error between predictions and actual values
mean_squared_error(predictions,test[target])


# In[21]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error

#Initialize the model class
RFR = RandomForestRegressor(n_estimators=100,min_samples_leaf=10, random_state=1)

#Fit the model on the training data
RFR.fit(train[columns],train[target])


# In[22]:


#Generate predictions for test set
predictions = RFR.predict(test[columns])

#Finding out the error between predictions and actual values
mean_squared_error(predictions,test[target])


# In[29]:


#printing the first row
test[columns].iloc[0]


# In[30]:


#Making prediction for a single game
LR_Rating=LR.predict(test[columns].iloc[0].values.reshape(1,-1))
RFR_Rating=RFR.predict(test[columns].iloc[0].values.reshape(1,-1))

#Printing the reesult
print(LR_Rating)
print(RFR_Rating)


# In[31]:


#Printing the actual value
test[target].iloc[0]


# In[ ]:




