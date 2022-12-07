#!/usr/bin/env python
# coding: utf-8

# # importing the libraries and loading the data into pandas
# import numpy as np
# import pandas as pd
# 
# energydata = pd.read_csv('energydata_complete.csv')
# energydata.head()
# 

# In[15]:


#quick description of the data
energydata.info()


# In[19]:


#normalize the dataset
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

energydata_scaled = pd.DataFrame(scaler.fit_transform(energydata),  columns = energydata.columns)

#get features and labels
X = energydata_scaled.drop(columns=['Appliances'])

y = energydata_scaled['Appliances']


# In[21]:


#split the train and test data

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(y_test.shape))


# In[24]:


#select a sample of the dataset
reg_df = energydata_scaled[['T2', 'T6']]

reg_df.head()


# In[26]:


#reshape sample dataset
x= reg_df['T2'].values.reshape(-1,1)
y = reg_df['T6'].values.reshape(-1,1)


# In[28]:


#split sample dataset into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)


# In[30]:


#linear model on sample of dataset
from sklearn import linear_model


lin_regr = linear_model.LinearRegression()

# Train the model using the training sets
lin_regr.fit(xtrain, ytrain)

# Make predictions using the testing set
pred = lin_regr.predict(xtest)


# In[32]:


#R-squared or Coefficient of determination
from sklearn.metrics import r2_score

r2_score = r2_score(ytest, pred)
print('R-squared:',(round(r2_score, 2)))


# In[34]:


from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
energy_pred = regr.predict(X_test)

print("Training set score: {:.3f}".format(regr.score(X_train, y_train)))
print("Test set score: {:.3f}".format(regr.score(X_test, y_test)))


# In[36]:


#mae
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, energy_pred)

print('MAE:',(round(mae, 2)))

#rss
import numpy as np
rss = np.sum(np.square(y_test - energy_pred))
print('RSS:',(round(rss, 2)))

#root mean squared error
from sklearn.metrics import  mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, energy_pred))
print('RMSE:',(round(rmse, 3)))

#R-squared or coefficient of determination
from sklearn.metrics import r2_score

r2_score = r2_score(y_test, energy_pred)
print('R-squared:',(round(r2_score, 2)))


# In[38]:


#comparing the effects of regularisation #This function is not originally mine.


def get_weights_df(model, feat, col_name):
    
  #this function returns the weight of every feature
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', col_name]
    weights_df[col_name].round(3)
    
    return weights_df


# In[42]:


#weights of linear model
linear_model_weights = get_weights_df(regr, X_train, 'Linear_Model_Weight')
linear_model_weights


# In[44]:


from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(X_train, y_train)

#obtain predictions
ridge_pred = ridge_reg.predict(X_test)

print("Training set score: {:.3f}".format(ridge_reg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(ridge_reg.score(X_test, y_test)))


# In[46]:


#root mean squared error
from sklearn.metrics import  mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
print('RMSE:',(round(rmse, 3)))


# In[48]:


from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)

#obtain predictions
lasso_pred = lasso_reg.predict(X_test)

print("Training set score: {:.3f}".format(lasso_reg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(lasso_reg.score(X_test, y_test)))


# In[50]:


#Lasso regression weights
lasso_weights_df = get_weights_df(lasso_reg, X_train, 'Lasso_weight')
lasso_weights_df


# In[52]:


# root mean squared error
from sklearn.metrics import  mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
print('RMSE:',(round(rmse, 3)))

