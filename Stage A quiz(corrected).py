#!/usr/bin/env python
# coding: utf-8

# # Hamoye Stage A - Introduction to python For Machine Learning
# ## Used The Food Balances dataset from UN

# ### In this notebook i will be attempting to understand the foodbalance data by doing some important exploration. I will also be trying to answer the questions of the first stage (A) of the hamoye internship program assigment, This is part of the requirements that i must meet in order to graduate at the end of the 4 months internship period. I will be answering around 20 questions.

#  

# ![un_image.png](attachment:un_image.png)

#  

#  

# In[6]:


#importing the pandas and numpy libraries
import pandas as pd
import numpy as np


#  

# In[7]:


#loading the UN foodbalances data into the ide
data=pd.read_csv('foodbalances.csv',encoding='latin-1')


# In[8]:


# getting the first 5 rows of the loaded data using the head method
data.head()


# # Question 1: 
# ### type of error?

# In[9]:


my_tuppy=(1,2,5,8)
my_tuppy[2]=6


# #### The question was about the type of error that would be generated  by trying to append a tupple, Remember that a tupple is a list that cannot be appended

# # Question 2: 
# ### What will be the output?

# In[59]:


### Create a 2d array
S=[['him','sell'],[90,28,43]]


# In[60]:


S[0][1][1]


# #### The value that was generated as the output based on array indexing is "e"

#  

# # Question 3: 
# ### Checking the number of rows and columns

# In[61]:


### df.shape


#  

# # Question 4:
# ### Which year had the least correlation with ‘Element Code’?

# In[62]:


##Selecting the Element Code column and all the years
a=data[['Element Code','Y2014','Y2015','Y2016','Y2017','Y2018']]


# In[63]:


a.corr(method ='pearson').sort_values('Element Code',ascending=True)


# #### From the results above, Year 2016 had a correlations of  0.023444 against Element code which is the smallest in comaprison to all the other years, Year 2014 had the largest correlation with Element Code

#   

#  

# # Question 5:
# ### Which of the following is a python inbuilt module?

# In[64]:


## nath module


#  

# # Question 6:
# ### Perform a groupby operation on ‘Element’. What is the total number of the sum ofProcessing in 2017?

# In[65]:


### here i created series named element2017
element2017=data[['Y2017']].groupby(data['Element']).sum()


# In[66]:


element2017.head(10)


# In[67]:


###filtering only the Processing element  from the data
element2017.loc[['Processing']]


# ####  In 2017, there was a  sum of of  292836.0 in the processing element in 2017 alone

# # Question 7:
# ### Consider the following list of tuples:

# In[68]:


## Creating a list of tuples
y=[(2,4),(7,8),(1,5
                ,9)]


# In[69]:


# asigning an element in the upple to variable x
x = y[1][-1]


# In[70]:


x


#   

# # Question 8:
# ### Select columns ‘Y2017’ and ‘Area’, Perform a groupby operation on ‘Area’. Which of these Areas had the 7th lowest sum in 2017?

# In[71]:


area_2017=data[['Y2017']].groupby(data['Area']).sum()


# In[72]:


area_2017.head(10)


# In[73]:


sorted_2017_area=area_2017.sort_values('Y2017')  ###sorting the values to get the smallest to largest


# In[24]:


sorted_2017_area.iloc[[6],[0]]


# #### position 7 from the least is Guinea Bissau

#  

# # Question 9:
# ### What is the total sum of Wine produced in 2015 and 2018 respectively?

# In[74]:


## Grouping Y2015 and Y2018  by Item

item_15_18=data[['Y2015','Y2018']].groupby(data['Item']).sum()


# In[75]:


item_15_18.head(10)


# In[76]:


##filtering the wine from the resulting dataframe using the loc method
item_15_18.loc[['Wine']]


# #### in Y2015 the value of wine was 4251.81  while in 2018 wine was 4039.32

#  

# # Question 11:
# ### select column ‘Y2017’ and ‘Area’, Perform a groupby operation on ‘Area’. Which ofthese Areas had the highest sum in 2017?

# In[77]:


### Grouping area by the values in 2017
area_2017=data[['Y2017']].groupby(data['Area']).sum()


# In[78]:


area_2017.head(10)


# In[79]:


area_2017.sort_values('Y2017',ascending=False).head(10)   ###Used to sort the values from the emerging data


# #### The results shows that Nigeria had the highest values in 2017

#  

#  

# # Question 12:
# ### How do you create a pandas DataFrame using this list, to look like the table below?

# In[80]:


###creating 2 lists to be used in buil;ding a dataframe
lst = [[35, 'Portugal', 94], [33, 'Argentina', 93], [30 , 'Brazil', 92]]
col = ['Age','Nationality','Overall']


# In[81]:


##Creating dataframe from the above lists
pd.DataFrame(lst, columns = col, index = [i for i in range(1,4)])


#  

#  

# # Question 13:
# ### What is the total number and percentage of missing data in 2014 to 3 decimal

# In[82]:


#getting total null values
nulls_2014=data['Y2014'].isnull().sum()


# In[83]:


nulls_2014


# In[84]:


#getting total null + non null values
total=nulls_2014 +data['Y2014'].notnull().sum()


# In[85]:


total


# In[86]:


nulls_2014/total*100


# #### The number of null values in Y2014 was 1589,  this wa a 2.6073544131401474% of the total values which is 60943
# 

#  

# # Question 14:
# ### Perform a groupby operation on ‘Element’. What year has the highest sum of StockVariation?

# In[87]:


#grouping values in years by the element values
element_by_years=data[['Y2014','Y2015','Y2016','Y2017','Y2018']].groupby(data['Element']).sum()


# In[88]:


element_by_years.head(10)


# In[89]:


#filtering only the Stock Variation
element_by_years.loc[['Stock Variation']]


# #### Under stock variation, Y2014 had the highest with 58749.83

#  

# # Question 15:
# ### What is the mean and standard deviation across the whole dataset for the year2017 to 2 decimal places?

# In[90]:


# The Mean
data['Y2017'].mean()


# In[91]:


# The Standard deveiation
data['Y2017'].std()


# #### The mean is 140.917764860268
# #### The standard devaition is  1671.8623590572788

#  

# # Question 16:
# ### What is the total number of unique countries in the dataset?

# In[92]:


#This generates a series of Countries
countries=data['Area'].unique()


# In[93]:


#Getting the shape of the unique array of countries generated 
countries.shape


# #### The are 49 countries inside the area column of this dataset

#  

#  

# # Question 17:
# ### Which of the following dataframe methods can be used to access elements acrossrows and columns?

# #### df.iloc[] and df.loc[] are used to access elements acrossrows and columns

#  

# # Question 18:
# ### A pandas Dataframe with dimensions (100,3) has how many features andobservations?

# #### 3 features, 100 observations, 
# #### features are the column or label names while obsewrvations are the row names

#  

# # Question 19
# ### What is the total Protein supply quantity in Madagascar in 2015?

# In[94]:


data.head(3)


#  

#   

#   

# In[95]:


madagascar_element=data.loc[data['Area']=='Madagascar',['Element','Y2015']]


# In[96]:


madagascar_element


# In[97]:


madagascar_proteins=madagascar_element[madagascar_element['Element']=='Protein supply quantity (g/capita/day)']


# In[98]:


madagascar_proteins


# In[99]:


madagascar_proteins[['Y2015']].sum(axis=0)


# #### in 2015, madagascar supply of proteins was   173.05 Protein supply quantity (g/capita/day)	

#  

# # Question 20:
# ### How would you select the elements in bold and italics from the array?

#  

# In[100]:


array = ([[94,89, 63],[93,92, 48],[92, 94, 56]])


# In[101]:


array


#  

# # ------------------------------------THE END OF EXPLORATION---------------------------------
# ### The steps involved in this notebook are generaly exploration of the food balances data including trying to come up with some certaing analysis that will make us understand this Data even More, More exploration of this data should be expected in the future 

# In[ ]:




