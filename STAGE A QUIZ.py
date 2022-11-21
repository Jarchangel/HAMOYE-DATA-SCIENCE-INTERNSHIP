#!/usr/bin/env python
# coding: utf-8

# In[12]:


#QUESTION1
import pandas as pd

y = [(2, 4), (7, 8), (1, 5, 9)]
x = y[1][1]
x


# In[13]:


#QUESTION2
import pandas as pd
lst = [[35, 'Portugal', 94], [33, 'Argentina', 93], [30 , 'Brazil', 92]]

col = ['Age','Nationality','Overall']
df = pd.DataFrame(lst, columns=col, index=[1,2,3])
df


# In[21]:


import pandas as pd
df = pd.read_csv('FoodBalanceSheets_E_Africa_NOFLAG.csv', encoding='latin-1')
df.head()


# In[25]:


df.groupby('Area')['Y2017'].sum()
df.head(7)


# In[26]:


S = [['him', 'sell'], [90, 28, 43]]

S[0][1][1]

