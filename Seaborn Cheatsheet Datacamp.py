#!/usr/bin/env python
# coding: utf-8

# #  Seaborn Cheatsheet from datacamp

# In[1]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[25]:


# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
sns.set_style('whitegrid')
sns.set()


# In[3]:


# Load dataset
tips = sns.load_dataset('tips')
titanic = sns.load_dataset('titanic')
iris = sns.load_dataset('iris')
data = pd.DataFrame({'x': np.arange(1,101), 'y': np.random.normal(0,4,100)})


# In[4]:


# Function for Data info
def data_info (df):
    column, nunique, null, null_p, dtype  = [],[],[],[],[]
    for col in df.columns:
        column.append(col)
        nunique.append(df[col].nunique())
        null.append(df[col].isnull().sum())
        null_p.append((df[col].isnull().sum()/df[col].count())*100)
        dtype.append(df[col].dtype)
    return pd.DataFrame({'Column': column, 'N-unique': nunique, 'Null': null, 'Null Percent': null_p,'Dtype': dtype})


# In[5]:


data_info(tips)


# In[6]:


data_info(iris)


# In[7]:


data_info(titanic)


# ### Axis Grids

# In[8]:


a = sns.FacetGrid(data=titanic, col='survived', row='sex')
a = a.map(plt.hist, 'age')


# In[9]:


sns.lmplot(data=iris, x='sepal_width', y='sepal_length', hue='species')


# In[10]:


p = sns.PairGrid(iris)
p = p.map(plt.scatter)


# In[11]:


sns.pairplot(iris)


# In[12]:


sns.jointplot(data=iris, x='sepal_length', y='sepal_width', kind='kde')


# In[13]:


iris.columns


# ### Categorical Plot

# In[14]:


# Strip Plot
sns.stripplot(data=iris, x='species', y='petal_length')


# In[15]:


# Scatter Plot
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species')


# In[16]:


# Bar Chart
sns.barplot(data=titanic, x='sex', y='survived', hue='class')


# In[17]:


# Count Plot
sns.countplot(data=titanic, x='deck', palette='Greens_d')


# In[18]:


# Point Plot
sns.pointplot(data=titanic, x='class', y='survived', markers=['^','o'], linestyles=['-','-o-']);


# In[19]:


# Box plot
sns.boxplot(data=iris, orient='h')


# In[20]:


# Violin Plot
sns.violinplot(data=titanic, x='age', y='sex', hue='survived')


# ### Regression Plot

# In[21]:


sns.regplot(data=iris, x='sepal_length', y='sepal_width')


# ### Distribution Plot

# In[26]:


sns.distplot(data['y'], kde=False, color='r');


# ### Matrix Plot

# In[24]:


sns.heatmap(iris.corr(), annot=True);


# In[ ]:




