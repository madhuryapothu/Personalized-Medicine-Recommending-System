#!/usr/bin/env python
# coding: utf-8

# # Project Name: Personalized Medicine Recommending System
# 
# 
# ## Madhurya

# In[1]:


import numpy as np
import pandas as pd
from warnings import filterwarnings
filterwarnings("ignore")


# In[2]:


df=pd.read_csv('medicine.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.dropna(inplace=True)


# In[7]:


df.duplicated().sum()


# In[8]:


df['Description']


# In[9]:


df['Description'].apply(lambda x:x.split())


# In[10]:


df['Reason'] = df['Reason'].apply(lambda x:x.split())
df['Description'] = df['Description'].apply(lambda x:x.split())


# In[11]:


df['Description'] = df['Description'].apply(lambda x:[i.replace(" ","") for i in x])


# In[12]:


df['tags'] = df['Description'] + df['Reason'] 


# In[13]:


new_df = df[['index','Drug_Name','tags']]


# In[14]:


new_df


# In[15]:


new_df['tags'].apply(lambda x:" ".join(x))


# In[16]:


new_df


# In[17]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[18]:


new_df


# In[19]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[20]:


new_df


# In[21]:


import nltk


# In[22]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english',max_features=5000)


# In[24]:


def stem(text):
  y = []

  for i in text.split():
    y.append(ps.stem(i))

  return " ".join(y) 


# In[25]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[26]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[27]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[28]:


cv.get_feature_names_out()


# In[29]:


from sklearn.metrics.pairwise import cosine_similarity


# In[30]:


cosine_similarity(vectors)


# In[31]:


similarity = cosine_similarity(vectors)


# In[32]:


similarity[1]


# In[33]:


def recommend(medicine):
    medicine_index = new_df[new_df['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in medicines_list:
        print(new_df.iloc[i[0]].Drug_Name)


# In[34]:


recommend("Paracetamol 125mg Syrup 60mlParacetamol 500mg Tablet 10'S")


# In[35]:


import pickle


# In[36]:


pickle.dump(new_df.to_dict(),open('medicine_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




