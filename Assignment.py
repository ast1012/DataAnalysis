#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Loading the dataset using pandas

# In[6]:


data = pd.read_csv(r'C:\Users\astha\OneDrive\Desktop\sample\dataset.csv')


# ### Displaying the first 5 rows of data 

# In[7]:


data = data.drop(['Unnamed: 3'],1)
print(data.head(5))


# ### Applying Label Encoder

# In[8]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['CLASS1'] = label_encoder.fit_transform(data['CLASS1'])
data['CLASS2'] = label_encoder.fit_transform(data['CLASS2'])


# ### Applying Tf-Idf Vectorizer
# #### Here categorical value is replaced with a numeric value between 0 and the number of classes minus 1. Suppose if the categorical variable value contains 5 distinct classes, we use (0, 1, 2, 3, and 4).

# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
x = v.fit_transform(data['QUESTIONS'])


# ### Extracting Features and Target Variables

# In[10]:


y = np.array(data.drop(['QUESTIONS'],1))


# In[11]:


x = np.array(data['QUESTIONS'])


# In[12]:


print(y)


# In[16]:


print(x)
#after applying tf-idf vectorizer the data is converted into numeric data so that it can be processed


# ### Splitting the data into training and testing data
# #### Here the test set containg 30 % of data and training set contains 70% of data

# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state=0)  


# ### Training the Model
# #### Here K Nearest Neighbours Model is imported and trained on training data

# In[18]:


from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)


# ### Predicting the result using test dataset

# In[19]:


#Predicting the test set result  
y_pred= classifier.predict(x_test) 


# ### Here the number reprsent the classes. 1st Column represents the prediction of class 1 and second column represent prediction of class 2

# In[20]:


print(y_pred)


# In[21]:


print("Class one is",y_pred[0][0],"Class two is",y_pred[0][1])  #prediction of data in row 0 of test dataset
print("Class one is",y_pred[1][0],"Class two is",y_pred[1][1])  #prediction of data in row 1 of test dataset
print("Class one is",y_pred[2][0],"Class two is",y_pred[2][1])  #prediction of data in row 2 of test dataset
print("Class one is",y_pred[10][0],"Class two is",y_pred[10][1])  #prediction of data in row 10 of test dataset
print("Class one is",y_pred[150][0],"Class two is",y_pred[150][1])  #prediction of data in row 150 of test dataset


# ### Saving the Model

# In[22]:


import pickle
with open("pickle_model", "wb") as file:
    pickle.dump(classifier, file)


# In[ ]:




