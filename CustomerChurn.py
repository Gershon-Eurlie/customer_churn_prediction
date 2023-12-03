#GITHUB:  https://github.com/Gershon-Eurlie/
#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('datasetCHURN.csv')
df.sample(10)


# ### Data Exploration

#  In machine learning, customerID is not needed. So I will drop that column...
# 

# In[5]:


df.drop('customerID', axis='columns', inplace=True) # This is drops customerID and updates the dataframe
df.dtypes


# As shown above the column 'customerID' has been dropped. All column datatypes are shown here.

# In[6]:


df.MonthlyCharges.values


# In[7]:


df.TotalCharges.values


# In[8]:


pd.to_numeric(df.TotalCharges,errors='coerce')


# In[9]:


pd.to_numeric(df.TotalCharges, errors='coerce').isnull()


# In[10]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[11]:


df.shape


# In[12]:


df.iloc[488].TotalCharges


# In[13]:


df[df.TotalCharges!=' '].shape


# In[ ]:





# ###  Removing Rows with Space in TotalCharges Columns

# In[14]:


data = df[df.TotalCharges!=' ']


# In[15]:


data.shape


# In[16]:


data.dtypes


# In[17]:


data.TotalCharges = pd.to_numeric(data.TotalCharges)


# In[18]:


data.TotalCharges.values


# In[19]:


data[data.Churn=='No']


# In[ ]:





# In[ ]:





# ## Data Visualizations

# In[20]:


tenure_churn_no = data[data.Churn=='No'].tenure
tenure_churn_yes = data[data.Churn=='Yes'].tenure

plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.title("Visual of Customer Churn Predicion")

#blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
#blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100]

plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['blue','red'],label=['Churn=Yes', 'Churn=No'])
plt.legend()


# In[ ]:





# In[21]:


mc_churn_no = data[data.Churn=='No'].MonthlyCharges
mc_churn_yes = data[data.Churn=='Yes'].MonthlyCharges

plt.xlabel('Monthly Charges')
plt.ylabel('Number of Customers')
plt.title('Churn Predition Visual')

plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['blue','red'], label=['Churn=Yes', 'Churn=No'])
plt.legend()


# ###### Some of the columns are Yes, No, etc. Let's print unique values in object colums see data values

# In[22]:


def printUniqueColumnValue(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column}: {df[column].unique()}')


# In[23]:


printUniqueColumnValue(data)


# #### some of the comuns have no internet service or no phone service, that can be replaced with a simple No

# In[24]:


data.replace('No internet service', 'No',  inplace=True)
data.replace('No phone service', 'No',  inplace=True)


# In[25]:


printUniqueColumnValue(data)


# ### Converting all yes and no to 1 and 0

# In[26]:


yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    data[col].replace({'Yes': 1,'No': 0},inplace=True)


# In[27]:


for col in data:
    print(f'{col}: {data[col].unique()}')


# In[28]:


data['gender'].replace({'Female':1, 'Male':0}, inplace=True)


# In[29]:


data.gender.unique()


# #### Enconding for the categorical data

# In[30]:


data1 = pd.get_dummies(data=data, columns=['InternetService', 'Contract', 'PaymentMethod' ])
data1.columns


# In[31]:


data1.sample(10)


# In[32]:


data1.dtypes


# In[33]:


cols_to_scale = ['tenure', 'MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data1[cols_to_scale] = scaler.fit_transform(data1[cols_to_scale])


# In[34]:


for col in data1:
    print(f'{col}: {data1[col].unique()}')


# ## Training the split dataset

# In[35]:


X = data1.drop('Churn', axis='columns')
y = data['Churn']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=5)


# In[36]:


X_train.shape


# In[37]:


X_test.shape


# In[38]:


X_train[ :10]


# In[39]:


len(X_train.columns)


# ## Building an Artificial Neural Network (ANN) Model in tensorflow/Keras

# In[41]:


import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)


# In[43]:


model.evaluate(X_test, y_test)


# In[44]:


yp = model.predict(X_test)
yp[:5]


# In[ ]:





# In[45]:


y_predict = []
for element in yp:
    if element > 0.5:
        y_predict.append(1)
    else:
        y_predict.append(0)
            


# In[46]:


y_predict[:10]


# In[50]:


y_test[:10]


# In[51]:


from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,y_predict))


# In[53]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predict)


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[55]:


y_test.shape # looking at the dimension of the test data


# ## ACCURACY

# In[57]:


round((881+203)/(881+203+118+205),2)


# ## Precision for Customers who did not Churn

# In[58]:


round((881/(881+205)),2)


# ## Precision for Customers who Churned

# In[59]:


round((203/(203+118)),2)


# ## Recall

# In[60]:


round((881/(881+118)),2)


# In[61]:


round((203/(203+205)),2)


# In[ ]:




