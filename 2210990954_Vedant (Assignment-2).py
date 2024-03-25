#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

data=pd.read_csv("C:\\Users\\vedant\\OneDrive\\Desktop\\AI-ML\\insurance.csv")
data


# ### 1. Display Top 5 Rows of The Dataset

# In[2]:


data.head(5)


# ### 2. Check Last 5 Rows of The Dataset

# In[3]:


data.tail(5)


# ### 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[4]:


print ("Number of Rows" , data.shape[0])
print ("Number of Columns" , data.shape[1])


# ### 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[5]:


data.info()


# ### 5.Check Null Values In The Dataset

# In[6]:


data.isnull()


# ### 6. Get Overall Statistics About The Dataset

# In[7]:


data.describe()


# ### 7. Covert Columns From String ['sex' ,'smoker','region' ] To Numerical Values

# In[3]:


sex_mapping = {'male': 0, 'female': 1}
smoker_mapping = {'no': 0, 'yes': 1}
region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}

data['sex'] = data['sex'].replace(sex_mapping)
data['smoker'] = data['smoker'].replace(smoker_mapping)
data['region'] = data['region'].replace(region_mapping)
data


# ### 8. Store Feature Matrix In X and Response(Target) In Vector y

# In[4]:


X = data.drop('charges', axis=1)
y = data['charges']
X


# ### 9. Train/Test split

# 1. Split data into two part : a training set and a testing set
# 2. Train the model(s) on training set
# 3. Test the Model(s) on Testing set

# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train


# ### 10. Import the models

# In[6]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ### Training the Models

# In[7]:


lr = LinearRegression()
lr.fit(X_train, y_train)

svm = SVR()
svm.fit(X_train, y_train)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

gr = GradientBoostingRegressor()
gr.fit(X_train, y_train)


# ### Check LR Model's Prediction Performance

# In[8]:


lr.score(X_train, y_train)
lr.score(X_test, y_test)


# ### Check SVM Model's Prediction Performance

# In[9]:


svm.score(X_train, y_train)
svm.score(X_test, y_test)


# ### Check RF Model's Prediction Performance

# In[10]:


rf.score(X_train, y_train)
rf.score(X_test, y_test)


# ### Check GR Model's Prediction Performance

# In[11]:


gr.score(X_train, y_train)
gr.score(X_test, y_test)


# ### 12. Prediction on Test Data

# In[12]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)

y_pred4


# ### 13. Compare Performance Visually

# In[22]:


data = pd.DataFrame({'Actual' : y_test, 'Lr' : y_pred1 , 'SVM' : y_pred2 , 'RF' : y_pred3 , 'GR' : y_pred4 })
data


# ### Plotting Graph

# In[23]:


import matplotlib.pyplot as plt

plt.subplot(221)
plt.plot(data['Actual'].iloc[0:11] , label='Actual')
plt.plot(data['Lr'].iloc[0:11], label = "Lr")
plt.legend()

plt.subplot(222)
plt.plot(data['Actual'].iloc[0:11] , label='Actual')
plt.plot(data['SVM'].iloc[0:11], label = "SVM")
plt.legend()

plt.subplot(223)
plt.plot(data['Actual'].iloc[0:11] , label='Actual')
plt.plot(data['RF'].iloc[0:11], label = "RF")
plt.legend()

plt.subplot(224)
plt.plot(data['Actual'].iloc[0:11] , label='Actual')
plt.plot(data['GR'].iloc[0:11], label = "GR")
plt.legend()


# ### Predict Charges for new customer

# In[20]:


data = {'age' : 40,
        'sex' : 1,
        'bmi' : 40.30,
        'children' : 4,
        'smoker' : 1,
        'region' : 2}

df = pd.DataFrame(data, index=[0])
df
new_pred = gr.predict(df)
new_pred


# In[24]:


new_pred = gr.predict(df)
print("Medical Insurance cost for New Customer is : ", new_pred[0])


# ### Save model using JobLib

# In[26]:


gr=GradientBoostingRegressor()
gr.fit(X,y)
X

import joblib

joblib.dump(gr,'model_joblib_gr')
model=joblib.load('model_joblib_gr')
model.predict(df)


# ### GUI

# In[27]:


from tkinter import*
import joblib


# In[29]:


def show_entry():

    p1=float(e1.get())
    p2=float(e2.get())
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    p6=float(e6.get())
    model=joblib.load('model_joblib_gr')
    result=model.predict([[p1,p2,p3,p4,p5,p6]])

    Label(master, text="Insurance Cost").grid(row=7)
    Label(master, text=result).grid(row=8)
 


master=Tk()
master.title("Insurance Cost Prediction")
label=Label(master,text="Insurance Cost Prediction",bg="black",fg="white").grid(row=0,columnspan=2)

Label(master,text="Enter your age").grid(row=1)
Label(master,text="MALE OR FEMALE [1/0]").grid(row=2)
Label(master,text="Enter your BMI values").grid(row=3)
Label(master,text="Enter Number of Children").grid(row=4)
Label(master,text="Smoker Yes/No [1/0]").grid(row=5)
Label(master,text="Region [1-4]").grid(row=6)


e1=Entry(master)
e2=Entry(master)
e3=Entry(master)
e4=Entry(master)
e5=Entry(master)
e6=Entry(master)



e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)



 
Button(master,text="Predict",command=show_entry).grid()             

master.mainloop()

