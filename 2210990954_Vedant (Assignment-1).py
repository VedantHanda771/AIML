#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

df = pd.read_csv("C:\\Users\\vedant\\OneDrive\\Desktop\\AI-ML\\ST-2 Supervised learning algorithm\\add.csv")
df


# In[5]:


import matplotlib.pyplot as plt
plt.scatter(df['x'], df['sum'])


# In[6]:


import matplotlib.pyplot as plt
plt.scatter(df['y'], df['sum'])


# ### 1. Store feature matrix in X and response (target) in vector Y

# In[7]:


x = df[['x', 'y']]
y = df['sum']

x


y


# ### Train / Test Split

# 1. Split data into 2 part :- a training set & a testing set
# 2. Train the model on training set
# 3. Train the model on testing set

# In[8]:


from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(
x, y , test_size = 0.33, random_state = 8)

y_test
x_train


# ### Import and Train the Model

# In[9]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

# when we are creating the model, we use fit method


# ### Check Model's Prediction Performance

# In[10]:


model.score(x_train, y_train)
model.score(x_test, y_test) 

# we use score method to predict the performance of training and testing variables


# ### Comparing The Results

# In[11]:


y_pred = model.predict(x_test)
y_pred

df = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
df


# ### Prediction

# In[12]:


model.predict([[10,20]])
model.predict([[100.2,210.3]])


# ### GUI

# In[13]:


from tkinter import*
import joblib


# In[14]:


dx={'x':32,'y':55}
dx=pd.DataFrame(dx,index=[0])
dx
new_pred=model.predict(dx)

new_pred


# ### Save model using JobLib

# In[15]:


import joblib as jb

jb.dump(model,'model_joblib_model')

lr=jb.load('model_joblib_model')

lr.predict(dx)


# In[ ]:


def show_entry():

    p1=float(e1.get())
    p2=float(e2.get())
    
    lr=jb.load('model_joblib_model')
    result=lr.predict([[p1,p2]])

    Label(master, text="Addition").grid(row=7)
    Label(master, text=result).grid(row=8)


master=Tk()
master.title("ADDition Prediction")
label=Label(master,text="addition Prediction",bg="black",fg="white").grid(row=0,columnspan=2)

Label(master,text="Enter first value").grid(row=1)
Label(master,text="enter second value").grid(row=2)


e1=Entry(master)
e2=Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)


Button(master,text="Addition",command=show_entry).grid()             

master.mainloop()

