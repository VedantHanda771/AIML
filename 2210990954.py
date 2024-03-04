#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np

identity_matrix = np.identity(3)
print("3x3 Indentity matrix")
print(identity_matrix)


# In[9]:


import numpy as np
null_matrix = np.zeros((3,
                        3))
print("3x3 null matrix")
print(null_matrix)


# In[10]:


one_matrix = np.ones((3,3))
print(one_matrix)


# In[15]:


list = [1,2,3,4,5]
print(list)
arr = np.array(list)
print(arr)


# In[16]:


list1=[1,2,3,4]
list2=[5,6,7,8]
arr = np.array(list1+list2)
print(arr)


# In[21]:


list=[]
for i in range(1, 11):
    list.append(i)
    
print(list)


# In[42]:


my_array = np.array([])
for i in range(1, 11):
    my_array = np.append(my_array,i)
    
my_array = my_array.astype(int)
print(my_array)


# In[43]:


list=[]

for i in range(10):
    user_input = input(f"Enter string #{i + 1}: ")
    list.append(user_input)
    
print(list)


# In[45]:


list=[]

for i in range(10):
    user_input = input(f"Enter string #{i + 1}: ")
    list.append(user_input)
    
    
arr = np.array(list)
print(arr)


# In[49]:


primes = []
number = 2

while len(primes) < 10000:
    is_prime = True

    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            is_prime = False
            break

    if is_prime:
        primes.append(number)

    number += 1

print(primes)


# In[ ]:




