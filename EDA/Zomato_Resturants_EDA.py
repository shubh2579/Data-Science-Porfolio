#!/usr/bin/env python
# coding: utf-8

# #  ZOMATO BANGALORE RESTURANTS EDA

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


zomato= pd.read_csv('zomato.csv')


# In[4]:


#checking the datatypes and getting to know the columns in the dataset


# In[5]:


zomato.head()


# In[6]:


zomato.describe()


# In[7]:


zomato.info()
zomato.shape


# In[8]:


# cleaning of data to prepare the dataset for vizualization
#From the dataset we can see the columns 'url','address','review list','phone number' 'menu_item' doesnot serve
#us much purpose for this EDA
# so we will be dropping them from the dataset


# In[9]:


zomato=zomato.drop(['url','address','phone','menu_item','reviews_list'],axis=1)
zomato.shape


# In[10]:


zomato.head()


# In[11]:


# will be doing further cleaning of the data
#will see if we can do something about the null values
zomato.info()


# In[12]:


print('rate null values: ',zomato.rate.isnull().sum())
print('location null values: ',zomato.location.isnull().sum())
print('resturant_type null values: ',zomato.rest_type.isnull().sum())
print('dish_liked null values: ',zomato.dish_liked.isnull().sum())
print('cuisines null values: ',zomato.cuisines.isnull().sum())


# In[13]:


zomato=zomato.rename(columns={"approx_cost(for two people)": "approx_cost"})


# In[14]:


print('apprximate cost for 2 people: ',zomato.approx_cost.isnull().sum())


# In[15]:


# for column 'dish_liked' as there are too many null values we will drop it.
# for columns 'rest_type','cuisines','approx_cost' and 'location' we will simply remove the rows with null values.


# In[16]:


zomato=zomato.drop('dish_liked',axis=1)


# In[17]:


zomato.shape


# In[18]:


zomato=zomato.dropna()
zomato.shape


# In[19]:


zomato.info()


# In[20]:


#For the above cleaning of data you can remove the null values for the coloumns mentioned except 'rate'.
# For rate we can assign a mean value in place of null values.
# but for the simplicity of EDA, we drop all the null values.


# In[21]:


zomato.head()


# In[25]:


zomato['rate']=zomato['rate'].str.replace('/5','')


# In[26]:


zomato=zomato[zomato['rate']!='NEW' ]
zomato=zomato[zomato['rate']!='-' ]


# In[27]:


zomato['rate']=zomato['rate'].astype(float)


# In[28]:


zomato.info()


# In[30]:


zomato.online_order.head()


# In[31]:


zomato['online_order']=zomato['online_order'].replace(('Yes','No'),(1,0))


# # Figuring out important facts/answers before deciding to open a resturant in Bangalore

# Which are the top restaurant chains in Bangaluru?

# In[36]:


plt.figure(figsize=(10,7))
zomato_name= zomato.name.value_counts(ascending=True).nlargest(15)
sns.barplot(x=zomato_name,y=zomato_name.index,palette='deep')
plt.title("Most famous restaurants chains in Bangaluru")
plt.xlabel("Number of outlets")
plt.ylabel("Resturant chain")


# As you can see Cafe coffee day,Onesta,Empire Resturant has the most number of outlets in and around bangalore.
# This is rather interesting,we will inspect each of them later.

# How many of the restuarants do not accept online orders?

# In[33]:


online=zomato.online_order.value_counts()
plt.pie(x=online,labels=['Yes','No'],explode=[0.1,0.1],autopct='%1.1f%%')


# As clearly indicated,almost 60 per cent of restaurants in Banglore accepts online orders.
# Nearly 40 per cent of the restaurants do not accept online orders.
# This might be because of the fact that these restaurants cannot afford to pay commission to zomoto for giving them orders online. zomato may want to consider giving them some more benefits if they want to increse the number of restaurants serving their customers online.

# What is the ratio b/w restaurants that provide and do not provide table booking ?

# In[34]:


book=zomato.book_table.value_counts(normalize=True)
print(book)
plt.pie(x=book,labels=['No','Yes'],explode=[0.1,0.1],autopct='%1.2f%%')


# 1. Almost 90 percent of restaurants in Banglore do not provide table booking facility.
# 2. In India you cannot find table booking facility in any average restaurants,usually only five star restaurants provides table booking.

# Which are the most common restaurant type in Banglore?

# Rating Distribution?

# In[60]:


sns.distplot(zomato.rate,bins=15)


# 1. Almost more than 50 percent of restaurants has rating between 3 and 4.
# 2. Restaurants having rating more than 4.5 are very rare.

#  Difference b/w votes of restaurants accepting and not accepting online orders?

# In[59]:


sns.boxplot(x=zomato.online_order,y=zomato.votes,palette='rainbow',linewidth=0.5,width=0.8)


# 1. No doubt about this as Banglore is known as the tech capital of India,people having busy and modern life will prefer Quick Bites.
# 2. We can observe tha Quick Bites type restaurants dominates.

# Common resturant types in Bangalore?

# In[64]:


plt.figure(figsize=(10,7))
r_type= zomato.rest_type.value_counts().nlargest(10)
sns.barplot(x=r_type,y=r_type.index,palette='deep')
plt.title("Major resturant types")
plt.xlabel('count')


# 1. No doubt about this as Banglore is known as the tech capital of India,people having busy and modern life will prefer Quick Bites.
# 2. We can observe tha Quick Bites type restaurants dominates.

# Minimum cost for 2 people ordering online

# In[87]:


plt.figure(figsize=(10,7))
sns.boxplot(y=zomato.approx_cost.value_counts(),hue=online)
plt.xlabel('Ordering Online')


# 1. The median approximate cost for two people is 400 for a single meal.
# 2. 50 percent of restaurants charge between 300 and 650 for single meal for two people.

# Finding best budget Resturants in any location?

# In[92]:


zomato['approx_cost']=zomato['approx_cost'].apply(lambda x: int(x.replace(',','')))


# In[93]:


def budget_rest(location,rest):
    budget=zomato[(zomato.approx_cost <=400)&(zomato.location==location)&(zomato.rate > 4)&(zomato.rest_type ==rest)]
    return (budget.name.unique())


# In[94]:


budget_rest('BTM',"Quick Bites")


# * I have implemented a simple filtering mechanism to find best budget restaurants in any locations in Bangalore.
# * You can pass location and restaurant type as parameteres,function will return name of restaurants.

# Most Foodie areas

# In[97]:


plt.figure(figsize=(10,7))
locations= zomato.location.value_counts().nlargest(10)
sns.barplot(x=locations,y=locations.index,palette='rocket')
plt.title("most foodie areas")
plt.xlabel('count')


# * We can see that BTM,HSR and Koranmangala 5th block has the most number of restaurants.
# * BTM dominates the section by having more than 5000 restaurants.

# Which are the most popular cuisines of Bangalore?

# In[112]:


plt.figure(figsize=(7,8))
cuisine=zomato.cuisines.value_counts().nlargest(15)
sns.barplot(cuisine,cuisine.index,palette='deep')
plt.title('Popoular Cuisines')
plt.xlabel('Count')


# We can observe that North Indian,chinese,South Indian and Biriyani are most common.
# Is this imply the fact that Banglore is more influenced by North Indian culture more than South?

# In[ ]:




