#!/usr/bin/env python
# coding: utf-8

# ![image info](https://ineuron.ai/images/ineuron-logo.png)`

# # Shipment Pricing Prediction
# **Life cycle of Machine learning Project**
# 
# * Understanding the Problem Statement
# * Data Collection
# * Data Cleaning
# * Exploratory data analysis
# * Data Pre-Processing
# * Model Training 
# * Choose best model

# ## 1) Problem statement.
# 
# - You work for a company that sells sculptures that are acquired from various artists around the world. Your task is to predict the cost required to ship these sculptures to customers based on the information provided in the dataset.

# ### 2)  Import Data 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# **Import the CSV Data as Pandas DataFrame**

# In[2]:


df = pd.read_csv('train.csv')


# **Show Top 5 Records**

# In[3]:


df.head()


# **Shape of the dataset**

# In[4]:


df.shape


# **Summary of the dataset**

# In[5]:


# Display summary statistics for a dataframe
df.describe()


# ### Statistical Inferences
# - Median Price Of Sculpture is 1192.
# - 25% of the population is Price Of Sculpture below 5
# - Average Artist Reputation of the population is 0.46
# 
# * The difference between 75th percentile and Max also suggests the skewness 
# 
# * We can confirm the insights we got from data distribution, as skewness is more for Height, Width, and Base Shipping Price, and skewness of Weight, Price of Sculpture because of Outliers 

# **Check Datatypes in the dataset**

# In[6]:


df.dtypes


# ## 3. EXPLORING DATA

# **Most of the features seems to be of object data type, Now let's seperate Categorical and numerical columns**

# In[7]:


target_feature = 'Cost'


# In[8]:


# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))


# ## Univariate Analysis

# * The term univariate analysis refers to the analysis of one variable prefix “uni” means “one.” The purpose of univariate analysis is to understand the distribution of values for a single variable.

# ### Numerical Features

# In[9]:


plt.figure(figsize=(15, 10))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numeric_features)):
    plt.subplot(4, 2, i+1)
    sns.kdeplot(x=df[numeric_features[i]], color='blue')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()


# <font size=3>**Report**</font> 
# 
# * <font size=3> `Height`, `Width` and `Base Shipping price` are positively skewed</font> 
# 
# * <font size=3>`Weight`, `Price of Sculpture` has many outliers</font> 

# ### Categorical Features

# In[10]:


df.describe()


# In[11]:


categorical_columns = []
for i in categorical_features:
    print(f'{i}: {len(df[i].unique())}')
    if len(df[i].unique()) < 10:
        categorical_columns.append(i)
        


# * `Customer Id`, `Artist Name`, `Customer location` have 6500 unique values, so they can be dropped
# 
# * `Scheduled date` and `Delivery date` needs feature engineering

# In[12]:


categorical_columns


# In[13]:


# categorical columns
plt.figure(figsize=(15, 20))
plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
for i in range(0, len(categorical_columns)):
    plt.subplot(4, 2, i+1)
    sns.countplot(x=df[categorical_columns[i]])
    plt.xlabel(categorical_columns[i])
    plt.xticks(rotation=45)
    plt.tight_layout()


# **Insights** 
# 
# * `Material column` has seven unique value, which are almost equally distributed</font> 
# 
# * There are 6 bi-variate categorical columns</font> 

# <font size=3>**Checking the Null Values in dataset**</font> 
# 

# In[14]:


df.isnull().sum()


# **Insights** 
# 
# - There are 7 columns which has null values

# ## Multivariate Analysis
# * Multivariate analysis is the analysis of more than one variable.

# <font size=4>**Correlation in Numerical Features**</font> 

# In[15]:


df[(list(df.columns)[1:])].corr()


# In[16]:


plt.figure(figsize = (15,10))
sns.heatmap(df.corr(), cmap="CMRmap_r", annot=True)
plt.show()


# **Insights** 
# 
# - There is a high correlation between `Height`-`Width`, and `Weight`-`Price of Sculpture` 
# 

# ### Relationship between target and numerical columns 

# In[17]:


fig = plt.figure(figsize=(12, 24))

for i in range(0, len(numeric_features)):
    ax = plt.subplot(6, 2, i+1)
    sns.scatterplot(data= df ,y=target_feature, x=numeric_features[i], color='b')
    if df[numeric_features[i]].max() > 30000:
        plt.xlim(-1000,30000)
    if df[numeric_features[i]].max() > 3000000:
        plt.xlim(-1000,3000000)
    plt.ylim(-1000, 300000)
    plt.tight_layout()


# **Insights** 
# 
# - We can observe that there seems to be very low linear relationship between the independent and dependent features
# 
# - There seems to be linear relationship between Price of sculpture and the target column

# ## Visualizing Independent columns

# <font size=3>**Target Feature**</font> 
# 

# In[18]:


plt.subplots(figsize=(14,7))
sns.histplot(df[target_feature], bins=200, kde=True, color = 'b')
plt.title("Cost Distribution", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=12)
plt.xlabel("Cost price in millions", weight="bold", fontsize=12)
plt.xlim(-50000,300000)
plt.show()


# In[19]:


plt.subplots(figsize=(14,7))
sns.boxplot(df[target_feature])
plt.title("Cost Distribution", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=12)
plt.xlabel("Cost price in millions", weight="bold", fontsize=12)
plt.xlim(-50000,300000);


# **Insights** 
# 
# - There are outliers in the target feature and we need to transform

# <font size=3>**How shipment type is affecting cost of shipment?**</font> 
# 

# In[20]:


international = df.groupby('International')[target_feature].median()
international.to_frame()

plt.subplots(figsize=(14,7))
sns.barplot(x=international.index, y=international.values, ec = "black", palette="Set2")
plt.title("Shipment Type vs Median Cost", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mean Cost", weight="bold", fontsize=15)
plt.xlabel("Shipment Type", weight="bold", fontsize=12)
plt.show()


# In[21]:


df['International'].value_counts()


# In[22]:


international_shipment_median = df[df['International'] == 'Yes']['Cost'].median()
domestic_shipment_median = df[df['International'] == 'No']['Cost'].median()

print(f"The difference between the median shipment price of International and domestic is {international_shipment_median-domestic_shipment_median}")


# **Insights**
# 
# - There is only a difference of 43 in the shipping cost between International and domestic, so it won't affect much at the prediction
# 
# - There are much more domestic shipement than international

# <font size=3>**How Express Shipment is affecting cost of shipment?**</font> 
# 

# In[23]:


express_ship = df.groupby('Express Shipment')[target_feature].median()
express_ship.to_frame()

plt.subplots(figsize=(14,7))
sns.barplot(x=express_ship.index, y=express_ship.values, ec = "black", palette="Set1")
plt.title("Express Shipment vs Median Cost", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mean Cost", weight="bold", fontsize=15)
plt.xlabel("Express Shipment", weight="bold", fontsize=12)
plt.show()


# In[24]:


df['Express Shipment'].value_counts()


# In[25]:


express_shipment_median = df[df['Express Shipment'] == 'Yes']['Cost'].median()
normal_shipment_median = df[df['Express Shipment'] == 'No']['Cost'].median()

print(f"The difference between the median shipment price of express shipment and normal shipment is {express_shipment_median-normal_shipment_median}")


# **Insights**
# 
# - In express shipment also, there is only a difference of 83 in the shipping cost between International and domestic, so it won't affect much at the prediction
# 
# - It's obvious that there are more normal delivery than express 

# <font size=3>**How Installation included or not is affecting cost of shipment?**</font> 
# 

# In[26]:


installation = df.groupby('Installation Included')[target_feature].median()
installation.to_frame()

plt.subplots(figsize=(14,7))
sns.barplot(x=installation.index, y=installation.values, ec = "black", palette="Set1")
plt.title("Installation Included vs Median Cost", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mean Cost", weight="bold", fontsize=15)
plt.xlabel("Installation Included", weight="bold", fontsize=12)
plt.show()


# In[27]:


df['Installation Included'].value_counts()


# In[28]:


installation_needed_median = df[df['Installation Included'] == 'Yes']['Cost'].median()
installation_not_needed_median = df[df['Installation Included'] == 'No']['Cost'].median()

print(f"The difference between the median shipment price of Installation Included and not is {installation_needed_median-installation_not_needed_median}")


# **Insights** 
# 
# - There is only a difference of 53 which won't make much a difference at the time of prediction
# 
# - We can see that the charges for not installation is less and values are more, which says that a business should try to take contract which requires installation

# <font size=3>**How Fragile Cost is affecting cost of shipment?**</font> 
# 

# In[29]:


fragile = df.groupby('Fragile')[target_feature].median()
fragile.to_frame()

plt.subplots(figsize=(14,7))
sns.barplot(x=fragile.index, y=fragile.values, ec = "black", palette="Set1")
plt.title("Fragile Cost vs Median Cost", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mean Cost", weight="bold", fontsize=15)
plt.xlabel("Was there any Fragile Cost?", weight="bold", fontsize=12)
plt.show()


# In[30]:


fragile_median = df[df['Fragile'] == 'Yes']['Cost'].median()
fragile_not_median = df[df['Fragile'] == 'No']['Cost'].median()

print(f"The difference between the median shipment price of Fragile cost and non-fragile cost is {fragile_median-fragile_not_median}")


# **Insights**
# 
# - There is only a difference of -80 which says that the median cost is more if there is no fragile cost and because the number is small it won't make much of a difference at the time of training model

# <font size=3>**How delivery location is affecting cost of shipment?**</font> 
# 

# In[31]:


remote = df.groupby('Remote Location')[target_feature].median()
remote.to_frame()

plt.subplots(figsize=(14,7))
sns.barplot(x=remote.index, y=remote.values, ec = "black", palette="Set2")
plt.title("Remote Location vs Median Cost", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mean Cost", weight="bold", fontsize=15)
plt.xlabel("Was the shipment outside city?", weight="bold", fontsize=12)
plt.show()


# In[32]:


remote_median = df[df['Remote Location'] == 'Yes']['Cost'].median()
not_remote_median = df[df['Remote Location'] == 'No']['Cost'].median()

print(f"The difference between the median shipment price of remote shipment and non-remote shipment cost is {remote_median-not_remote_median}")


# **Insights** 
# 
# - There is only a difference of 27 which won't make much a difference at the time of prediction
# 

# <font size=3>**How Customer Financial condition is affecting cost of shipment?**</font> 
# 

# In[33]:


cust_info = df.groupby('Customer Information')[target_feature].median()
cust_info.to_frame()

plt.subplots(figsize=(14,7))
sns.barplot(x=cust_info.index, y=cust_info.values, ec = "black", palette="Set2")
plt.title("Customer Information vs Median Cost", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mean Cost", weight="bold", fontsize=15)
plt.xlabel("Customer Financial Condition", weight="bold", fontsize=12)
plt.show()


# In[34]:


customer_info_wealthy_median = df[df['Customer Information'] == 'Wealthy']['Cost'].median()
customer_info_working_class_median = df[df['Customer Information'] == 'Working Class']['Cost'].median()


print(f"The difference between the median shipment price of remote shipment and non-remote shipment cost is {customer_info_wealthy_median-customer_info_working_class_median}")


# **Insights**
# 
# - There is only a difference of 66 which won't make much a difference at the time of prediction
# 

# <font size=3>**Report**</font> 
# 
# <font size=3>After understanding relationship between the target column and categorical columns, that there doesn't seem to be much pattern, which model can find in the categorical columns also</font>

# ## Visualizing date columns
# 

# In[35]:


df['Scheduled Date'] = pd.to_datetime(df['Scheduled Date'])
df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])


# In[36]:


df['year'] = pd.DatetimeIndex(df['Scheduled Date']).year
df['month'] = pd.DatetimeIndex(df['Scheduled Date']).month


# In[37]:


df.head()


# In[47]:


x = df.groupby(["year", "month"])["Cost"].mean()
df_wide = x.unstack().T
df_wide.head()


# In[52]:


df_wide[2015]


# In[66]:


for i in df_wide.columns:
    plt.subplots(figsize=(14,7))
    sns.barplot(y=df_wide[i],x=df_wide.index,ec = "black", palette="Set2")
    plt.title(str(i) + " monthly mean Shipment cost", weight="bold",fontsize=20, pad=20)
    plt.ylabel("mean", weight="bold", fontsize=11)
    plt.xlabel(i, weight="bold", fontsize=12)
    plt.show()


# **Insights**
#  - Monthly distribution of every year is different
#  - For 2015, 2016 and 2018 There were more cost at the start and end of the year.
#  - For 2017 and 2019 cost were high mid year.

# In[43]:


plt.subplots(figsize=(14,7))
sns.countplot(x=df.year,ec = "black", palette="Set2")
plt.title("Year vs Shipment Orders", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=11)
plt.xlabel("Year", weight="bold", fontsize=12)
plt.show()


# In[42]:


plt.subplots(figsize=(14,7))
sns.countplot(x=df.month,ec = "black", palette="Set2")
plt.title("Customer Information vs Median Cost", weight="bold",fontsize=20, pad=20)
plt.ylabel("Mean Cost", weight="bold", fontsize=15)
plt.xlabel("Customer Financial Condition", weight="bold", fontsize=12)
plt.show()


# # Final Report

# 
# * The `Cost` column is the target to predict.
# * The target variable here is continuous.
# * There are outliers in some columns we have to remove outliers.
# * `date` column should be configured to extract `year` and `month`.
# * Null values in `Artist Reputation`, `Height`, `Width`, `Weight`, `Material`, `Transport`, `Remote Location` needs to be handled

# In[ ]:




