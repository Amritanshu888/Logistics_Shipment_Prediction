#!/usr/bin/env python
# coding: utf-8

# ![image info](https://ineuron.ai/images/ineuron-logo.png)

# # Data Pre-Processing

# #### Import Packages and CSV

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore")
pd.pandas.set_option("display.max_columns", None)


# In[2]:


df = pd.read_csv('train.csv')
df.head()


# ## Data Cleaning

# #### Check Null Values

# In[3]:


##these are the features with nan value
features_with_na=[features for features in df.columns if df[features].isnull().sum()>=1]
for feature in features_with_na:
    print(feature,np.round(df[feature].isnull().mean()*100,5), '% missing values')


# **What does NaN mean?**
# 
# **Ans:** Missing Values in a DataFrame
# 
# **Why might a value be missing?** 
# 
# **Ans:** When considering the potential impact of the missing data on the Dataset, it is important to consider the underlying reasons for why the data are missing
# Missing data are of three categories:
# * Missing completely at random (MCAR)
# * Missing at random (MAR).
# * Missing not at random (MNAR). 
# 
# Here it is MCAR type of missing values in dataset which are Missing completely at Random. 
# 
# When we make this assumption, we are assuming the missing data is completely unrelated to the other information in the data.

# #### Checking duplicated values
# 

# In[4]:


df.duplicated().sum()


#  * There are no duplicates values in the data

# #### Seperating Features
# 

# In[5]:


target_feature = 'Cost'

# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
numeric_features.remove(target_feature)
non_numeric_features = [feature for feature in df.columns if df[feature].dtype == 'O']

print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} Non Numeric features : {}'.format(len(non_numeric_features), non_numeric_features))


# Imputing null values with median to check the multicollinearity

# In[6]:


df1 = df.copy()
for i in numeric_features:
    df1[i].fillna(df1[i].median(), inplace=True)


# ### Multicollinearity Check
# #### Variance Inflation Factor (VIF)

# * Multicollinearity occurs when there are two or more independent variables in a multiple regression model, which have a high correlation among themselves. When some features are highly correlated.
# * Multicollinearity can be detected using various techniques, one such technique being the Variance Inflation Factor(VIF).

# In[7]:


import statsmodels.api as sm

for i in range(0, len(numeric_features)):
    x = df1.loc[:, df1.columns == numeric_features[i]]
    y = df1.Cost
    model = sm.OLS(x,y)
    results = model.fit()
    rsq = results.rsquared
    vif = round(1 / (1 - rsq), 2)
    print(
        "Variance Inflation Factor of {} column is {} \n".format(
            numeric_features[i], vif)
        )


# ### Note:  All the columns has VIF around 1, so there is not severe multicollinearity

# In[8]:


for i in numeric_features:
        plt.figure(figsize=(10,6))
        sns.set_style('ticks')
        ax = sns.boxplot(df1[i])


# In[9]:


plt.figure(figsize=(15, 15))
plt.suptitle('Data Distribution for Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=df[numeric_features[i]],shade=True, color='b')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()


# In[10]:


df[numeric_features].skew()


# **Insighs**
# 
#  - From EDA we have seen that the `Price Of Sculpture ` and `Weight` were extremely positively skewed may be because outliers
# 
#  - So we can handle this type of case using Power Transformer or log transformer
#  
#  - `Width` and `Base Shipping Price` too are slighty positively skewed but not severe as shown by skewness coefficient.
# 

# #### Checking the power transformer on outlier columns

# In[11]:


outlier_feature = ['Price Of Sculpture', 'Weight']


# In[12]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(standardize=False)

outlier_data = df1[outlier_feature]


# In[13]:


df1[outlier_feature] = pt.fit_transform(df1[outlier_feature])


# In[14]:


df_outlier = pd.DataFrame(outlier_data, columns=outlier_feature)


# In[15]:


plt.figure(figsize=(15, 15))
plt.suptitle('Data Distribution for Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(outlier_feature)):
    plt.subplot(5, 3, i+1)
    sns.histplot(x=df1[outlier_feature[i]], color='b')
    plt.xlabel(outlier_feature[i])
    plt.tight_layout()


# In[16]:


df1[outlier_feature].skew()


# **Insights**
#  - Power transformer does a better job at overcoming skewness and handling outliers

# **Why outliers?**
# * Data Entry error : Human error. 
# * Measurement error: Instrument error. 
# * Natural error: it will be Real life data.
# * Intentional error: People give wrong inputs
# 
# **Impact of Outliers ?**
# * Ouliers can very high impact on few Machine learning models.
# * Can Cause bias in the output.

# In[17]:


for i in numeric_features:
    plt.figure(figsize=(10,6))
    sns.set_style('ticks')
    ax = sns.boxplot(df1[i])


# #### Removing outliers from Target Column
# * Because most model would perform better on noiseless data, as Outlier might skew the findings of your model in one direction

# In[18]:


plt.subplots(figsize=(14,7))
sns.histplot(df['Cost'], bins=200, kde=True, color = 'b')
plt.title("Cost Distribution", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=12)
plt.xlabel("Cost price in millions", weight="bold", fontsize=12)
plt.show()


# Need to apply transformation to target column to make it normally distributed. we use 1p to avoid `inf` values arise dued to `0` in the target column

# In[19]:


df1['Cost'] = np.log1p(df1['Cost'])


# In[20]:


plt.hist(df1['Cost'])


# Data looks good after the log transformation but not Normally distributed.

# #### Feature Creation for Date Columns

# In[21]:


# Convert object datatype to datetime
df['Scheduled Date'] = pd.to_datetime(df['Scheduled Date'])
df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])


# In[22]:


df['Month'] = pd.to_datetime(df['Scheduled Date']).dt.month
df['Year'] = pd.to_datetime(df['Scheduled Date']).dt.year

numeric_features.append('Month')
numeric_features.append('Year')


# #### Drop Unneccessary column

# Drop `Customer Id`, `Artist Name` and `Customer Location` columns having huge number of unique values.
# 
# `Scheduled Date` and `Delivery Date` can be dropped since the required features are extracted form it

# In[23]:


to_drop_columns = ['Customer Id', 'Artist Name', 'Customer Location', 'Scheduled Date', 'Delivery Date']


# In[24]:


df.drop(columns=to_drop_columns, inplace=True)


# ### Model Building

# In[25]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, LabelEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


# ### Split X and y

# In[26]:


X = df.drop(columns = ['Cost'], axis=1)
y = np.abs(df['Cost'])


# In[27]:


y = np.log1p(y)


# ## Train Test Split
# 
# * The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model.
# 
# * It is a fast and easy procedure to perform, the results of which allow you to compare the performance of machine learning algorithms.

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Feature Encoding and Scaling

# **Selecting Numerical and categorical features for feature scaling**

# In[29]:


numeric_features = [x for x in numeric_features if x not in outlier_feature]
categorical_features = [x for x in non_numeric_features if x not in to_drop_columns]

(numeric_features), (categorical_features), (outlier_feature)


# * **Mean imputer:** Since we handeled outlier by powertransformer for the continuous features the mean and median of the features are nearly same. When there are no outliers mean performs an a better imputer.
# * **Standardscaler:** As the features distribution are nearly normal we use standard scaler.
# * **Power Transformer:** Since avg_glucose_level are skewed we use Power Transformer on them

# In[30]:


numeric_features_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

categorical_features_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot_encoder', OneHotEncoder()),
]
)

outliers_features_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('transformer', PowerTransformer(standardize=True))
])

preprocessor = ColumnTransformer(
    [
        ("Numeric Pipeline",numeric_features_pipeline, numeric_features),
        ("Categorical Features Pipeline",categorical_features_pipeline, categorical_features),
        ("Outliers Features Pipeline", outliers_features_pipeline, outlier_feature)
]
)


# In[31]:


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# ## Model Selection
# * **Here should understand the Various Classification models with default values from these models we can choose top 4 with Highest Accuracy score and proceed with HyperParameter Tuning**

# In[33]:


# functions which takes true and predicted values to calculate metrics
def evaluate_reg(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# In[34]:


# function which can evaluate models and return a report 
def evaluate_models(X_train, X_test, y_train, y_test, models):
    '''
    This function takes in X and y and models dictionary as input
    It splits the data into Train Test split
    Iterates through the given model dictionary and evaluates the metrics
    Returns: Dataframe which contains report of all models metrics with cost
    '''    
    models_list = []
    r2_list = []
    
    for i in range(len(list(models))):
        model = list(models.values())[i]
        model.fit(X_train, y_train) # Train model

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate Train and Test dataset
        model_train_mae , model_train_rmse, model_train_r2 = evaluate_reg(y_train, y_train_pred)

        model_test_mae , model_test_rmse, model_test_r2 = evaluate_reg(y_test, y_test_pred)


        print(list(models.keys())[i])
        models_list.append(list(models.keys())[i])

        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')

        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
        r2_list.append(model_test_r2)

        print('='*35)
        print('\n')
        
    report=pd.DataFrame(list(zip(models_list, r2_list)), columns=['Model Name', 'r2_score']).sort_values(by=['r2_score'], ascending=False)
        
    return report


# In[35]:


base_report = evaluate_models(X_train, X_test, y_train, y_test, models)


# In[32]:


# Initialize models required for model
models = {
    "Linear Regression": LinearRegression(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(), 
    "CatBoosting Regressor": CatBoostRegressor(verbose=False, max_depth=5),
    "AdaBoost Regressor": AdaBoostRegressor(), 
    "SVR": SVR()
}


# In[36]:


base_report


# #### Hyperparameter Tuning

# In[37]:


#Initialize few parameter for Hyperparamter tuning
rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20, 30],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]}

cat_params = {"learning_rate": [0.1, 0.01, 0.6, 0.5],
              "max_depth": [4, 5, 6, 8, 12]}


# In[38]:


# Models list for Hyperparameter tuning
randomcv_models = [("RF", RandomForestRegressor(), rf_params),
                   ('XGBoost', XGBRegressor(), xgboost_params),
                   ('CatBoost', CatBoostRegressor(verbose=False), cat_params)
                   ]


# In[39]:


from sklearn.model_selection import RandomizedSearchCV

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])


# In[40]:


# reInitialize models with best parameters
models = {
    "Random Forest Regressor": RandomForestRegressor(**model_param['RF'], n_jobs=-1),
    "XGBRegressor": XGBRegressor(**model_param['XGBoost'], n_jobs=-1), 
     "CatBoosting Regressor": CatBoostRegressor(**model_param['CatBoost'],verbose=False)
}


# In[41]:


retrained_report = evaluate_models(X_train, X_test, y_train, y_test, models)


# In[42]:


retrained_report


# ### Best Model is CATBoost Regressor with 97.17% Score

# In[ ]:




