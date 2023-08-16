#!/usr/bin/env python
# coding: utf-8

# # 1. Load the data file

# In[83]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


# In[84]:


data=pd.read_csv('googleplaystore.csv')


# In[85]:


data.head()


# In[86]:


data.tail()


# In[87]:


data.info()


# In[88]:


data.shape


# # 2. Check for null values

# In[89]:


data.isna().sum()


# In[90]:


data.isnull().any()


# # 3. Drop records with nulls in any of the columns

# In[91]:


data=data.dropna()


# In[92]:


data.isnull().any()


# In[93]:


data.shape


# # 4(1). Fix incorrect type and inconsistent formatting

# In[94]:


data["Size"] = [ float(i.split('M')[0]) if 'M' in i else float(0) for i in data["Size"]  ]


# In[95]:


data.head()


# In[96]:


data["Size"] = 1000 * data["Size"]


# In[97]:


data


# # 4(2). Convert string field to numeric

# In[98]:


data.info()


# In[99]:


data["Reviews"] = data["Reviews"].astype(float)


# In[100]:


data.info()


# # 4(3). Convert installs field to integer

# In[101]:


data["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in data["Installs"] ]


# In[102]:


data.head()


# In[103]:


data.info()


# In[104]:


data["Installs"] = data["Installs"].astype(int)


# In[105]:


data.info()


# # 4(4). Convert string to numeric

# In[106]:


data["Price"] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in data["Price"]  ]


# In[107]:


data.head()


# In[108]:


data.info()


# In[109]:


data["Price"] = data["Price"].astype(int)


# In[110]:


data.info()


# # 4(5-A). Sanity checks

# In[111]:


data.shape


# In[112]:


data.drop(data[(data['Reviews'] < 1) & (data['Reviews'] > 5 )].index, inplace = True)


# In[113]:


data.shape


# # 4(5-B).

# In[114]:


data.shape


# In[115]:


data.drop(data[data['Installs'] < data['Reviews'] ].index, inplace = True)


# In[116]:


data.shape


# # 4(5-C).

# In[117]:


data.shape


# In[118]:


data.drop(data[(data['Type'] =='Free') & (data['Price'] > 0 )].index, inplace = True)


# In[119]:


data.shape


# # 5(I). Univariate analysis

# In[120]:


#Boxplot for price
sn.set(rc={'figure.figsize':(12,8)})
sn.boxplot(x=data['Price'])


# Yes, there are some outliers in the price column. There are some apps whose price is more than usual apps on the playstore. 

# # 5(II).

# In[121]:


#Boxplot for Reviews
sn.boxplot(x=data['Reviews'])


# Yes, there are some apps that have high no. of reviews.

# # 5(III).

# In[122]:


#Histogram for Rating
plt.hist(data["Rating"])
plt.xlabel('Rating')
plt.ylabel('Frequency')


# There is a -ve skewness. Some apps seems to have higher ratings than usual.

# # 5(IV).

# In[123]:


#Histogram for Size 
plt.hist(data["Size"])
plt.xlabel('Size')
plt.ylabel('Frequency')


# There is a +ve skewness.

# # 6(1)1. Handling outliers

# As per the above observation of plots, there seems to be some outliers in the Price & Reviews column
# 
# In the Installs column as well

# In[124]:


#Price of $200 and above for an application is expected to be very high
data.loc[data["Price"] > 200].shape[0]


# # 6(1)2.

# In[125]:


#Dropping the Junk apps
data.drop(data[data["Price"] > 200].index, inplace = True)


# In[126]:


data.shape


# # 6(2).

# In[127]:


#Dropping the Star apps as these will skew the analysis,
#checking the shape after dropping
data.drop(data[data["Reviews"] > 2000000].index, inplace = True)


# In[128]:


data.shape


# # 6(3)1.

# In[129]:


#Find out the Percentiles of Installs and decide a threshold as cutoff for outlier
data.quantile([.1, .25, .5, .70, .90, .95, .99], axis = 0)


# In[130]:


#Dropping Installs values that are more than 10000000
data.drop(data[data['Installs'] > 10000000].index, inplace = True)


# In[131]:


data.shape


# # 7(1). Bivariate analysis

# In[132]:


#Scatter plot/jointplot for Rating Vs. Price
sn.scatterplot(x='Rating',y='Price',data=data)


# Plot show a positive linear relationship; as the price of an app increases its rating also increases, which states that the paid apps have the highest of Ratings.

# # 7(2).

# In[133]:


#Scatterplot/jointplot for Rating Vs. Size
sn.scatterplot(x='Rating',y='Size',data=data)


# The plots show a positive linear relationship; as the Size increases the Ratings increases. This stats the heavier apps are rated better.

# # 7(3).

# In[134]:


#Scatterplot for Ratings Vs. Reviews
sn.scatterplot(x='Rating',y='Reviews',data=data)


# The plot shows a positive linear relationship between Ratings and Reviews. More reviews mean better ratings indeed.

# # 7(4).

# In[135]:


#Boxplot for Ratings Vs. Content Rating
sn.boxplot(x='Rating',y='Content Rating',data=data)


# The above plot shows the apps for Everyone is worst rated as it contain the highest number of outliers followed by apps for Mature 17+ and Everyone 10+ along with Teen. The catergory Adults only 18+ is rated better and falls under most liked type

# # 7(5).

# In[136]:


# Boxplot for Ratings Vs. Category
sn.boxplot(x='Rating',y='Category',data=data)


# Events category has the best Ratings out of all other app genres.

# # Data Preprocessing

# # 8(1).

# In[137]:


inp1 = data


# In[138]:


inp1.head()


# Reviews and Installs column still have some relatively high values, before building the linear regression model we need to reduce the skew; columns needs log transformation

# In[139]:


inp1.skew()


# In[140]:


#Apply log transformation to Reviews
reviewsskew = np.log1p(inp1['Reviews'])
inp1['Reviews'] = reviewsskew


# In[141]:


reviewsskew.skew()


# In[142]:


#Apply log transformation to Installs
installsskew = np.log1p(inp1['Installs'])
inp1['Installs']


# In[143]:


installsskew.skew()


# In[144]:


inp1.head()


# # 8(2).

# In[145]:


#Dropping the columns- App, Last Updated, Current Ver, Type, & Andriod Ver as these won't be useful for our model 
inp1.drop(["App","Last Updated","Current Ver","Android Ver","Type"],axis=1,inplace=True)


# In[146]:


inp1.head()


# In[147]:


inp1.shape


# # 8(3).

# In[148]:


#Create a copy of dataframe
inp2 = inp1


# In[149]:


inp2.head()


# As Model does not understand any Catergorical variable hence these need to be converted to numerical
# 
# Dummy Encoding is one way to convert these columns into numerical

# In[150]:


#Get unique values in Column "Category"
inp2.Category.unique()


# In[151]:


inp2.Category = pd.Categorical(inp2.Category)

x = inp2[['Category']]
del inp2['Category']

dummies = pd.get_dummies(x, prefix = 'Category')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[152]:


inp2.shape


# Dummy Encoding on Column "Genres"

# In[153]:


#Get unique values in Column "Genres"
inp2.Genres.unique()

=> Since, There are too many categories under Genres. Hence, we will try to reduce some categories which have very few samples under them and put them under one new common category i.e. "Other".
# In[154]:


#Create an empty list
lists = []
#Get the total genres count & genres count of particular genres count less than 20, append those into the list
for i in inp2.Genres.value_counts().index:
    if inp2.Genres.value_counts()[i]<20:
        lists.append(i)
#Changing the genres which are in the list to other
inp2.Genres = ['Other' if i in lists else i for i in inp2.Genres] 


# In[155]:


inp2["Genres"].unique()


# In[156]:


#Storing the genres column into x variable & delete the genres column from dataframe inp2
inp2.Genres = pd.Categorical(inp2['Genres'])
x = inp2[["Genres"]]
del inp2['Genres']
dummies = pd.get_dummies(x, prefix = 'Genres')
inp2 = pd.concat([inp2,dummies], axis=1)


# In[157]:


inp2.head()


# In[158]:


inp2.shape


# Dummy Encoding on Column "Content Rating"

# In[159]:


#Get unique values in Column "Content Rating"
inp2["Content Rating"].unique()


# In[160]:


#Storing the Content Rating column into x variable & delete the Content Rating column from dataframe inp2
inp2['Content Rating'] = pd.Categorical(inp2['Content Rating'])

x = inp2[['Content Rating']]
del inp2['Content Rating']

dummies = pd.get_dummies(x, prefix = 'Content Rating')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[161]:


inp2.shape


# # 9. and 10. Splitting tha data into training and testing

# In[162]:


#Importing the neccessary libraries from sklearn to split the data and and for model building
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics


# In[163]:


#Creating the variable X and Y which contains the X features as independent features and Y is the target feature
data2=inp2
X = data2.drop("Rating", axis=1)
Y = data2["Rating"]

#Dividing the X and y into test and train data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state = 0)


# # 11. Model Building

# In[164]:


#Create a linear reggression obj by calling the linear reggressor algorithm
regressor=LinearRegression()
regressor.fit(X_train, Y_train)


# In[165]:


R2_score_train_data=round(regressor.score(X_train, Y_train),3)
print("The R2 value of the training set is :{}".format(R2_score_train_data))


# # 12.Make Predictions on Test set and report R2

# In[166]:


Y_pred=regressor.predict(X_test)
R2_score_test_data=metrics.r2_score(Y_test,Y_pred)
R2_score_test_data


# In[169]:


R2_score_test_data=round(regressor.score(X_test,Y_test),3)
print("The R2 value of the training set is : {}".format(R2_score_test_data))


# In[ ]:




