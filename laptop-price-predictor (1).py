#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[30]:


df = pd.read_csv('laptop_data.csv')
df



# In[31]:


df.head()


# In[32]:


df.shape


# In[33]:


df.info()


# In[34]:


df.duplicated().sum()


# In[35]:


df.isnull().sum()


# In[36]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[37]:


df.head()


# In[38]:


df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')


# In[39]:


df.head()


# In[40]:


df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')


# In[41]:


df.info()


# In[42]:


import seaborn as sns


# In[43]:


sns.distplot(df['Price'])


# In[44]:


df['Company'].value_counts().plot(kind='bar')


# In[45]:


sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[46]:


df['TypeName'].value_counts().plot(kind='bar')


# In[47]:


sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[48]:


sns.distplot(df['Inches'])


# In[49]:


sns.scatterplot(x=df['Inches'],y=df['Price'])


# In[50]:


df['ScreenResolution'].value_counts()


# In[51]:


df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[52]:


df.sample(5)


# In[53]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[54]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])


# In[55]:


df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[56]:


df.head()


# In[57]:


df['Ips'].value_counts().plot(kind='bar')


# In[58]:


sns.barplot(x=df['Ips'],y=df['Price'])


# In[59]:


new = df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[60]:


df['X_res'] = new[0]
df['Y_res'] = new[1]


# In[61]:


df.sample(5)


# In[62]:


df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[63]:


df.head()


# In[64]:


df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')


# In[65]:


df.info()


# In[66]:


df.corr()['Price']


# In[67]:


df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')


# In[68]:


df.corr()['Price']


# In[69]:


df.drop(columns=['ScreenResolution'],inplace=True)


# In[70]:


df.head()


# In[71]:


df.drop(columns=['Inches','X_res','Y_res'],inplace=True)


# In[72]:


df.head()


# In[73]:


df['Cpu'].value_counts()


# In[74]:


df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[75]:


df.head()


# In[76]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[77]:


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[78]:


df.head()


# In[79]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[80]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[81]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[82]:


df.head()


# In[83]:


df['Ram'].value_counts().plot(kind='bar')


# In[84]:


sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[85]:


df['Memory'].value_counts()


# In[86]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[87]:


df.sample(5)


# In[88]:


df.drop(columns=['Memory'],inplace=True)


# In[89]:


df.head()


# In[90]:


df.corr()['Price']


# In[91]:


df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[92]:


df.head()


# In[93]:


df['Gpu'].value_counts()


# In[94]:


df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])


# In[95]:


df.head()


# In[96]:


df['Gpu brand'].value_counts()


# In[97]:


df = df[df['Gpu brand'] != 'ARM']


# In[98]:


df['Gpu brand'].value_counts()


# In[99]:


sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[100]:


df.drop(columns=['Gpu'],inplace=True)


# In[101]:


df.head()


# In[102]:


df['OpSys'].value_counts()


# In[103]:


sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[104]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[105]:


df['os'] = df['OpSys'].apply(cat_os)


# In[106]:


df.head()


# In[107]:


df.drop(columns=['OpSys'],inplace=True)


# In[108]:


sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[109]:


sns.distplot(df['Weight'])


# In[110]:


sns.scatterplot(x=df['Weight'],y=df['Price'])


# In[111]:


df.corr()['Price']


# In[114]:


sns.heatmap(df.corr(),annot=True)


# In[85]:


sns.distplot(np.log(df['Price']))


# In[86]:


X = df.drop(columns=['Price'])
y = np.log(df['Price'])


# In[87]:


X


# In[88]:


y


# In[89]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)


# In[90]:


X_train


# In[91]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[93]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# ### Linear regression

# In[94]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### KNN

# In[97]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Decision Tree

# In[98]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Random Forest

# In[100]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Exporting the Model

# In[105]:


import pickle
pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[106]:


df


# In[ ]:





# In[ ]:




