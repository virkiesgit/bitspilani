#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
import pydot


# In[2]:


from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[3]:


# Load dataset
#names = ['id','age','sex','region','income','married','children','car','saving_acc','current_acc','mortgage','loan']
dataset = pandas.read_csv('./bank_data.csv')


# In[4]:


# shape - show dataset range 
print(dataset.shape)


# In[5]:


# head - print data headers
print(dataset.head(5))


# In[6]:


# descriptions - describe data - general stats
print(dataset.describe())


# In[7]:


#identify columns with null values & number of rows with nulls for that column
dataset.apply(lambda x: sum(x.isnull()),axis=0)


# In[8]:


# identify datatypes for each column
print(dataset.dtypes)


# In[9]:


# create a copy
import copy
ds_orig = copy.deepcopy(dataset)


# In[10]:


#get object type column names
obj_cols = list(dataset.select_dtypes(include=['object']).columns.values)
print(obj_cols)


# In[11]:


# Encoding object type to numberic type
le = LabelEncoder()
for i in obj_cols:
    dataset[i] = le.fit_transform(dataset[i])


# In[12]:


print(dataset.dtypes)


# In[13]:


# head
print(dataset.head(5))


# In[14]:


dataset['income'].hist(bins=50)


# In[15]:


dataset.boxplot(column='income')


# In[16]:


dataset.boxplot(column='income', by = 'sex')


# In[17]:


get_ipython().run_cell_magic('capture', '', '# gather features\nY = dataset[\'loan\']\ndataset = dataset.drop([\'loan\'], axis = 1)\nfeatures = "+".join(dataset.columns)\n\ndataset[\'loan\'] = Y\n\n# get y and X dataframes based on this regression:\ny, X = dmatrices(\'loan ~\' + features, dataset, return_type=\'dataframe\')')


# In[18]:


#correlation of features with dependent field
dataset['loan'] = le.fit_transform(dataset['loan'])
corr_matrix = dataset.corr()
corr_matrix['loan'].sort_values(ascending=False)


# In[19]:


# Create Heatmap
sns.heatmap(corr_matrix)


# In[20]:


# trying to identify few relations in data
temp = ds_orig.groupby(['married', 'sex', 'loan'])['married'].count().unstack('loan')
print(temp)


# In[21]:


temp.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[22]:


temp1 = ds_orig['married'].value_counts(ascending=True)
temp2 = ds_orig.pivot_table(values='loan',
index=['married'],aggfunc=
                       lambda x: x.map({'YES':1,'NO':0}).mean())
print('Frequency Table for Married:') 
print(temp1)

print('\nProbility of getting loan for each Married class:')
print(temp2)


# In[23]:


fig = plt.figure(figsize=(14,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Marriage Status')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Marriage status")
temp1.plot(kind='bar')


# In[24]:


#temp3 = pd.crosstab(df['married'], df['loan'])
temp3 = ds_orig.groupby(['married','loan'])['married'].count().unstack('loan')
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[25]:


# Feature extraction
# For each X, calculate VIF and save in dataframe
vif = pandas.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns


# In[26]:


print(vif.round(1))


# In[27]:


def calculate_vif_(A, thresh=10.0):
    features = []
    for index, row in A.iterrows():
        if(row[0] <= thresh):
            features.append(row[1])

    print('Remaining variables:')
    print(features)
    return features


# In[28]:


vif.apply(lambda x: print(x[0]))


# In[29]:


feature_list = calculate_vif_(vif)
print(feature_list)
print(len(feature_list))


# In[30]:


# removing id; useless feature
feature_list.remove('id')
feature_list.remove('current_acc')
feature_list.remove('car')
print(feature_list)


# In[31]:


new_df = dataset[feature_list]
print(new_df.head())


# In[32]:


# class distribution
print(ds_orig.groupby('loan').size())
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


new_df.shape


# In[34]:


from sklearn import model_selection
seed = 5
scoring = 'accuracy'
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(new_df, Y, test_size=.3,random_state=seed)


# In[35]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[36]:


# Make predictions on validation dataset
from sklearn import tree
knn = tree.DecisionTreeClassifier()
knn = knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


# In[37]:


from sklearn.externals.six import StringIO  
import pydot

import os
os.environ["PATH"] += os.pathsep + 'C:/Anaconda3/pkgs/graphviz-2.38.0-4/Library/bin/graphviz'

dot_data = StringIO() 
tree.export_graphviz(knn, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 

graph[0].write_pdf("DecisionTree-loan.pdf")

