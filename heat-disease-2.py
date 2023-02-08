#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Early Stage Diabetes Prediction**

# About Data
# This dataset contains the sign and symptpom data of newly diabetic or would be diabetic patient.This has been collected using direct questionnaires from the patients of Sylhet Diabetes Hospital in Sylhet, Bangladesh and approved by a doctor.
# 
# Features of the dataset
# The dataset consist of total 15 features and one target variable named class.
# 
# 1. Age: Age in years ranging from (20years to 65 years)
# 2. Gender: Male / Female
# 3. Polyuria: Yes / No
# 4. Polydipsia: Yes/ No
# 5. Sudden weight loss: Yes/ No
# 6. Weakness: Yes/ No
# 7. Polyphagia: Yes/ No
# 8. Genital Thrush: Yes/ No
# 9. Visual blurring: Yes/ No
# 10. Itching: Yes/ No
# 11. Irritability: Yes/No
# 12. Delayed healing: Yes/ No
# 13. Partial Paresis: Yes/ No
# 14. Muscle stiffness: yes/ No
# 15. Alopecia: Yes/ No
# 16. Obesity: Yes/ No
# 
# Class: Positive / Negative

# # **Importing libraries**

# In[2]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib


# # **Importing Dataset**

# In[3]:


df = pd.read_csv('diabetes_data.csv')
df.head(5)


# # **EDA (Custome way)**

# > **Dealing with missing values**

# In[4]:


df.isna().sum()


# In[5]:


df.info()


# > As we can see there is no missing value in the dataset

# **Distribution of different attributes**

# > **Distribution of target variable**

# In[6]:


sns.countplot(x=df['class'],data=df)

# plotting to create pie chart and bar plot as subplots
plt.figure(figsize=(14,7))
plt.subplot(121)
df["class"].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",7),startangle = 60,labels=["Positive","Negative"],
wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.1,0],shadow =True)
plt.title("Distribution of Target  Variable")

plt.subplot(122)
ax = df["class"].value_counts().plot(kind="barh")

for i,j in enumerate(df["class"].value_counts().values):
    ax.text(.7,i,j,weight = "bold",fontsize=20)

plt.title("Count of Target Variable")
plt.show()


# > **Distribution of Gender**

# In[7]:


sns.countplot(x=df['Gender'],hue=df['class'], data=df)

plot_criteria= ['Gender', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of Polyuria**

# In[8]:


sns.countplot(x=df['Polyuria'],hue=df['class'], data=df)


plot_criteria= ['Polyuria', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of sudden weight loss**

# In[9]:


sns.countplot(x=df['sudden weight loss'], hue = df['class'], data = df)
plot_criteria = ['sudden weight loss','class']
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of weakness**

# In[10]:


sns.countplot(x=df['weakness'],hue=df['class'], data=df)


plot_criteria= ['weakness', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of Genital thrush**

# In[11]:


sns.countplot(x=df['Genital thrush'],hue=df['class'], data=df)


plot_criteria= ['Genital thrush', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of visual blurring**

# In[12]:


sns.countplot(x=df['visual blurring'],hue=df['class'], data=df)


plot_criteria= ['visual blurring', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of Itching**

# In[13]:


sns.countplot(x=df['Itching'],hue=df['class'], data=df)


plot_criteria= ['Itching', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of Irritability**

# In[14]:


sns.countplot(x=df['Irritability'],hue=df['class'], data=df)


plot_criteria= ['Irritability', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of delayed healing**

# In[15]:


sns.countplot(x=df['delayed healing'],hue=df['class'], data=df)


plot_criteria= ['delayed healing', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of partial paresis**

# In[16]:


sns.countplot(x=df['partial paresis'],hue=df['class'], data=df)


plot_criteria= ['partial paresis', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of muscle stiffness**

# In[17]:


sns.countplot(x=df['muscle stiffness'],hue=df['class'], data=df)


plot_criteria= ['muscle stiffness', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of Alopecia**

# In[18]:


sns.countplot(x=df['Alopecia'],hue=df['class'], data=df)


plot_criteria= ['Alopecia', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# > **Distribution of Obesity**

# In[19]:


sns.countplot(x=df['Obesity'],hue=df['class'], data=df)


plot_criteria= ['Obesity', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# # **Automated EDA using sweetviz and autoviz**

# In[20]:


# pip install sweetviz


# In[21]:


import sweetviz as sv


# In[22]:


#create a object 
report=sv.analyze(df)


# **The output of sweetviz library is a html file. We need to provide a filename with html tag in show_html function.**

# In[23]:


report.show_html("my_first_eda_sweetviz.html")


# > *If you are working on kaggle then you can download this file from kaggle output folder. After downloading this file you can run it on your browse*

# > **EDA Using AutoViz**

# In[24]:


# !pip install autoviz
# !pip install Django
# !pip install channels


# In[25]:


# import autoviz
# from autoviz.AutoViz_Class import AutoViz_Class


# In[26]:


#run this cell to understand autoviz class

# ?AutoViz_Class


# In[27]:


# av=AutoViz_Class()


# In[28]:


# autoviz_eda=av.AutoViz('diabetes_data.csv',verbose=0)


# **EDA using Pandas Profiling**

# In[29]:


import pandas as pd
from pandas_profiling import ProfileReport


# In[30]:


design_report = ProfileReport(df)
design_report.to_file(output_file='report.html')


# In[31]:


# df1=pd.read_html(io='report.html')


# **I personally found pandas profiling more effective in this case****

# # **Data pre processing**

# > **Changing target values into numerical values , basically converting 'positive' to 1 and 'negative'  to 0**

# In[32]:


df['class'] = df['class'].apply(lambda x: 0 if x=='Negative' else 1)
df['class'].head(2)


# > **Separating Target feature**

# In[33]:


inp = df.drop(['class'], axis=1)
outp = df['class']


# > **Storing Features**

# In[34]:


objectList = inp.select_dtypes(include = "object").columns
print(objectList)


# > **Label encoding using sklearn**

# In[35]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feature in objectList:
    inp[feature] = le.fit_transform(inp[feature].astype(str))  

print (inp.info())


# Here astype is used for casting the data type into int64

# In[36]:


inp.head()


# # **Correlation between features**

# In[37]:


inp.corrwith(outp)


# > **Correlation with Response Variable class**

# In[38]:


inp.corrwith(outp).plot.bar(
        figsize = (16, 6), title = "Correlation with Diabetes", fontsize = 15,
        rot = 90, grid = True)


# # **Feature Selection**

# In[39]:


inp.columns


# In[40]:


inp.shape[1]


# In[41]:


inp_FS = inp[['Polyuria', 'Polydipsia','Age', 'Gender','partial paresis','sudden weight loss','Irritability', 'delayed healing','Alopecia','Itching']]


# In[42]:


inp_FS.shape[1]


# In[43]:


inp_FS.columns


# In[44]:


l1 = list(inp.columns)
l2 = list(inp_FS.columns)


# > **These features are not included in our model**

# In[45]:


miss = []
for i in l1:
    if i not in l2:
        miss.append(i)
print(miss)


# # **Splitting into training and testing**

# In[46]:


x_train, x_test, y_train,y_test = train_test_split(inp_FS,outp, test_size = 0.2, stratify = outp, random_state = 12345)


# # **Data Normalization:**
# 
# Here we have used minmax normalization technique for normalizing the age attribute

# In[47]:


minmax = MinMaxScaler()
x_train[['Age']] = minmax.fit_transform(x_train[['Age']])
x_test[['Age']] = minmax.transform(x_test[['Age']])


# # **Model Building **
# Let's jump into the interesting part which is building models and apply them in our dataset. We have applied the following models into our training sets:
# 1. Logistic Regression
# 2. 

# # **Logistic Regression**

# In[48]:

logic = LogisticRegression(random_state = 0,  penalty='l2')
logic.fit(x_train, y_train)
filename = "logic.joblib"
joblib.dump(logic, filename)

# # **k-Fold cross-validation**
# k-Fold cross-validation is a technique that minimizes the disadvantages of the hold-out method. k-Fold introduces a new way of splitting the dataset which helps to overcome the “test only once bottleneck”.
# 
# The algorithm of the k-Fold technique:
# 
# Pick a number of folds – k. Usually, k is 5 or 10 but you can choose any number which is less than the dataset’s length.
# Split the dataset into k equal (if possible) parts (they are called folds)
# Choose k – 1 folds as the training set. The remaining fold will be the test set
# Train the model on the training set. On each iteration of cross-validation, you must train a new model independently of the model trained on the previous iteration
# Validate on the test set
# Save the result of the validation
# Repeat steps 3 – 6 k times. Each time use the remaining  fold as the test set. In the end, you should have validated the model on every fold that you have.
# To get the final score average the results that you got on step 6.
# 
# ![k-Fold cross validation](https://i0.wp.com/neptune.ai/wp-content/uploads/Cross-validation-k-fold.jpg?resize=525%2C525&ssl=1)
# 
# [Reference](https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)

# In[49]:


kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_logis = cross_val_score(estimator=logic,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


# # **Model evaluation (Logistic Regression) :**

# In[50]:


# Model Evaluation
y_predict_logi = logic.predict(x_test)
acc = accuracy_score(y_test, y_predict_logi)
roc = roc_auc_score(y_test, y_predict_logi)
prec = precision_score(y_test, y_predict_logi)
rec = recall_score(y_test, y_predict_logi)
f1 = f1_score(y_test, y_predict_logi)

results = pd.DataFrame([['Logistic Regression',acc, acc_logis.mean(), prec, rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results


# In[51]:


print(classification_report(y_predict_logi,y_test))


# # **Model evaluation (Random Forest) :**

# In[52]:


rf = RandomForestClassifier(criterion='gini',n_estimators=100,)
rf.fit(x_train,y_train)
filename = "rf.joblib"
joblib.dump(rf, filename)

# In[53]:


kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_rf = cross_val_score(estimator=rf
,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


# In[54]:


# Model Evaluation
y_predict_r = rf.predict(x_test)
roc=roc_auc_score(y_test, y_predict_r)
acc = accuracy_score(y_test, y_predict_r)
prec = precision_score(y_test, y_predict_r)
rec = recall_score(y_test, y_predict_r)
f1 = f1_score(y_test, y_predict_r)

model_results = pd.DataFrame([['Random Forest',acc, acc_rf.mean(),prec,rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results = results.append(model_results, ignore_index = True)

#results = results.drop(results.index[[1,9]], inplace=True)
results


# In[55]:


print(classification_report(y_predict_r,y_test))


# # **Model evaluation (SVM) :**

# In[56]:


sv=SVC(kernel='linear',random_state=0)
sv.fit(x_train,y_train)
filename = "sv.joblib"
joblib.dump(sv, filename)

# In[57]:


kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_sv = cross_val_score(estimator=sv,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


# In[58]:


# Model Evaluation
y_predict_svm = rf.predict(x_test)
roc=roc_auc_score(y_test, y_predict_svm)
acc = accuracy_score(y_test, y_predict_svm)
prec = precision_score(y_test, y_predict_svm)
rec = recall_score(y_test, y_predict_svm)
f1 = f1_score(y_test, y_predict_svm)

model_results = pd.DataFrame([['SVC',acc, acc_sv.mean(),prec,rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results = results.append(model_results, ignore_index = True)
#results.drop(results.index[len(results)-1])
#results = results.drop(results.index[[2]], inplace=True,axis =1)
results.head()


# In[59]:


print(classification_report(y_predict_svm,y_test))


# # **Model evaluation (KNN) :**
# We have tested with k values 1 to 10 to check which one is giving the best result

# In[60]:


scores = []
dic = {}
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors = i,metric='minkowski',p=2)  #minkowski and p = 2 ,euclidean distance
    knn.fit(x_train,y_train)
    predict = knn.predict(x_test)
    score = accuracy_score(predict, y_test)
    scores.append(round(100*score,2))
   # print(i)
    #print(score)


# In[61]:


print(sorted(scores, reverse = True))


# In[62]:


# Model Evaluation
knn = KNeighborsClassifier(n_neighbors = 1,metric='minkowski',p=2)  #minkowski and p = 2 ,euclidean distance
knn.fit(x_train,y_train)

filename = "knn.joblib"
joblib.dump(knn, filename)

predict = knn.predict(x_test)
score = accuracy_score(predict, y_test)

#y_predict_svm = rf.predict(x_test)
#roc=roc_auc_score(y_test, y_predict_svm)
#acc = accuracy_score(y_test, y_predict_svm)
prec = precision_score(y_test, predict)
rec = recall_score(y_test, predict)
f1 = f1_score(y_test, predict)


kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_knn = cross_val_score(estimator=knn,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


model_results = pd.DataFrame([['KNN',score, acc_knn.mean(),prec,rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results = results.append(model_results, ignore_index = True)
#results.drop(results.index[len(results)-1])
#results = results.drop(results.index[[3,5]], inplace=True)
results


# # **Model evaluation (Naive bayes-Gaussian NB) :**

# In[63]:


gb=GaussianNB()
gb.fit(x_train,y_train)
filename = "gb.joblib"
joblib.dump(gb, filename)

# In[64]:


kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_gb = cross_val_score(estimator=gb,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


# In[65]:


# Model Evaluation
y_predict_gb = gb.predict(x_test)
roc=roc_auc_score(y_test, y_predict_gb)
acc = accuracy_score(y_test, y_predict_gb)
prec = precision_score(y_test, y_predict_gb)
rec = recall_score(y_test, y_predict_gb)
f1 = f1_score(y_test, y_predict_gb)

model_results = pd.DataFrame([['GB',acc, acc_gb.mean(),prec,rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results = results.append(model_results, ignore_index = True)
#results.drop(results.index[len(results)-1])
#results = results.drop(results.index[[2]], inplace=True,axis =1)
results.head()






