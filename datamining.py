
import numpy as np 
import pandas as pd 


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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


#Importing Dataset


df = pd.read_csv('diabetes_data.csv')
df.head(5)


#EDA (Custome way)


# Dealing with missing values


df.isna().sum()


df.info()



# Distribution of target variable



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



# Distribution of Gender


sns.countplot(x=df['Gender'],hue=df['class'], data=df)

plot_criteria= ['Gender', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of Polyuria


sns.countplot(x=df['Polyuria'],hue=df['class'], data=df)


plot_criteria= ['Polyuria', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of sudden weight loss


sns.countplot(x=df['sudden weight loss'], hue = df['class'], data = df)
plot_criteria = ['sudden weight loss','class']
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of weakness


sns.countplot(x=df['weakness'],hue=df['class'], data=df)


plot_criteria= ['weakness', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of Genital thrush


sns.countplot(x=df['Genital thrush'],hue=df['class'], data=df)


plot_criteria= ['Genital thrush', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of visual blurring


sns.countplot(x=df['visual blurring'],hue=df['class'], data=df)


plot_criteria= ['visual blurring', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of Itching


sns.countplot(x=df['Itching'],hue=df['class'], data=df)


plot_criteria= ['Itching', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of Irritability


sns.countplot(x=df['Irritability'],hue=df['class'], data=df)


plot_criteria= ['Irritability', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of delayed healing


sns.countplot(x=df['delayed healing'],hue=df['class'], data=df)


plot_criteria= ['delayed healing', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of partial paresis


sns.countplot(x=df['partial paresis'],hue=df['class'], data=df)


plot_criteria= ['partial paresis', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of muscle stiffness


sns.countplot(x=df['muscle stiffness'],hue=df['class'], data=df)


plot_criteria= ['muscle stiffness', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of Alopecia


sns.countplot(x=df['Alopecia'],hue=df['class'], data=df)


plot_criteria= ['Alopecia', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# Distribution of Obesity


sns.countplot(x=df['Obesity'],hue=df['class'], data=df)


plot_criteria= ['Obesity', 'class']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(df[plot_criteria[0]], df[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)



#Data pre processing


# Changing target values into numerical values , basically converting 'positive' to 1 and 'negative'  to 0


df['class'] = df['class'].apply(lambda x: 0 if x=='Negative' else 1)
df['class'].head(2)


# Separating Target feature

inp = df.drop(['class'], axis=1)
outp = df['class']


# Storing Features


objectList = inp.select_dtypes(include = "object").columns
print(objectList)


# Label encoding using sklearn


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feature in objectList:
    inp[feature] = le.fit_transform(inp[feature].astype(str))  

print (inp.info())


# Here astype is used for casting the data type into int64


inp.head()


#Correlation between features


inp.corrwith(outp)


# Correlation with Response Variable class


inp.corrwith(outp).plot.bar(
        figsize = (16, 6), title = "Correlation with Diabetes", fontsize = 15,
        rot = 90, grid = True)



#Feature Selection


inp.columns


from sklearn.linear_model import LassoCV
import matplotlib as matplotlib
reg = LassoCV()
reg.fit(inp,outp)
print("Best alpha using builtin lassocv %f" ,reg.alpha_)
print("Best score using builtin lassocv %f" %reg.score(inp,outp))
coef = pd.Series(reg.coef_,index=inp.columns)
print("Lasso picked "+ str(sum(coef!=0))+ "variables and eliminated the other " + str(sum(coef==0)) + "variables" )
imp_coef = coef.sort_values()

imp_coef.plot(kind="barh")
plt.title("feature importance using Lasso model")
print(coef)



inp.shape[1]


inp_FS = inp = inp.drop(['Alopecia'], axis=1)


inp_FS.shape[1]


inp_FS.columns


l1 = list(inp.columns)
l2 = list(inp_FS.columns)


# These features are not included in our model


miss = []
for i in l1:
    if i not in l2:
        miss.append(i)
print(miss)


#Splitting into training and testing


x_train, x_test, y_train,y_test = train_test_split(inp_FS,outp, test_size = 0.2, stratify = outp, random_state = 12345)


#Data Normalization:
# 
# Here we have used minmax normalization technique for normalizing the age attribute


minmax = MinMaxScaler()
x_train[['Age']] = minmax.fit_transform(x_train[['Age']])
x_test[['Age']] = minmax.transform(x_test[['Age']])





#Logistic Regression


logic = LogisticRegression(random_state = 0,  penalty='l2')
logic.fit(x_train, y_train)


#k-Fold cross-validation


kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_logis = cross_val_score(estimator=logic,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


#Model evaluation (Logistic Regression) :


# Model Evaluation
y_predict_logi = logic.predict(x_test)
acc = accuracy_score(y_test, y_predict_logi)
roc = roc_auc_score(y_test, y_predict_logi)
prec = precision_score(y_test, y_predict_logi)
rec = recall_score(y_test, y_predict_logi)
f1 = f1_score(y_test, y_predict_logi)

results = pd.DataFrame([['Logistic Regression',acc, acc_logis.mean(), prec, rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results


print(classification_report(y_predict_logi,y_test))


#Model evaluation (Random Forest) :


rf = RandomForestClassifier(criterion='gini',n_estimators=100,)
rf.fit(x_train,y_train)


kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_rf = cross_val_score(estimator=rf
,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


# Model Evaluation
y_predict_r = rf.predict(x_test)
roc=roc_auc_score(y_test, y_predict_r)
acc = accuracy_score(y_test, y_predict_r)
prec = precision_score(y_test, y_predict_r)
rec = recall_score(y_test, y_predict_r)
f1 = f1_score(y_test, y_predict_r)

model_results = pd.DataFrame([['Random Forest',acc, acc_rf.mean(),prec,rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results = results.append(model_results, ignore_index = True)

results


print(classification_report(y_predict_r,y_test))


#Model evaluation (SVM) :


sv=SVC(kernel='linear',random_state=0)
sv.fit(x_train,y_train)


kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_sv = cross_val_score(estimator=sv,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


# Model Evaluation
y_predict_svm = rf.predict(x_test)
roc=roc_auc_score(y_test, y_predict_svm)
acc = accuracy_score(y_test, y_predict_svm)
prec = precision_score(y_test, y_predict_svm)
rec = recall_score(y_test, y_predict_svm)
f1 = f1_score(y_test, y_predict_svm)

model_results = pd.DataFrame([['SVC',acc, acc_sv.mean(),prec,rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results = results.append(model_results, ignore_index = True)
results.head()


print(classification_report(y_predict_svm,y_test))


#Model evaluation (KNN) :
# We have tested with k values 1 to 10 to check which one is giving the best result


scores = []
dic = {}
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors = i,metric='minkowski',p=2)  #minkowski and p = 2 ,euclidean distance
    knn.fit(x_train,y_train)
    predict = knn.predict(x_test)
    score = accuracy_score(predict, y_test)
    scores.append(round(100*score,2))


print(sorted(scores, reverse = True))


# Model Evaluation
knn = KNeighborsClassifier(n_neighbors = 1,metric='minkowski',p=2) 
knn.fit(x_train,y_train)
predict = knn.predict(x_test)
score = accuracy_score(predict, y_test)


prec = precision_score(y_test, predict)
rec = recall_score(y_test, predict)
f1 = f1_score(y_test, predict)


kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_knn = cross_val_score(estimator=knn,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


model_results = pd.DataFrame([['KNN',score, acc_knn.mean(),prec,rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results = results.append(model_results, ignore_index = True)
results


#Model evaluation (Naive bayes-Gaussian NB) :


gb=GaussianNB()
gb.fit(x_train,y_train)



kfold = model_selection.KFold(n_splits=8, random_state=7,shuffle = True)

acc_gb = cross_val_score(estimator=gb,X = x_train,y =y_train, cv = kfold,scoring='accuracy')


# Model Evaluation
y_predict_gb = gb.predict(x_test)
roc=roc_auc_score(y_test, y_predict_gb)
acc = accuracy_score(y_test, y_predict_gb)
prec = precision_score(y_test, y_predict_gb)
rec = recall_score(y_test, y_predict_gb)
f1 = f1_score(y_test, y_predict_gb)

model_results = pd.DataFrame([['GB',acc, acc_gb.mean(),prec,rec, f1,roc]], columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results = results.append(model_results, ignore_index = True)
results.head()


# export models
import joblib
filename = "logic.joblib"
joblib.dump(logic, filename)
filename = "rf.joblib"
joblib.dump(rf, filename)
filename = "sv.joblib"
joblib.dump(sv, filename)
filename = "knn.joblib"
joblib.dump(knn, filename)
filename = "gb.joblib"
joblib.dump(gb, filename)





