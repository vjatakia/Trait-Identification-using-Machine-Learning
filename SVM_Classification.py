import numpy as np
import pandas as pd
import sklearn as skt
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn import metrics


#get the train set
train_set=pd.read_csv('test_gender.csv')


#extract train set features
tr=train_set.iloc[:,2:197]

#extract label
tr_label=train_set.iloc[:,197]


#train test split
xtrain1,xtest1,ytrain1,ytest1=train_test_split(tr,tr_label,test_size=0.3,random_state=38)


#SVM
clf=svm.SVC(kernel='rbf',gamma=0.01)


#fit the model
clf.fit(xtrain1,ytrain1)


# Predict validation labels
t_pred=clf.predict(xtest1)


#Validation Accuracy
print("Accuracy",metrics.accuracy_score(ytest1,t_pred))


#Extract test dataset
dataset=pd.read_csv('gender_features_with_labs_10000.csv')


#take only the feature set
feature_set=dataset.iloc[:,2:197]


#get target only
target_gender=dataset.iloc[:,204]


#predict on test set
t_pred1=clf.predict(feature_set)


#Accuracy
print("Accuracy",metrics.accuracy_score(target_gender,t_pred1))







