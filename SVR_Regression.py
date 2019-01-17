
import numpy as np
import pandas as pd
import sklearn as skt
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn import metrics


#import data
train_set=pd.read_csv('test_age.csv')

#extracting only features
feature_set=train_set.iloc[:,2:197]

#extract the label
train_label=train_set.iloc[:,198]

#Train Test Split 
X_train,X_test,y_train,y_test=train_test_split(feature_set,train_label,test_size=0.3,random_state=109)

#Model fitting
reg=svm.SVR(kernel='rbf',gamma=0.01)

reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

print('The MAE for validation set',metrics.mean_absolute_error(y_test,y_pred))


#Get the test data
test_set=pd.read_csv('age_features_with_labs_10000.csv')

test_features=test_set.iloc[:,2:197]

test_label=test_set.iloc[:,205]

#prediction on the test set
y_predt=reg.predict(test_features)

print('The MAE',metrics.mean_absolute_error(test_label,y_predt))

