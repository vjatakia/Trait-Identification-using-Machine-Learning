# Directory Specification
core_dir = 'trial_vatsal/'

# Importing of libraries 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import recall_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

############################## DATA IMPORT AND PREPROCESSING ########################
# Importing Data into Python
age = pd.read_csv(core_dir+'test_age.csv')
age.drop(columns = ['Unnamed: 0'],inplace = True)
age['age'] = 2018-age['age']


#Extracting column names from data
col_names = [str(i) for i in range(1,196)]

#Standardization of data
for i in col_names:
   age[i] = (age[i]-np.mean(age[i]))/np.std(age[i])



#Splitting data into validation and train
age_train, age_test = train_test_split(age, test_size = 0.2)

X_train = age_train[col_names]
y_train = age_train['age']

X_val = age_test[col_names]
y_val = age_test['age']

################################## MODEL TRAINING #########################################

# Create model
def baseline_model():

  model = Sequential()
  model.add(Dense(500, input_dim=X_train.shape[1], activation='relu',kernel_initializer = 'normal'))
  model.add(Dense(500, activation='relu'))
  model.add(Dense(1))
  adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay= 0.00001, amsgrad=False)
  model.compile(loss='mae', optimizer= adam)
  
  return model

# Fit the model
estimator = KerasRegressor(build_fn=baseline_model, epochs=1000, batch_size=32)
hist = estimator.fit(X_train, y_train)

#Plotting the convergence graph
plt.plot(hist.history['loss'])
plt.savefig(core_dir+'dnn_age_train.png')
plt.show()

#Using the model for prediction on Validation data
pred_val = estimator.predict(X_val)
val_error =  np.abs(y_val - pred_val)
print('MAE on Validation Data:',np.mean(val_error))


######################## USING THE MODEL FOR TESTING ##########################

#Importing  the test data
test = pd.read_csv(core_dir+'age_features_with_labs_10000.csv')
test.head()

#Extracting columns names of relevant columns
cols = [str(i) for i in range(0,195)]

#Standardizing the data
for i in cols:
  test[i] = (test[i]-np.mean(test[i]))/np.std(test[i])

#Preprocessing the data
X_test= test[cols]

y_test = np.asarray(test['age'])
y_test = y_test.astype(int)

#Using the above estimated model for prediction
pred_test = estimator.predict(X_test)
test_error =  np.abs(y_test - pred_test)
print('MAE on Test Data:',np.mean(test_error))
