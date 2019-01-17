core_dir = 'trial_vatsal/'

#Importing the libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils.np_utils import to_categorical
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import recall_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


######################## DATA IMPORT AND PREPROCESSING ######################
#Importing the data 
age = pd.read_csv(core_dir+'test_age.csv')
age.drop(columns = ['Unnamed: 0'],inplace = True)
age.head()

#Extracting the relevant column names
col_names = [str(i) for i in range(1,196)]

#Standardizing the data 
for i in col_names:
   age[i] = (age[i]-np.mean(age[i]))/np.std(age[i])

#Extracting the age 
age['age']=2018-age['age']

#Splitting the data into training and validation set
age_train, age_test = train_test_split(age, test_size = 0.1)

X_train = age_train[col_names]
y_train = age_train['age']

X_val = age_test[col_names]
y_val = age_test['age']

#Reshaping the data to include 3 timesteps 
X_train = np.asarray(X_train)
X_train =  X_train.reshape((X_train.shape[0],3,X_train.shape[1]//3))

X_val = np.asarray(X_val)
X_val =  X_val.reshape((X_val.shape[0],3,X_val.shape[1]//3))


############################## MODEL CREATION AND  TRAINING ##################################

# create model
def baseline_model():
  model = Sequential()
  model.add(LSTM(800, input_shape=(X_train.shape[1],X_train.shape[2]),dropout = 0.2))
  model.add(Dense(1))
  rms = optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0)
  model.compile(loss='mae', optimizer= rms)
  return model

#Training the model
estimator = KerasRegressor(build_fn=baseline_model, epochs=1000, batch_size=32)
hist = estimator.fit(X_train, y_train)

#Plotting the convergence graph
plt.plot(hist.history['loss'])
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(core_dir+'lstm_age_train1.png')
plt.show()

#Using the model for prediction on the validation data 
prediction = estimator.predict(X_val)
test_error =  np.abs(y_val - prediction)
print('MAE=',np.mean(test_error))

############################# USING THE MODEL FOR TESTING #############################
#Importing the data 
test = pd.read_csv(core_dir+'age_features_with_labs_10000.csv')
test.head()

#Extracting the relevant column names 
cols = [str(i) for i in range(0,195)]

#Standardizing the data 
for i in cols:
  test[i] = (test[i]-np.mean(test[i]))/np.std(test[i])

#Reshaping the data to match the shape of training data   
X_test= test[cols]

X_test = np.array(X_test)
X_test =  X_test.reshape((X_test.shape[0],3,X_test.shape[1]//3))

y_test = test['age']

#Using the estimated model for prediction on test data 
y_pred1 = estimator.predict(X_test, verbose=1)
test_error =  np.abs(y_test - y_pred1)
print('MAE on Testing data =',np.mean(test_error))








