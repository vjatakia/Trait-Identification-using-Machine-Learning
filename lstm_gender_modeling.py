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
from sklearn.metrics import accuracy_score


############################### DATA IMPORT AND PREPROCESSING ############################
gender = pd.read_csv(core_dir+'test_gender.csv')
gender.head()

# Extracting the relevant columns
col_names = [str(i) for i in range(1,196)]

#Standardizing the data 
for i in col_names:
   gender[i] = (gender[i]-np.mean(gender[i]))/np.std(gender[i])

gender.head()


# Splitting the data into train and validation sets 
gender_train, gender_val = train_test_split(gender, test_size = 0.2)

X_train = gender_train[col_names]
y_train = gender_train['196']

X_val = gender_val[col_names]
y_val = gender_val['196']


#Reshaping the data to include 3 timesteps 
X_train = np.asarray(X_train)
X_train =  X_train.reshape((X_train.shape[0],3,X_train.shape[1]//3))

X_val = np.asarray(X_val)
X_val =  X_val.reshape((X_val.shape[0],3,X_val.shape[1]//3))

#Converting the labels into categorical
b, y_train = np.unique(y_train, return_inverse=True)
y_train = to_categorical(y_train)
b, y_val = np.unique(y_val, return_inverse=True)
y_val = to_categorical(y_val)

################################## MODEL CREATION AND TRAINING #################################
# create model
model = Sequential()
model.add(LSTM(512, return_sequences = True, input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(LSTM(512))
model.add(Dense(2, activation='sigmoid'))
rms = optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=None,decay = 0.00001)
model.compile(loss='categorical_crossentropy', optimizer= rms, metrics=['accuracy'])

#Early stopping implementation
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=2), ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs= 1000, batch_size= 32,shuffle = True)

#Plotting the convergence graph
plt.plot(history.history['val_loss'],'y-',label = 'Validation')
plt.plot(history.history['loss'],label = 'Training')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(core_dir+'lstm_train111.png')
plt.show()

# Using the model for prediction on Validation data 
y_pred = model.predict_classes(X_val, verbose=1)
print('Validation Accuracy: ',accuracy_score(np.argmax(y_val,axis = 1),y_pred))
print('Validation Recall: ',recall_score(np.argmax(y_val,axis = 1),y_pred))

######################### USING THE ESTIMATED MODEL FOR TESTING ################################
#Importing the data
test = pd.read_csv(core_dir+'gender_features_with_labs_10000.csv')
test.head()

#Extracting the relevant columns 
cols = [str(i) for i in range(0,195)]


#Reshaping the data 
X_test= test[cols]

X_test = np.array(X_test)
X_test =  X_test.reshape((X_test.shape[0],3,X_test.shape[1]//3))

#Converting the data into categorical form
y_test = test['gender']
b, y_test = np.unique(y_test, return_inverse=True)
y_test = to_categorical(y_test)

#Using the estimated model for prediction on the test data 
y_pred1 = model.predict_classes(X_test, verbose=1)
print("Testing Accuracy:",accuracy_score(np.argmax(y_test,axis = 1),y_pred1))
print("Testing Recall:",recall_score(np.argmax(y_test,axis = 1),y_pred1))
