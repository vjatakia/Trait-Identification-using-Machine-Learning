core_dir = 'trial_vatsal/'

# Importing all the relevant libraries
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
from sklearn.metrics import accuracy_score

################################ DATA IMPORTING  AND PREPROCESSING ###########################

# Importing the data into Python
gender = pd.read_csv(core_dir+'test_gender.csv')
gender.head()

# Extracting the relevant column names
col_names = [str(i) for i in range(1,196)]

#  Standardiziing the data
for i in col_names:
   gender[i] = (gender[i]-np.mean(gender[i]))/np.std(gender[i])

gender.head()

#Splitting the data into training and validation sets
gender_train, gender_val = train_test_split(gender, test_size = 0.2)

X_train = gender_train[col_names]
y_train = gender_train['196']

X_val = gender_val[col_names]
y_val = gender_val['196']

#Converting the labels to categorical data
b, y_train = np.unique(y_train, return_inverse=True)
y_train = to_categorical(y_train)
b, y_val = np.unique(y_val, return_inverse=True)
y_val = to_categorical(y_val)

############################## MODEL CREATION AND TRAINING #############################
# create model
model = Sequential()
model.add(Dense(300, input_dim=X_train.shape[1], activation='relu',kernel_initializer = 'normal'))
model.add(Dense(300, activation='relu'))
model.add(Dense(2, activation='softmax'))
adam = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay= 0.000001, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=0), ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=3000,callbacks = callbacks, batch_size= 32,shuffle = True)

# Plotting the data
plt.plot(history.history['val_loss'],'y-',label = 'Validation')
plt.plot(history.history['loss'],label = 'Training')
plt.savefig(core_dir+'dnn_gender_train.png')
plt.legend()
plt.show()

#Using the model for prediction on validation data 
y_pred = model.predict_classes(X_val, verbose=1)
print('Accuracy on Validation data',accuracy_score(np.argmax(y_val,axis = 1),y_pred))
print('Recall on Validation data',recall_score(np.argmax(y_val,axis = 1),y_pred))

################################ USING THE ESTIMATED MODEL FOR TESTING ##################################
#Importing the test data
test = pd.read_csv(core_dir+'gender_features_with_labs_10000.csv')
test.head()

#Extracting the relevant column
cols = [str(i) for i in range(0,195)]

X_test= test[cols]

#Converting the columns to categorical
y_test = test['gender']
b, y_test = np.unique(y_test, return_inverse=True)
y_test = to_categorical(y_test)

#Using the model for prediction
y_pred1 = model.predict_classes(X_test, verbose=1)
print("Testing Accuracy:",accuracy_score(np.argmax(y_test,axis = 1),y_pred1))
print("Testing Recall:",recall_score(np.argmax(y_test,axis = 1),y_pred1))

