# Trait-Identification-using-Machine-Learning
This repository contains the Python 3 code for creating models to identify traits like gender and age from audio files using Deep Learning Models like DNN and LSTMs. This problem is a combination of both- Classification and Regression problems. 

The data used is: Stanford Amazing Grace Dataset

Models used: Deep Neural Network(DNN), LSTM

The following is the file description: 
1. dnn_age_modeling.py: This file contains the Python code for creating a DNN Regression model
2. dnn_gender_modeling.py: This file contains the Python code for creating a DNN Classification model
3. lstm_age_modeling.py: This file contains the Python code for creating a LSTM Regression model 
4. lstm_gender_modeling.py: This file contains the Python code for creating a LSTM Classification model
5. SVM_Classification: This file contains the Python code for creating an SVM Classification model used as a baseline for comparisons
6. SVR_Regression: This file contains the Python code for creating an SVM Regression model used as a baseline for comparisons

Results Obtained: 
Model | Accuracy(%) |  MAE
----------------------------
SVM | 87.1 | 11.52
---------------------------
DNN 91.45 10.39
LSTM 94.5 9.34
Table 1: Validation Results
Model Accuracy(%) MAE
SVM 64.2 13.04
DNN 69.02 12.2
LSTM 67.4 11.12
Table 2: Test Results

