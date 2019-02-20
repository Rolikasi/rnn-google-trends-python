# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:27:05 2019

@author: Rolikasi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

#activate gpu
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()


data_2 = pd.read_csv('data/Google_stock_price_2.csv')
data_2.drop(['Date'], axis = 1, inplace = True)
data_2['Volume']= data_2['Volume'].astype(float)
training_set = data_2.iloc[:2227, 0:4].values # Using multiple predictors.
dataset_train = data_2.iloc[:2227, 0:4]
#training_set = training_set.as_matrix() # Using multiple predictors.



#dataset_train = pd.read_csv('data/Google_Stock_Price_Train.csv')
#training_set = dataset_train.iloc[:, 1:2].values

#for RNN use normalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

sc_predict = StandardScaler()
 
sc_predict.fit_transform(training_set[:,0:1])

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

n_future = 20  # Number of days you want to predict into the future
n_past = 60  # Number of past days you want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:4])
    y_train.append(training_set_scaled[i+n_future-1:i + n_future, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping 3D structure
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

#Adding layers
regressor.add(LSTM(units = 40, return_sequences= True, input_shape = ( n_past, 4)))
regressor.add(Dropout(0.2)) 
regressor.add(LSTM(units = 40, return_sequences= True))
regressor.add(LSTM(units = 20, return_sequences= False))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

#compile RNN
regressor.compile(optimizer='adam', loss= 'mean_squared_error')

#fit the RNN to training set
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=17, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
history = regressor.fit(X_train, y_train, epochs=100,
                        callbacks=[es, rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=64)


#Predictions

#get real stock price of 2017

#dataset_test = pd.read_csv('data/Google_Stock_Price_Test.csv')
dataset_test = data_2.iloc[2227:, 0:4]
real_stock_price = dataset_test.iloc[:, 0:1].values

#get predictions on 2017
#dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

data_2 = data_2.iloc[:, 0:4]
inputs = data_2[len(data_2)-len(dataset_test) - n_past:].values
#inputs = inputs.as_matrix()
#inputs = inputs.reshape(-1,1)
inputs_scaled = sc.transform(inputs)
#data_test = dataset_test.values
#data_test_scaled = sc.transform(data_test)

X_test = []
for i in range(n_past, len(inputs)):
    X_test.append(inputs_scaled[i-n_past:i, 0:4])

X_test = np.array(X_test)
#X_test = X_test.as_matrix()
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc_predict.inverse_transform(predicted_stock_price)

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

#Visualising the result
plt.plot(real_stock_price, color = 'red', label= 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


hfm, = plt.plot(sc_predict.inverse_transform(y_train), 'r', label='actual_training_stock_price')
hfm2, = plt.plot(sc_predict.inverse_transform(regressor.predict(X_train)),'b', label = 'predicted_training_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions vs Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Training')
plt.savefig('graph_training.png', bbox_inches='tight')
plt.show()
plt.close()