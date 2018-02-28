from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils

import struct
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl


np.random.seed(5)


(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Numpy array converted from 2D to 1D
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32')

#Normalize Intesity values of pixel to range 0-1
X_train = X_train / 255
X_test = X_test / 255

#Do One Hot Encoding 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


model = Sequential()

#input layer
model.add(Dense(784 , input_dim=784, kernel_initializer='random_uniform',activation='sigmoid'))

#Hidden layer which has 16 neurons in each layer
model.add(Dense(16 , activation='sigmoid'))
model.add(Dense(16 , activation='sigmoid'))

#output layer
model.add(Dense(10 , activation='sigmoid'))

#compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model fitting
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
#Printing the data to output
print("Final Baseline Error(Using Sigmoid): %.2f%%" % (100-scores[1]*100))


#RELU
model_relu = Sequential()

model_relu.add(Dense(784 , input_dim=784, kernel_initializer='random_uniform',activation='relu'))

model_relu.add(Dense(16 , activation='relu'))
model_relu.add(Dense(16 , activation='relu'))
model_relu.add(Dense(10 , activation='sigmoid'))

model_relu.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_relu.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model_relu.evaluate(X_test, y_test, verbose=0)

print("Final Baseline Error(Using Relu): %.2f%%" % (100-scores[1]*100))
