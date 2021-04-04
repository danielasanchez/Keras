# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:17:56 2021
pip install opencv-python <--------- instalar
@author: Daniela Sánchez

This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
 along with a test set of 10,000 images.
"""
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)# Eliminar warning
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist



(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=0.20,stratify=Y_train) # training / validation


height=X_train.shape[1]
width=X_train.shape[2]
channel=1


X_train= X_train.reshape(X_train.shape[0],height,width,channel).astype('float32')
X_validation= X_validation.reshape(X_validation.shape[0],height,width,channel).astype('float32')
X_test= X_test.reshape(X_test.shape[0],height,width,channel).astype('float32')


Y_train = np_utils.to_categorical(Y_train)   
Y_validation = np_utils.to_categorical(Y_validation)  
Y_test = np_utils.to_categorical(Y_test)   


num_classes= Y_test.shape[1]

start = time.time()

model= Sequential()
              #n mapas #filtro
model.add(Conv2D(16,(3,3),input_shape=(width,height,channel),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(528,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(342,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

#entrenando
history = model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=10,batch_size=300)

results = model.evaluate(X_test,Y_test,verbose=0)

print("Accuracy: %0.2f%%" % (results[1]*100))
#print(model.metrics_names)

done = time.time()
elapsed = done - start

print("Time: %0.2f sec"%(elapsed))

print('listo')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy del modelo')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['entrenamiento','validacion'],loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss del modelo')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['entrenamiento','validacion'],loc='upper right')
plt.show()

