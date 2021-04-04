# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:53:14 2021

@author: Daniela Sánchez

Flowers database, usando un iterador para evitar cargar las imagenes en memoria.
Importante, todas las imagenes deben estar en la misma carpeta, no en subcarpetas
y tener en un archivo las etiquetas correspondientes.

"""

from glob import glob
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)# Eliminar warning


#path
images = np.array(sorted(glob("Flores/imagenes/*"))) #path

#labels
mat = loadmat('Flores/imagelabels.mat')
labels = mat['labels'][0] - 1

print("%d imagenes" % len(images))

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=23, stratify=labels)
# Train/validation
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=0.25, random_state=43, stratify=y_train)

#ruta y etiqueta de cada imagen, la idea es no manejar la imagen, sino la ruta
train_frame = pd.DataFrame(np.array([X_train, y_train]).T, columns=['image','class'])
valid_frame = pd.DataFrame(np.array([X_validation, y_validation]).T, columns=['image','class'])
test_frame = pd.DataFrame(np.array([X_test, y_test]).T, columns=['image','class'])


# Aqui podemos agregar el preprocesamiento a realizar, aqui normaliza a valores entre 0 y 1
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


batch=200

train_iter = train_datagen.flow_from_dataframe( dataframe=train_frame,
                                                x_col='image',
                                                y_col='class',
                                                target_size=(100, 120), 
                                                class_mode='categorical',
                                                batch_size=batch, 
                                                shuffle=True)

valid_iter = valid_datagen.flow_from_dataframe(dataframe=valid_frame, 
                                                x_col='image', 
                                                y_col='class',
                                                target_size=(100, 120), 
                                                class_mode='categorical',
                                                batch_size=batch, 
                                                shuffle=False)

test_iter = test_datagen.flow_from_dataframe(dataframe=test_frame, 
                                              x_col='image', 
                                              y_col='class',
                                              target_size=(100, 120), 
                                              class_mode='categorical',
                                              batch_size=batch, 
                                              shuffle=False)


# definimos el modelo 
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(100, 120, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(102, activation='softmax'))

sgd = SGD(lr=0.01, decay=0.002, momentum=0.8, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


history = model.fit(train_iter,validation_data=valid_iter,epochs=10)# training / validation

results = model.evaluate(test_iter,verbose=0)

print("Accuracy: %0.2f%%" % (results[1]*100))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['entrenamiento','validacion'],loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['entrenamiento','validacion'],loc='upper right')
plt.show()
