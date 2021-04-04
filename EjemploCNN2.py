# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:17:56 2021
pip install opencv-python <--------- instalar
@author: Daniela Sánchez

ORL database con validacion cruzada "manual"
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
import os
import cv2
import numpy as np

#from google.colab import drive <---- si corres en google colab
#drive.mount('/content/gdrive') <---- si corres en google colab

#ruta de la base de datos
path='ORL/'
#path='/content/gdrive/MyDrive/1BasesDatos/ORLejemplo/' <---- si corres en google colab, poner ubicacion de la BD

folders=40
imgs=[]
targets=[]
c=-1

for f in range(1,folders+1):
        folder = path+str(f)
        c+=1
        #print('Cargando imagenes de la carpeta '+str(f))
        for filename in os.listdir(folder):
            #imgR= folder+'/'+filename
            imgR = os.path.join(folder,filename)
            img = cv2.imread(imgR) 
            # img = cv2.resize(img,(100,100))
            # print(imgR)
            if img is not None:
                imgs.append(img)
                targets.append(c)
                
                
#plt.imshow(img)
print('imagenes cargadas')

#[muestras][ancho][alto][canales] <===== en CNN
X=np.array(imgs)#.astype('float32')
targets=np.array(targets)#.astype('float32')
y = np_utils.to_categorical(targets)    


cvscores = []
start = time.time()

#k-fold = 5
#cross-validation, correr la misma arquitectura de la red, con diferentes datos
for cv in range(1,6):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30) # training / testing
    X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.20) # training / validation
    num_classes= y_test.shape[1]
    
    height=X_train.shape[1]
    width=X_train.shape[2]
    channel=X_train.shape[3]
    
    
    X_train= X_train.reshape(X_train.shape[0],height,width,channel).astype('float32')
    X_validation= X_validation.reshape(X_validation.shape[0],height,width,channel).astype('float32')
    X_test= X_test.reshape(X_test.shape[0],height,width,channel).astype('float32')

    
    model= Sequential()
                  #n mapas #filtro
    model.add(Conv2D(16,(5,5),input_shape=(width,height,channel),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(528,activation='relu'))
    model.add(Dense(342,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    #entrenando
    model.fit(X_train,y_train,validation_data=(X_validation,y_validation),epochs=20,batch_size=20,verbose=0)
    
    results = model.evaluate(X_test,y_test,verbose=0)
    print("%s %d-fold: %.2f%%" % (model.metrics_names[1],cv,results[1]*100))
    cvscores.append(results[1]*100)
      

print("Accuracy: %.2f%% (+/- %.2f%%)" %(np.mean(cvscores),np.std(cvscores)))
#print(model.metrics_names)

done = time.time()
elapsed = done - start

print("Time: %0.2f sec"%(elapsed))

print('listo')

