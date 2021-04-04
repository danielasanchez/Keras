# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:17:56 2021
pip install opencv-python <--------- instalar
@author: Daniela Sánchez
ORL database con validacion cruzada (StratifiedKFold)
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
from sklearn.model_selection import train_test_split
import time
import os
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
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

num_classes= y.shape[1]

height=img.shape[0]
width=img.shape[1]
channel=img.shape[2]

#reformateamos todas las imagenes, sin importar si son para entrenamiento,validacion o prueba
X= X.reshape(X.shape[0],height,width,channel).astype('float32')

start = time.time()


#Definimos el K fold, importante, creo que automaticamente da un 70/30 para train y test
kfold= StratifiedKFold(n_splits=5,shuffle=True)
cvscores = []

for train,test in kfold.split(X,targets):
    #definimos el modelo keras
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
    #capa de salida (Si fuera binaria
    #model.add(Dense(1,activation='sigmoid')) #binaria
    #model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) #binaria
    #capa de salida
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    #De los datos para entrenamiento, saco para la validacion
    X_train,X_validation,y_train,y_validation = train_test_split(X[train],y[train],test_size=0.20) 
    # Entrenar
    # model.fit(X[train],y[train],epochs=10,batch_size=10,verbose=1) <---- si no tuviera conjunto para validacion
    # verbose=0 no muestra paso por paso
    model.fit(X_train,y_train,validation_data=(X_validation,y_validation),epochs=10,batch_size=20,verbose=0)# training / validation
    # Evaluar
    scores=model.evaluate(X[test],y[test],verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
    cvscores.append(scores[1]*100)



print("%.2f%% (+/- %.2f%%)" %(np.mean(cvscores),np.std(cvscores)))
#print(model.metrics_names)

done = time.time()
elapsed = done - start

print("Time: %0.2f sec"%(elapsed))

print('listo')

