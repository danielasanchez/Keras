# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:17:56 2021
pip install opencv-python <--------- instalar
@author: Daniela Sánchez

ORL database con CNN simple
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
import scipy.io
import pickle
import matplotlib.pyplot as plt
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

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,stratify=y) # training / testing
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.30,stratify=y_train) # training / validation
num_classes= y_test.shape[1]
#np.sum(y_validation, axis=0)


#sacar el tamaño de las imagenes
height=X_train.shape[1]
width=X_train.shape[2]
channel=X_train.shape[3]


X_train= X_train.reshape(X_train.shape[0],height,width,channel).astype('float32')
X_validation= X_validation.reshape(X_validation.shape[0],height,width,channel).astype('float32')
X_test= X_test.reshape(X_test.shape[0],height,width,channel).astype('float32')

start = time.time()# empieza el entrenamiento

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

#entrenando, history guardara los valores de perdida y exactitud
history = model.fit(X_train,y_train,validation_data=(X_validation,y_validation),epochs=30,batch_size=30)

results = model.evaluate(X_test,y_test,verbose=0)

print("Accuracy: %0.2f%%" % (results[1]*100))
#print(model.metrics_names)

done = time.time()
elapsed = done - start

print("Time: %0.2f sec"%(elapsed))

print('listo')


# si usas google colab poner toda la ruta para guardar ---> /content/gdrive/MyDrive/1BasesDatos/testCNN.mat'
#Para guardar para MatLab, necesitas crear un diccionario y listar todas las variables
#aqui solo guardo dos
scipy.io.savemat('testCNN.mat', {'done':done,'X_test':X_test})

#guardar CNN y pesos
model.save('testCNN.h5') 

#guardar datos para python
f = open('testCNN.pckl', 'wb')
#listar todas las variables que quieres guardar
pickle.dump([X_train,X_validation,X_test], f)
f.close()


##Para importar modelo
#from keras.models import load_model
#model = load_model('testCNN.h5')

# scipy.io.loadmat('testCNN.mat')



# import pickle
# a = 3; b = [11,223,435];
# pickle.dump([a,b], open("trial.p", "wb"))
# c,d = pickle.load(open("trial.p","rb"))
# print(c,d) ## To verify
##https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python

print(history.history.keys())

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


# https://stackoverflow.com/questions/35472712/how-to-split-data-on-balanced-training-set-and-test-set-on-sklearn