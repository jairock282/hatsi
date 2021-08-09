"""
 __| |_____________________________________________________________________________________| |__
(__   _____________________________________________________________________________________   __)
   | |                                                                                     | |
   | |                                 modelTraining Module                                | |
   | |                                                                                     | |
   | |             Trains the LSTM model with the sliding windows of 15 frames             | |
 __| |_____________________________________________________________________________________| |__
(__   _____________________________________________________________________________________   __)
   | |                                                                                     | |
"""
import glob
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import LSTM,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

## ------------------------------------------------ Loading Data ------------------------------------------------------------------
files = glob.glob(r'C:\Users\khmap\depthai-python\Ejemplos_Python\Datos_Completos_L/*.csv') ##Read all the CSV files
tam=len(files) ##Total of files
tTrain=(70*tam)/100 ##Gets 70% of the files to the train process
tTest=tam-tTrain ##Gets 30% of the files to the test process

## -------------- Data matrices --------------
x_train=np.zeros((int(tTrain),  15, 201))
x_test=np.zeros((int(tTest),  15, 201))
y_train=np.zeros(int(tTrain))
y_test=np.zeros(int(tTest))

## ----------------- Phrases -------------------
phrases=np.array(['A','B','C','Diarrea','DolordeCabeza','DolordeCuerpo','D','E','Fatiga','Fiebre','F','G','H','I','J','K','L','M','N','O','P','Q','R','Sin sena','S','Tos','T','U','V','W','X','Y','Z','Ñ']) ##Phrases
label_map = {label:num for num, label in enumerate(phrases)} ##Phrases mapping


cont=0   ##Counter to separate 70% of the data to the training process and 30% to the testing process
contNum=0 ##Counter to assign to ytest and ytrain
cont_x_tra=0  ##Counter of the vector x_train
cont_x_tes=0  ##Counter of the vector x_test
cont_y_tra=0  ##Counter of the vector y_train
cont_y_tes=0  ##Counter of the vector y_test

## Iterate over each CSV file
for i in range(0, tam):
    fRead= pd.read_csv(files[i]) ##Read file
    res= fRead.values ##Gets all the values
    res = res[0:len(res), 1:len(res[1])]

    if cont<70:  ## Training data
        x_train[cont_x_tra]=res
        y_train[cont_y_tra]=contNum
        cont=cont+1
        cont_x_tra=cont_x_tra + 1
        cont_y_tra = cont_y_tra + 1

    else: ## Testing data
        x_test[cont_x_tes] = res
        y_test[cont_y_tes] = contNum
        cont = cont + 1
        cont_x_tes =cont_x_tes + 1
        cont_y_tes = cont_y_tes + 1

    if cont==100:
        cont=0
        contNum=contNum+1

##Converts to binary matrix
y_train=to_categorical (y_train).astype(int)
y_test=to_categorical (y_test).astype(int)
print("Datos Guardados")

## -------------------------------------- Model ------------------------------------------------
model=Sequential()
model.add(LSTM(3400,return_sequences=True,activation='relu',input_shape=(15,201))) ##Input layer
model.add(LSTM(400,return_sequences=True,activation='relu')) ##Hidden layers
model.add(LSTM(128,return_sequences=False,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(34,activation='softmax')) ##Output layer
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
model.fit(x_train,y_train,epochs=200)
model.summary() ## Summary of the model results

print("Modelo entrenado")
resul=model.predict(x_test) ##Prediction

## ---------------- Model evaluation ------------------------
print("Evaluacion")
ytrue=np.argmax(y_test,axis=1).tolist()
yhat=np.argmax(resul,axis=1).tolist()
matriz=multilabel_confusion_matrix(ytrue,yhat)
ac = accuracy_score(ytrue,yhat)

model.save('Entrenamiento_ABC_Enf_1.h5') ##Saves the model
