"""
 __| |_____________________________________________________________________________________| |__
(__   _____________________________________________________________________________________   __)
   | |                                                                                     | |
   | |                                      realTimeDemo                                   | |
   | |                                                                                     | |
   | |                  Prediction in real time of diferent dynamics signs                 | |
 __| |_____________________________________________________________________________________| |__
(__   _____________________________________________________________________________________   __)
   | |                                                                                     | |
"""
import cv2
import oakD ## oakD.py module, initial configuration of the camera
import numpy as np
import depthai as dai
import mediapipe as mp
import bodyPointsDetector as bp ## bodyPointsDetector.py module, acquisition of the keypoints values
from keras.layers import LSTM,Dense
from tensorflow.keras.models import Sequential

## ------------------------- Flags for the calibration process to define the chest as the reference for the Z coordinate -------------------------------
calibrar = False ##Init calibration process
distPecho = 0 ##Distance of the chest
PCuerpoZ2 = []
PCuerpoX2 = []
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## ----------------------------- Flags for the acquisition of the frames for the prediction process(sliding window of 15 frames) -----------------------
Escribir = [] ##List of frames acquired
nCuadros = 15 ##Size of the sliding window of frames
contaCuadros = 0 ##Frames counter
ErrPos = False ##Flag to indicate if the user is in a correct or incorrect position with respect to the camera
Capturando = True ##Flag to indicate the state of the frame acquisition process
## -----------------------------------------------------------------------------------------------------------------------------------------------------


## ---------------------------------------------------------- Load Model --------------------------------------------------------------------------------
phrases=np.array(['Diarrea','DolordeCabeza','DolordeCuerpo','Fatiga','Fiebre','Sin sena','Tos']) ##Phrases to predict
rutaModelo = r'C:\Users\khmap\depthai-python\Ejemplos_Python\Datos nuevos\En_Enf_1.h5' ##Path to the pre-trained model weights

model = Sequential()
model.add(LSTM(420, return_sequences=True, activation='relu', input_shape=(15, 201)))  # capa entrada
model.add(LSTM(128, return_sequences=True, activation='relu'))  # capas ocultas
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))  # capa de salida
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.load_weights(rutaModelo)
## -----------------------------------------------------------------------------------------------------------------------------------------------------


## Gets model predictions
def Prediccion(Escribir):
    global model
    res = Escribir
    res = res[0:len(res), 0:len(res[1])]

    x_test = np.zeros((int(1), 15, 201))
    x_test[0] = res
    resul = model.predict(x_test)

    ## Evaluates prediction
    yhat = np.argmax(resul, axis=1).tolist()

    if(resul[0][yhat]>0.8):  ##Accuracy umbral
        print("numerito",resul[0][yhat]) ##Accuracy
        print("seña:",phrases[yhat]) ##Predicted sign

## Gets frames for the prediction process
def Captura(vector,Capturando):
    global Escribir
    v = []
    for i in range(0, len(vector)):
        k = len(vector[i])
        for j in range(0, k):
            v = np.append(v, vector[i][j])

    if len(Escribir)== 0:
        Escribir = v
    else:
        Escribir = np.vstack([Escribir,v])

    ## ---------------- Update empty data -----------------------------------
    if (len(Escribir)>=3) & (len(Escribir)!=201):
        vecActual = Escribir[len(Escribir) - 2]
        
        indicesFaltantes = [i for i, x in enumerate(vecActual) if x == 0]

        if indicesFaltantes != []:
            indiceAnt = len(Escribir)-3
            indiceAc = len(Escribir)-2
            indicePos =len(Escribir)-1

            vecPosterior = Escribir[indicePos]
            vecAnterior = Escribir[indiceAnt]

            for i in range(len(indicesFaltantes)):
                indice = indicesFaltantes[i]
                vecActual[indice] = (vecAnterior[indice]+vecPosterior[indice])/2

            Escribir[indiceAc] = vecActual

        ## Sends the sliding window of 15 frames to the prediction process
        if Capturando == False:
            Prediccion(Escribir)
            Escribir = []

## ------------------------------------------------------------------ Main -----------------------------------------------------------------------------

oak = oakD.oakD() ##Creates the oakD object with all the neccesary configuration to use the camera

with dai.Device(oak.pipeline) as device:
    device.startPipeline() ##Inits the pipeline
    colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False) ##Queue with the RGB frames
    dispQueue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False) ##Queue with the depth frames
    points = bp.bodyPointsDetector() ##Creates a bodyPointDetector object
    mp_holistic = mp.solutions.holistic ##Defines the mediapipe model to use

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ## Starts acquisition of frames for the prediction process
            if len(Escribir) == 0:
                Capturando = True
                contaCuadros = 0
            
            ##Gets the Queue and frames from the oakD camera
            inDisp = dispQueue.get()
            inColor = colorQueue.get()
            dispFrame = inDisp.getFrame()
            colorFrame = inColor.getCvFrame()

            ##MediaPipe needs RGB frames
            colorFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2RGB)

            colorFrame.flags.writeable = False
            results = holistic.process(colorFrame) ##Model prediction using MediaPipe
            bdPoints = points.getPointsCoordsXYZ(colorFrame, dispFrame, results) ##Get all the keypoint data
            colorFrame.flags.writeable = True

            ##Turns back the RGB frames to BGR
            colorFrame = cv2.cvtColor(colorFrame, cv2.COLOR_RGB2BGR)

            ##Draws all the landmarks in the RGb and Depth frames
            points.drawMarks(colorFrame, dispFrame, bdPoints)

            ## ------------------- Calibration process to get the z coordinate of each keypoint with respect to the chest ---------------------
            if calibrar and distPecho == 0:
                distPecho = points.calibrateTPose(bdPoints, distPecho) ##Gets the actual chest distance
    
            ##Gets a dictionary with the real world coordinates in meters of each keypoint
            rw_BP = points.getRealWorldCoordsXYZ(colorFrame, bdPoints, distPecho)
            ## --------------------------------------------------------------------------------------------------------------------------------

            """
            #################################################################################################
            #   All the keypoints data are in the rw_BP dicctionary:                                        #    
            #   rw_BP = {                                                                                   #    
            #       "rostro" : [[x_values], [y_values], [z_values]], ##Face                                 #
            #       "manoD" :  [[x_values], [y_values], [z_values]], ##Right hand                           #
            #       "manoI" :  [[x_values], [y_values], [z_values]], ##Left hand                            #
            #       "cuerpo" : [[x_values], [y_values], [z_values]], ##Body                                 #
            #   }                                                                                           #
            #                                                                                               #
            #   face: 20 keypoints                                                                          #
            #       left_eye = 4 keypoints                                                                  #
            #       right_eye = 4 keypoints                                                                 #
            #       lips = 4 keypoints                                                                      #
            #       left_eyebrow = 4 keypoints                                                              #
            #       right_eyebrow = 4 keypoints                                                             #
            #                                                                                               #
            #   right_hand = 21 keypoints                                                                   #
            #   left_hand = 21 keypoints                                                                    #
            #   body = 5 keypoints                                                                          #
            #                                                                                               #
            # ------------------------------------TOTAL = 67 keypoints ------------------------------------ #
            #################################################################################################
            """
            ## ------------------------------------- Data acquisition process -------------------------------------------------
            PCuerpoZ2 = rw_BP["cuerpo"][2][:] ## All Z values of the body
            PCuerpoX2 = rw_BP["cuerpo"][0][:] ## All X values of the body

            cv2.imshow("RGB", colorFrame)
            if Capturando == True: ## Checks if the data acquisiton is already running
                if ErrPos == True: ## The user is in a WRONG position
                    print("Posicionate al centro y entre 1 y 2 m") ## Asks the user to be in the center of the image with a distance of 1-2m with respect to the camera

                if ((round(PCuerpoZ2[4], 2)< 1)|(round(PCuerpoZ2[4], 2)> 2)|(round(PCuerpoX2[4], 2)<-0.1)|(round(PCuerpoX2[4], 2)>0.1)): ##Checks if the user is in a WRONG position
                    ErrPos = True ##The user is in a wrong position

                if ((round(PCuerpoZ2[4], 2) >= 1) & (round(PCuerpoZ2[4], 2) <= 2) & (round(PCuerpoX2[4], 2) >= -0.1) & (round(PCuerpoX2[4], 2) <= 0.1)): ##Checks if the user is in a GOOD position

                    ErrPos = False ## The user is in a GOOD position

                    ## Keypoints values of the body
                    PCuerpoX = rw_BP["cuerpo"][0][:] ## X values
                    PCuerpoY = rw_BP["cuerpo"][1][:] ## Y values
                    PCuerpoZ = rw_BP["cuerpo"][2][:] ## Z values

                    ## Keypoints values of the face
                    ProstroX = rw_BP["rostro"][0][:] ## X values
                    ProstroY = rw_BP["rostro"][1][:] ## Y values
                    ProstroZ = rw_BP["rostro"][2][:] ## Z values

                    ## Keypoints of the left hand
                    PManoIX = rw_BP["manoI"][0][:] ## X values
                    PManoIY = rw_BP["manoI"][1][:] ## Y values
                    PManoIZ = rw_BP["manoI"][2][:] ## Z values

                    ## Keypoints of the right hand
                    PManoDX = rw_BP["manoD"][0][:] ## X values
                    PManoDY = rw_BP["manoD"][1][:] ## Y values
                    PManoDZ = rw_BP["manoD"][2][:] ## Z values

                    vector = [PCuerpoX, PCuerpoY, PCuerpoZ, 
                              ProstroX, ProstroY, ProstroZ, 
                              PManoIX, PManoIY, PManoIZ,
                              PManoDX, PManoDY, PManoDZ]

                    contaCuadros = contaCuadros+1 ##Increases frame counter

                    ## Verifies if the frames filled up the sliding window
                    if nCuadros == contaCuadros:
                        Capturando = False
                        Captura(vector, Capturando)
                    else:
                        Captura(vector,Capturando)

            if cv2.waitKey(1) == ord('q'):
                break