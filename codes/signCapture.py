"""
 __| |______________________________________________________________________________________| |__
(__   ______________________________________________________________________________________   __)
   | |                                                                                      | |
   | |                                 signCapture Module                                   | |
   | |                                                                                      | |
   | |      Captures a sliding window of 15 frames per sign and generates its CSV file.     | |
   | |             Setted up to acquire 100 samples (CSV files) per sign.                   | |
 __| |______________________________________________________________________________________| |__
(__   ______________________________________________________________________________________   __)
   | |                                                                                     | |
"""

import os
import cv2
import oakD
import time
import numpy as np
import pandas as pd
import depthai as dai
import mediapipe as mp
import bodyPointsDetector as bp

## ------------------------- Flags for the calibration process to define the chest as the reference for the Z coordinate -------------------------------
calibrar = False ##Init calibration process
distPecho = 0 ##Distance of the chest
PCuerpoZ2 = []
PCuerpoX2 = []
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## ---------------------------------------------- Flags for the acquisition of the frames --------------------------------------------------------------
Escribir = []  ## List of frames acquired
nCuadros = 0 ##Size of the sliding window of frames
contaCuadros = 0 ##Frames counter
ContaArchivos = 0 ##File counter
BanderaCap = False ##Flag to init frames acquisition process
Capturando = False ##Flag to indicate the state of the frame acquisition process
ErrPos = False ##Flag to indicate if the user is in a correct or incorrect position with respect to the camera
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## ------------------------------------- Saves all the data as CSV ------------------------------------------------
def Guardar(nombre,Escribir):
    csvname = (nombre + "_Datos.csv") ##CSV file name
    path = r'C:\Users\khmap\depthai-python\Ejemplos_Python\Datos nuevos\C' ##Path to the saving point
    df = pd.DataFrame(Escribir)
    df.to_csv(os.path.join(path, csvname))

## ---------------------------------------- Saves data frames -----------------------------------------------------
def Captura(vector,Capturando):
    path = r'C:\Users\khmap\depthai-python\Ejemplos_Python\Datos nuevos\C' ##Path to the saving point
    imagename = (nombre + "_" + str(contaCuadros) + ".jpg")
    cv2.imwrite(os.path.join(path, imagename), colorFrame)
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
    if Capturando == False:
        Guardar(nombre,Escribir)


##------------------------------------------------ Main --------------------------------------------------------------------------------

oak = oakD.oakD() ##Creates the oakD object with all the neccesary configuration to use the camera

with dai.Device(oak.pipeline) as device:
    device.startPipeline() ##Inits the pipeline
    colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False) ##Queue with the RGB frames
    dispQueue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False) ##Queue with the depth frames
    points = bp.bodyPointsDetector() ##Creates a bodyPointDetector object
    mp_holistic = mp.solutions.holistic ##Defines the mediapipe model to use

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            if ((ContaArchivos % 100) == 0)&(ContaArchivos != 0): ##Verifies if there are already 100 samples for the sign
                print("\n\n\nListo!")
                BanderaCap = False

            if (BanderaCap == True)&(Capturando == False): ##Gets the current files number
                ContaArchivos = ContaArchivos+1
                print("\n\n\nArchivo:",ContaArchivos )
                nombre = "C_15_"+str(ContaArchivos)
                nCuadros = 15 ##Defines the number of frames per sample

                if nCuadros == 0:
                    break

                ##Inits the necessary flags to start a new sample acquisition process
                contaCuadros = 0
                Capturando = True
                Escribir = []
                ErrPos = False

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
            if calibrar and distPecho == 0:  ##Verifies is the flag to calibrate("calibrar") is True
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
                    if ErrPos == True:
                        ## Messages to indicates that the data acquisition process will start
                        print("\n\nPosicion recuperada")
                        print("Capturando datos en 3")
                        time.sleep(0.33)
                        print("Capturando datos en 2")
                        time.sleep(0.33)
                        print("Capturando datos en 1")
                        time.sleep(0.33)

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

            ## ------------------------------ Start data acquisition process ----------------------------
            if cv2.waitKey(1) == ord('m'): 
                BanderaCap = True
                print("Capturando datos en 3")
                time.sleep(1)
                print("Capturando datos en 2")
                time.sleep(1)
                print("Capturando datos en 1")
                time.sleep(1)

            if cv2.waitKey(1) == ord('q'):
                break

