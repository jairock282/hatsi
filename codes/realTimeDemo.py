"""
 __| |_____________________________________________________________________________________| |__
(__   _____________________________________________________________________________________   __)
   | |                                                                                     | |
   | |                                     realTimeDemo                                    | |
   | |                                                                                     | |
   | |                  Prediction in real time of diferent dynamics signs                 | |
 __| |_____________________________________________________________________________________| |__
(__   _____________________________________________________________________________________   __)
   | |                                                                                     | |
"""
import cv2
import oakD ## oakD.py module, initial configuration of the camera
import imutils
import numpy as np
import tkinter as tk
import depthai as dai
from PIL import Image
import mediapipe as mp
from PIL import ImageTk
import statistics as stat
import bodyPointsDetector as bp ## bodyPointsDetector.py module, acquisition of the keypoints values
from keras.layers import LSTM,Dense
from tensorflow.keras.models import Sequential


## ------------------------- Flags for the calibration process to define the chest as the reference for the Z coordinate -------------------------------
calibrar = False ##Init calibration process
distPecho = 0 ##Distance of the chest
PCuerpoZ2 = []
PCuerpoX2 = []
oak = oakD.oakD() ##Creates the oakD object with all the neccesary configuration to use the camera
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## ----------------------------- Flags for the acquisition of the frames for the prediction process(sliding window of 15 frames) -----------------------
Escribir = [] ##List of frames acquired
nCuadros = 15 ##Size of the sliding window of frames
contaCuadros = 0 ##Frames counter
moda = -1 ##Defines the most repeated prediction
salida = False ##Exits from the UI
predicciones = [] ##List of all predictions attempts
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## ---------------------------------------------------------- Load Model --------------------------------------------------------------------------------
rutaModelo = r'En_Enf_500_1.h5' ##Path to the pre-trained model weights
model = Sequential()
model.add(LSTM(2450, return_sequences=True, activation='relu', input_shape=(15, 201))) ##Input layer
model.add(LSTM(128, return_sequences=True, activation='relu')) ##Hidden layers
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax')) ##Output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.load_weights(rutaModelo)
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## ---------------------------------------------------------- Predictions --------------------------------------------------------------------------------
video_path = ''
##list of symptoms['Diarrhea', 'headache', 'Body pain', 'Fatigue', 'Fever', 'No sign', 'Cough']
phrases=np.array(['Diarrea','Dolor_Cabeza','Dolor_Cuerpo','Fatiga','Fiebre','Sin_Sena','Tos'])
respuestas = [] ##List of all the correct predictions
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## Gets model predictions
def Prediccion(Escribir):
    global salida
    global model
    global predicciones
    global moda
    res = Escribir
    res = res[0:len(res), 0:len(res[1])]

    x_test = np.zeros((int(1), 15, 201))
    x_test[0] = res
    resul = model.predict(x_test)

    ## Evaluates prediction   
    yhat = np.argmax(resul, axis=1).tolist()

    ##If there is more than one sign predicted at the same time
    if len(predicciones) == 5:
        moda = stat.mode(predicciones) ##Gets the most repeated prediction
        predicciones = []
        salida = True
    else:
        predicciones.append(yhat[0]) ##Saves the prediction


## Gets the sliding window of frames to the prediction process
def Captura(vector):
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

    ## ------------------------- Updates empty data -----------------------------------
    if (len(Escribir)>=3) & (len(Escribir)!=201):

        vecActual = Escribir[len(Escribir) - 2]
        indicesFaltantes = [i for i, x in enumerate(vecActual) if x == 0] ##Gets all the frames with zero value

        if indicesFaltantes != []:
            indiceAnt = len(Escribir)-3
            indiceAc = len(Escribir)-2
            indicePos =len(Escribir)-1

            vecPosterior = Escribir[indicePos]
            vecAnterior = Escribir[indiceAnt]

            for i in range(len(indicesFaltantes)):
                indice = indicesFaltantes[i]
                vecActual[indice] = (vecAnterior[indice]+vecPosterior[indice])/2 ##Fills each zero value with the average of the previous and next frame value

            Escribir[indiceAc] = vecActual

## UI
def  visualizar():
    global cap ##Frames
    global btnComenzar ##Start button
    global video_path ##video path
    global btnDeclinarSena ##The sign is ok
    global btnConfirmarSena ##The sign is wrong
    global moda ##Predictions mode
    global btnNoOtraSena ##The user doesn't want to do another sign
    global btnSiOtraSena ##The user wants to do another sign

    if cap is not None: ##New sign frames
        ret,frame = cap.read() ## Gets frames
        if (ret == True):
            frame = imutils.resize(frame,width=640)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)##Transforms BGR to RGB

            im = Image.fromarray(frame) #Gets each frame one by one
            img = ImageTk.PhotoImage(image = im) ##Transforms the frame to TK format

            ##Defines the frame to show
            lblVideo.configure(image = img) 
            lblVideo.image = img
            lblVideo.after(20,visualizar) 
        else:
            
            ##Actives all the buttons
            btnComenzar.configure(state='active')
            btnRepetirV.configure(state='active')
            btnConfirmarSena.configure(state='active')
            btnDeclinarSena.configure(state='active')
            btnNoOtraSena.configure(state='active')
            btnSiOtraSena.configure(state='active')
            btnVolverIniciar.configure(state='active')

            lblVideo.image = ""  ##Clean the TK window
            cap.release()  ##Releases the frames

            if phrases[moda] == 'Sin_Sena':
                video_entrada()

##Defines the video message to show
def leerVideo (): 
    global btnComenzar
    global video_path
    global btnDeclinarSena
    global btnConfirmarSena
    global btnNoOtraSena
    global btnSiOtraSena

    ##Deactives all the buttons
    btnComenzar.configure(state='disabled')
    btnRepetirV.configure(state='disabled')
    btnConfirmarSena.configure(state='disabled')
    btnDeclinarSena.configure(state='disabled')
    btnNoOtraSena.configure(state='disabled')
    btnSiOtraSena.configure(state='disabled')
    btnVolverIniciar.configure(state='disabled')

    global cap 
    global video_path
    if cap is not None: ##If there is already a video, it is released to choose another
        lblVideo.image = ""
        cap.release()
        cap = None

    ##Searches the video to show
    if len(video_path) > 0:
        cap = cv2.VideoCapture(video_path) ##Read the video with OpenCV
        visualizar() ##Shows the video

##Loops the process to defines the video message to show
def Repetir():
    global video_path
    global btnComenzar
    global video_path
    global btnDeclinarSena
    global btnConfirmarSena
    global btnNoOtraSena
    global btnSiOtraSena

    ##Deactives all the buttons
    btnComenzar.configure(state='disabled')
    btnRepetirV.configure(state='disabled')
    btnConfirmarSena.configure(state='disabled')
    btnDeclinarSena.configure(state='disabled')
    btnNoOtraSena.configure(state='disabled')
    btnSiOtraSena.configure(state='disabled')
    btnVolverIniciar.configure(state='disabled')

    leerVideo()

##Welcome video message
def vBienvenida():
    global video_path
    global moda
    global respuestas

    #lblInstruccion.configure(text="Bienvenido al traductor de LSM a Español\nPor favor, oprime el botón comenzar!") ##Spanish
    lblInstruccion.configure(text="Welcome to the Mexican Sign Language-Spanish Translator\nPush the Start button!") ##English
    btnVolverIniciar.grid_forget()
    btnComenzar.grid(column=0, row=1, padx=2, pady=5, columnspan=2)
    moda = -1
    respuestas = []

    video_path = r'videos/Bienvenida.mp4' ##Path to the welcome video
    leerVideo() ##Shows the video message

##Gets the video to process
def video_entrada():
    global btnComenzar
    global btnRepetirV
    global video_path
    global btnDeclinarSena
    global btnConfirmarSena
    global moda
    global btnNoOtraSena
    global btnSiOtraSena

    btnSiOtraSena.grid_forget()
    btnNoOtraSena.grid_forget()

    ##Deactives all the buttons
    btnComenzar.configure(state='disabled')
    btnRepetirV.configure(state='disabled')
    btnConfirmarSena.configure(state='disabled')
    btnDeclinarSena.configure(state='disabled')

    global salida
    global Escribir
    global distPecho
    global contaCuadros
    global predicciones
    ############################
    moda = -1
    Escribir = []
    salida = False
    contaCuadros = 0
    predicciones = []
    ############################

    with dai.Device(oak.pipeline) as device:
        device.startPipeline() ##Inits the OakD pipeline
        colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False)  ##Queue with the RGB frames
        dispQueue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)  ##Queue with the depth frames
        points = bp.bodyPointsDetector()  ##Creates a bodyPointDetector object
        mp_holistic = mp.solutions.holistic  ##Defines the mediapipe model to use

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                ## Starts acquisition of frames for the prediction process
                if len(Escribir) == nCuadros:
                    Prediccion(Escribir)
                    contaCuadros = 0
                    Escribir = []

                ##Gets the Queue and frames from the oakD camera
                inDisp = dispQueue.get()
                inColor = colorQueue.get()
                dispFrame = inDisp.getFrame()
                colorFrame = inColor.getCvFrame()

                ##MediaPipe needs RGB frames
                colorFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2RGB)

                ##Obtenemos el diccionario con los puntos solicitados
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
                    distPecho = points.calibrateTPose(bdPoints, distPecho)  ##Gets the actual chest distance

                ##Gets a dictionary with the real world coordinates in meters of each keypoint
                rw_BP = points.getRealWorldCoordsXYZ(colorFrame, bdPoints, distPecho)
                ## ----------------------------------------------------------------------------

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
                ##Checks if the user is in a correct position to start the data acquisition
                if ((round(PCuerpoZ2[4], 2) >= 1) & (round(PCuerpoZ2[4], 2) <= 2) & (round(PCuerpoX2[4], 2) >= -0.1) & (round(PCuerpoX2[4], 2) <= 0.1)):
                    ## Keypoints values of the body (X,Y,Z)
                    PCuerpo = rw_BP["cuerpo"][0][:], rw_BP["cuerpo"][1][:], rw_BP["cuerpo"][2][:]
                    
                    ## Keypoints values of the face (X,Y,Z)
                    Prostro = rw_BP["rostro"][0][:], rw_BP["rostro"][1][:], rw_BP["rostro"][2][:]

                    ## Keypoints of the left hand (X,Y,Z)
                    PManoI = rw_BP["manoI"][0][:], rw_BP["manoI"][1][:], rw_BP["manoI"][2][:]

                    ## Keypoints of the right hand (X,Y,Z)
                    PManoD = rw_BP["manoD"][0][:], rw_BP["manoD"][1][:], rw_BP["manoD"][2][:]

                    vector = [PCuerpo, Prostro, PManoI, PManoD] ##All the Keypoints

                    contaCuadros = contaCuadros + 1 ##Increases frame counter
                    Captura(vector) ##Gets frames to process

                if salida == True:
                    cv2.destroyAllWindows()
                    break

                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break

    btnComenzar.grid_forget()
    verificarSena()

##Verifies if the sign predicted is correct
def verificarSena():
    global moda
    global video_path
    
    #lblInstruccion.configure(text="¿Esta es la seña que ingresó?") ##Spanish
    lblInstruccion.configure(text="Is this the sign that you have done?") ##English

    btnConfirmarSena.grid(column=0, row=6, padx=5, pady=5, columnspan=2) ##The sign is correct
    btnDeclinarSena.grid(column=1, row=6, padx=5, pady=5, columnspan=2) ##The sign isn't correct

    video_path = r'videos/'+phrases[moda]+'.mp4' ##Defines the video message to show
    leerVideo() ##Shows video message

##Aks if the user wanst to translate other sign
def GuardarOtraSena():
    global video_path
    global respuestas
    global moda

    respuestas = np.append(respuestas,int(moda))

    #lblInstruccion.configure(text="¿Desea traducir otra seña?") ##Spanish
    lblInstruccion.configure(text="Do you want to translate another sign?") ##English

    btnConfirmarSena.grid_forget()
    btnDeclinarSena.grid_forget()

    btnSiOtraSena.grid(column=0, row=6, padx=5, pady=5, columnspan=2)# si quiero guardar otra seña video_entrada
    btnNoOtraSena.grid(column=1, row=6, padx=5, pady=5, columnspan=2)# Finalizar

    video_path = r'videos/Traducir_Otra.mp4'
    leerVideo() ##Shows video message

##Exit message
def vDespedida ():
    global video_path

    #lblInstruccion.configure(text="¡Gracias por utilizar el traductor de LSM a español!") ##Spanish
    lblInstruccion.configure(text="Thank you for using the MSL-Spanish Translator!") ##English

    btnSiOtraSena.grid_forget()
    btnNoOtraSena.grid_forget()
    btnVolverIniciar.grid(column=0, row=1, padx=2, pady=5, columnspan=2)
    video_path= r'videos/Despedida.mp4' ##Exist video message path
    leerVideo() ##shows video message

    print("Resultados",respuestas)
    for i in range (0,len(respuestas)):
        print("Sintoma", phrases[int(respuestas[i])])


## ----------------------------------------------------------- TK window configuration -------------------------------------------------------------------------
cap = None
ventana = tk.Tk()

#lblInstruccion = tk.Label(ventana, text="Hola, gracias por usar el traductor de LSM a español\nPor favor, oprime el boton comenzar") ##Spanish
lblInstruccion = tk.Label(ventana, text="Hi!, Thank you for using the Mexican Sign Language - Spanish Translator\nPush the Start button!") ##English
lblInstruccion.grid(column = 0, row = 0, columnspan=2) ##Position

#btnComenzar = tk.Button(ventana, text ="Comenzar", state= tk.NORMAL, command = video_entrada) ##Spanish
btnComenzar = tk.Button(ventana, text ="Start", state= tk.NORMAL, command = video_entrada) ##English
btnComenzar.grid(column = 0, row = 1, padx = 2, pady = 5, columnspan = 2) ##Position

#btnRepetirV = tk.Button(ventana, text ="Repetir instruccion", state= tk.NORMAL, command = Repetir) ##Spanish
btnRepetirV = tk.Button(ventana, text ="Repeat instruction", state= tk.NORMAL, command = Repetir) ##English
btnRepetirV.grid(column = 0, row = 2, padx = 2, pady = 5, columnspan = 2)

lblVideo = tk.Label(ventana) ##Video space
lblVideo.grid(column = 0, row = 5, columnspan = 2) ##Position

#btnConfirmarSena = tk.Button(ventana, text ="Si", state= tk.NORMAL, command = GuardarOtraSena) ##Spanish
btnConfirmarSena = tk.Button(ventana, text ="Yes", state= tk.NORMAL, command = GuardarOtraSena) ##English
btnDeclinarSena = tk.Button(ventana, text ="No", state= tk.NORMAL, command = video_entrada)

#btnSiOtraSena = tk.Button(ventana, text ="Si", state= tk.NORMAL, command = video_entrada) ##Spanish
btnSiOtraSena = tk.Button(ventana, text ="Yes", state= tk.NORMAL, command = video_entrada) ##English
btnNoOtraSena = tk.Button(ventana, text ="No", state= tk.NORMAL, command = vDespedida)

#btnVolverIniciar = tk.Button(ventana, text ="Volver al inicio", state= tk.NORMAL, command = vBienvenida) ##Spanish
btnVolverIniciar = tk.Button(ventana, text ="Return", state= tk.NORMAL, command = vBienvenida) ##English

vBienvenida() ##Welcome video message
ventana.mainloop() ##Deploys main window
