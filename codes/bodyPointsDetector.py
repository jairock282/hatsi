"""
 __| |________________________________________________________________________________________________________________| |__
(__   ________________________________________________________________________________________________________________   __)
   | |                                                                                                                | |
   | |                                         bodyPointsDetector Module                                              | |
   | |                                                                                                                | |
   | | Gets de x,y,z of each keypoint in two modes:                                                                   | |
   | | getPointsCoordsXYZ -> obtains the x and y coordinates in pixels and Z in meters with respect the camera        | |
   | |                                                                                                                | |
   | | getRealWorldCoordsXYZ -> obtains the X and Y coordinates in meters with respect the center point of the image  | |
   | |                          and the Z in meters with respect the camera or the chest.                             | |
 __| |________________________________________________________________________________________________________________| |__
(__   ________________________________________________________________________________________________________________   __)
   | |                                                                                                                | |
"""

import cv2
import oakD
import numpy as np
import depthai as dai
import mediapipe as mp

##OAK-D
FOCAL_LENGTH = 857.06
BASE_LINE_DIST = 0.075

## Text configuration
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
lineType = cv2.LINE_AA

class bodyPointsDetector():
    def __init__(self, sqrSizeBody=10, sqrSizeFace=10, sqrSizeHands=10):
        
        ##Hand keypoints
        self.mano = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        ##Face keypoints
        self.ojoIzq = [249,374,382,386]
        self.ojoDer = [7, 145, 155, 159]
        self.labios = [13,14,78,324]
        self.cejaIzq = [276, 282, 283, 295]
        self.cejaDer = [46, 52, 53, 65]
        self.rostro = self.ojoIzq + self.ojoDer + self.labios + self.cejaIzq + self.cejaDer 

        ##Body keypoints
        self.cuerpo = [11,12,13,14,-1] 
        ##Size of ROI
        self.sqrSizeBody = sqrSizeBody
        self.sqrSizeFace = sqrSizeFace
        self.sqrSizeHands = sqrSizeHands


    ##Get the distance in the Z coordinate
    def getDistanceROI(self, mediana):
        ##Calculate the distance in real world
        if mediana != 0:
            distance = ((FOCAL_LENGTH * BASE_LINE_DIST)/mediana) / 2 ##Divided by 2 to use half of the default resolution
        else:
            distance = 0

        return distance ##Returns the distance in meters

    ##Gets the median of each ROI
    def getMedianROI(self, ROI):
        mediana = 0
        if ROI.size != 0:
            ##Removes all the zero values and then calculate the median
            mediana = np.median(ROI[ROI!=0])

        return mediana
    
    ##Draws the marks for each keypoint
    def drawMarks(self, colorFrame, dispFrame, points):
        ##Draws a point in the center of the image for position reference 
        h,w,c = colorFrame.shape
        cv2.circle(colorFrame, (int(w/2), int(h/2)), 3, (255,255,0), 5)
        
        ##------ Draw landmarks --------
        ##Hands landmarks
        for i in range(0,len(self.mano)):            
            cv2.circle(colorFrame, (points["manoD"][0][i],points["manoD"][1][i]), 2, (255,0,0), 3)##RGB image
            cv2.circle(colorFrame, (points["manoI"][0][i],points["manoI"][1][i]), 2, (255,0,0), 3)

            cv2.rectangle(dispFrame, (points["manoD"][0][i]-self.sqrSizeHands, points["manoD"][1][i]-self.sqrSizeHands), (points["manoD"][0][i]+self.sqrSizeHands, points["manoD"][1][i]+self.sqrSizeHands), (255,255,255), 3)##Depth image
            cv2.rectangle(dispFrame, (points["manoI"][0][i]-self.sqrSizeHands, points["manoI"][1][i]-self.sqrSizeHands), (points["manoI"][0][i]+self.sqrSizeHands, points["manoI"][1][i]+self.sqrSizeHands), (255,255,255), 3)

        ##Face landmarks
        for i in range(0,len(self.rostro)):
            cv2.circle(colorFrame, (points["rostro"][0][i],points["rostro"][1][i]), 1, (0,255,0), 2) ##RGB image
            cv2.rectangle(dispFrame, (points["rostro"][0][i]-self.sqrSizeFace, points["rostro"][1][i]-self.sqrSizeFace), (points["rostro"][0][i]+self.sqrSizeFace, points["rostro"][1][i]+self.sqrSizeFace), (255,255,255), 3) ##Depth image

        ##Body landmarks
        for i in range(0, len(self.cuerpo)):
            cv2.circle(colorFrame, (points["cuerpo"][0][i],points["cuerpo"][1][i]), 1, (0,0,255), 2)##RGB image
            cv2.rectangle(dispFrame, (points["cuerpo"][0][i]-self.sqrSizeBody, points["cuerpo"][1][i]-self.sqrSizeBody), (points["cuerpo"][0][i]+self.sqrSizeBody, points["cuerpo"][1][i]+self.sqrSizeBody), (255,255,255), 3)##Depth image

    ##Gets a dictionary with the xyz values for each keypoint
    def getPointsCoordsXYZ(self, colorFrame, dispFrame, results):
        ##Dictionary to save the xyz values
        bodyPoints = {
            "rostro" : [[0]*len(self.rostro), [0]*len(self.rostro), [0.0]*len(self.rostro)], ##Face
            "manoD" : [[0]*len(self.mano), [0]*len(self.mano), [0.0]*len(self.mano)],##Right hand
            "manoI" : [[0]*len(self.mano), [0]*len(self.mano), [0.0]*len(self.mano)],##Left hand
            "cuerpo" : [[0]*len(self.cuerpo), [0]*len(self.cuerpo), [0.0]*len(self.cuerpo)]##Body
        }

        colorFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2RGB)##MediaPipe works with RGB images       
        h,w,c = colorFrame.shape ##Frame size
        
        ##Saves x and y value for each keypoint
        ##FACE
        if results.face_landmarks:
            for i,lm in enumerate(self.rostro):
                bodyPoints["rostro"][0][i] = int(results.face_landmarks.landmark[lm].x * w) ##x value in pixels
                bodyPoints["rostro"][1][i] = int(results.face_landmarks.landmark[lm].y * h) ##y value in pixels
                
                ##Gets z value in the real world(meters)
                faceRoi = dispFrame[bodyPoints["rostro"][1][i]-self.sqrSizeFace:bodyPoints["rostro"][1][i]+self.sqrSizeFace, bodyPoints["rostro"][0][i]-self.sqrSizeFace:bodyPoints["rostro"][0][i]+self.sqrSizeFace]##Defines ROI
                medianFace = self.getMedianROI(faceRoi) ##Median of ROI
                distanceFace = self.getDistanceROI(medianFace) ##Distance from the camere to ROI in meters
                bodyPoints["rostro"][2][i] = distanceFace ##Saves the distance

        ##RIGHT HAND
        if results.right_hand_landmarks:
            for i in self.mano:
                bodyPoints["manoD"][0][i] = int(results.right_hand_landmarks.landmark[i].x * w) ##x value in pixels
                bodyPoints["manoD"][1][i] = int(results.right_hand_landmarks.landmark[i].y * h) ##y value in pixels

                ##Gets z value in the real world(meters)
                rHandRoi = dispFrame[bodyPoints["manoD"][1][i]-self.sqrSizeHands:bodyPoints["manoD"][1][i]+self.sqrSizeHands, bodyPoints["manoD"][0][i]-self.sqrSizeHands:bodyPoints["manoD"][0][i]+self.sqrSizeHands]##Defines ROI
                medianRHand = self.getMedianROI(rHandRoi) ##Median of ROI
                distanceRHand = self.getDistanceROI(medianRHand) ##Distance from the camera to ROI in meters
                bodyPoints["manoD"][2][i] = distanceRHand ##Saves the distance

        ##LEFT HAND
        if results.left_hand_landmarks:
            for i in self.mano:
                bodyPoints["manoI"][0][i] = int(results.left_hand_landmarks.landmark[i].x * w) ##x value in pixels
                bodyPoints["manoI"][1][i] = int(results.left_hand_landmarks.landmark[i].y * h) ##y value in pixels

                ##Gets z value in the real world(meters)
                lHandRoi = dispFrame[bodyPoints["manoI"][1][i]-self.sqrSizeHands:bodyPoints["manoI"][1][i]+self.sqrSizeHands, bodyPoints["manoI"][0][i]-self.sqrSizeHands:bodyPoints["manoI"][0][i]+self.sqrSizeHands]##Defines ROI
                medianLHand = self.getMedianROI(lHandRoi) ##Median of ROI
                distanceLHand = self.getDistanceROI(medianLHand) ##Distance from the camera to ROI in meters
                bodyPoints["manoI"][2][i] = distanceLHand ##Saves the distance
        
        ##BODY
        if results.pose_landmarks:
            for i,lm in enumerate(self.cuerpo):
                if i < len(self.cuerpo)-1:
                    ##shoulders
                    bodyPoints["cuerpo"][0][i] = int(results.pose_landmarks.landmark[lm].x * w) ##x value in pixels
                    bodyPoints["cuerpo"][1][i] = int(results.pose_landmarks.landmark[lm].y * h) ##y value in pixels

                else:
                    ##The chest keypoint is in the middle of the shoulders keypoints
                    bodyPoints["cuerpo"][0][i] = int((bodyPoints["cuerpo"][0][0] + bodyPoints["cuerpo"][0][1]) / 2) ##x value in pixels
                    bodyPoints["cuerpo"][1][i] = int((bodyPoints["cuerpo"][1][0] + bodyPoints["cuerpo"][1][1]) / 2) ##y value in pixels


                ##Get z value in the real world(meters)
                bodyRoi = dispFrame[bodyPoints["cuerpo"][1][i]-self.sqrSizeBody:bodyPoints["cuerpo"][1][i]+self.sqrSizeBody, bodyPoints["cuerpo"][0][i]-self.sqrSizeBody: bodyPoints["cuerpo"][0][i]+self.sqrSizeBody] ##Defines ROI
                medianBody = self.getMedianROI(bodyRoi) ##Median of ROI
                distanceBody = self.getDistanceROI(medianBody) ##Distance from the camera to ROI in meters
                bodyPoints["cuerpo"][2][i] = distanceBody ##Saves the distance

        return bodyPoints

    ##Calibration process to define the distance of the chest to get the value of the other keypoints with respect to it
    def calibrateTPose(self, points, refDistChest):
        ##Put the hands up to save the chest distance
        if(abs(points["cuerpo"][1][1] - points["cuerpo"][1][3]) < 10 or abs(points["cuerpo"][1][0] - points["cuerpo"][1][2]) < 10):
            refDistChest = points["cuerpo"][2][-1]

        return refDistChest
    
    """
    Gets the values of XYZ in meters:
    X, Y = with respect to the center of the image
    Z = with respect to the camera 
    """
    def getRealWorldCoordsXYZ(self, colorFrame, bodyPoints, distChest):
        
        realBodyPoints = bodyPoints ##copies the dictionary with the xyz values to recalculate them
        
        h,w,c = colorFrame.shape ##Frame size

        ##Image center
        cx = int(w/2) ##x
        cy = int(h/2) ##y
        f = FOCAL_LENGTH / 2 ##Focal length 

        for key in bodyPoints:        
            for i in range(len(bodyPoints[key][0])):     
                xReal = (( bodyPoints[key][0][i] - cx ) * bodyPoints[key][2][i]) / f ##X value in meters with respect to the image center
                yReal = (( bodyPoints[key][1][i] - cy ) * bodyPoints[key][2][i]) / f ##Y value in meters with respect to the image center
                
                realBodyPoints[key][0][i] = xReal
                realBodyPoints[key][1][i] = yReal
                realBodyPoints[key][2][i] = abs(realBodyPoints[key][2][i] - distChest) ##Z value in meter with respect to the distance from the chest

        return realBodyPoints


def main():

    ##--------------------------------------------------------------------- EXAMPLE ---------------------------------------------------------
    print("------------- bodyPointsDetector Module -------------")
    oak = oakD.oakD()
    
    ##------ Z coordinate of each keypoint with respect to the chest? -----
    respPecho = True ##Flag to activate it
    distPecho = 0 ##Distance of the chest
    ##---------------------------------------------------------------------

    with dai.Device(oak.pipeline) as device:
        device.startPipeline()

        colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False)
        dispQueue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

        points = bodyPointsDetector()
        
        ##MediaPipe
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while True:

                ##Gets the frames queue for the RGB and depth images
                inDisp = dispQueue.get()
                inColor = colorQueue.get()

                ##Gets the RGB and depth frame
                dispFrame = inDisp.getFrame()
                colorFrame = inColor.getCvFrame()

                ##BGR to RGB
                colorFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2RGB)
                
                ##Makes the prediction of the keypoints with mediaPipe model
                colorFrame.flags.writeable = False
                results = holistic.process(colorFrame)
                bdPoints = points.getPointsCoordsXYZ(colorFrame, dispFrame, results)              
                colorFrame.flags.writeable = True

                ##RGB to BGR
                colorFrame = cv2.cvtColor(colorFrame, cv2.COLOR_RGB2BGR)
                
                ##Draws all the landMarks
                points.drawMarks(colorFrame, dispFrame, bdPoints)
                
                ##Calibration process
                calibrar = "Off"
                if respPecho and distPecho == 0:
                    distPecho = points.calibrateTPose(bdPoints, distPecho)
                    calibrar = "On"
                    print("Calibrando! {0}".format(distPecho))
                
                ##Real world XYZ values
                rw_BP = points.getRealWorldCoordsXYZ(colorFrame, bdPoints, distPecho)

                
                mano_x = rw_BP["manoD"][0][0]
                mano_y = rw_BP["manoD"][1][0]
                mano_z = rw_BP["manoD"][2][0]

                ##Values of the coordinates of the rigth hand
                cv2.putText(colorFrame, f"Cali:{calibrar}", (50,60), font, fontScale, (255,0,0), thickness, cv2.LINE_AA)
                cv2.putText(colorFrame, f"X: {round(mano_x, 2)} m", (50,90), font, fontScale, (0,255,255), thickness, cv2.LINE_AA)
                cv2.putText(colorFrame, f"Y: {round(mano_y, 2)} m", (50,130), font, fontScale, (255,0,255), thickness, cv2.LINE_AA)
                cv2.putText(colorFrame, f"Z: {round(mano_z, 2)} m", (50,170), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)

                
                cv2.imshow("RGB", colorFrame)
                cv2.imshow("Disparidad", dispFrame)

                if cv2.waitKey(1) == ord('q'):
                    break

if __name__ == "__main__":
    main()
