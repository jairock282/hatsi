"""
 __| |_____________________________________________________________________________________| |__
(__   _____________________________________________________________________________________   __)
   | |                                                                                     | |
   | |                                        oakD Module                                  | |
   | |                                                                                     | |
   | | The class oakD() creates an oakD object with all the neccesary camera configuration | |
   | | to use in any other different module.                                               | |
   | |                                                                                     | |
   | |     The configuration of the oakD camera includes:                                  | |
   | |         -Data acquisition from RGB and Stereo cameras                               | |
   | |         -Aligment between RGB and Stereo frames                                     | |
 __| |_____________________________________________________________________________________| |__
(__   _____________________________________________________________________________________   __)
   | |                                                                                     | |
"""

import cv2
import numpy as np
import depthai as dai

class oakD():

    def __init__(self):
        
        ##Creates the pipeline
        self.pipeline = dai.Pipeline()

        ##Nodes for data acquisition
        self.colorCamera = self.pipeline.createColorCamera()
        self.monoLeft = self.pipeline.createMonoCamera()
        self.monoRight = self.pipeline.createMonoCamera()
        self.stereo = self.pipeline.createStereoDepth()

        ##Output nodes
        self.xoutColor = self.pipeline.createXLinkOut()
        self.xoutDisp = self.pipeline.createXLinkOut()
        self.xoutColor.setStreamName("color")
        self.xoutDisp.setStreamName("disparity")

        ## --- Configuration ---
        ##ColorCamera
        self.colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.colorCamera.setIspScale(1,3) ##Re-sizes frame to 360*640
        self.colorCamera.initialControl.setManualFocus(130) ##The focus must be manual to be able to align with the depth image

        ##MonoCameras
        self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        ##Stero/Depth
        self.stereo.setOutputDepth(True)
        self.stereo.setOutputRectified(False)
        self.stereo.setConfidenceThreshold(255)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        self.stereo.setSubpixel(False)

        ##Conexion of the output nodes
        self.monoLeft.out.link(self.stereo.left)
        self.monoRight.out.link(self.stereo.right)
        self.colorCamera.isp.link(self.xoutColor.input)
        self.stereo.disparity.link(self.xoutDisp.input)
        

def main():
    ##----------- EXAMPLE -------------
    print("------------- oakD Module -------------")
    
    oak = oakD() ##oakD object

    with dai.Device(oak.pipeline) as device:
        device.startPipeline() ##Inits the pipeline

        colorQueue = device.getOutputQueue(name="color", maxSize=4, blocking=False) ##Gets rgb queue
        dispQueue = device.getOutputQueue(name="disparity", maxSize=4, blocking=False) ##Gets depth queue

        while True:
        
            inDisp = dispQueue.get() ##depth queue frames
            inColor = colorQueue.get() ##rgb queue frames

            dispFrame = inDisp.getFrame() ##depth frames
            colorFrame = inColor.getCvFrame() ##rgb frames
            
            cv2.imshow("Disp", dispFrame)
            cv2.imshow("Color", colorFrame)

            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == "__main__":
    main()
