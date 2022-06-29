#This model was trained on MSCOCO dataset.
#https://github.com/spmallick/learnopencv/tree/master/Mask-RCNN
#https://learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-rcnn-in-opencv-python-c/


# #use this to run .py file in spyder console while passing arguments
# run /home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Mask_RCNN/Mask_rcnn_opencv.py --image /home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Mask_RCNN/cars.jpg


import cv2 as cv
import argparse
import numpy as np
import os.path
import sys
import random
import copy



class Instance_segmentation():
    def __init__(self, confThreshold = 0.5 ,maskThreshold = 0.3 ):
        self.confThreshold = confThreshold
        self.maskThreshold = maskThreshold
        
        
        # Load names of classes
        classesFile = "/home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Mask_RCNN/mscoco_labels.names"
        self.classes = None
        with open(classesFile, 'rt') as f:
           self.classes = f.read().rstrip('\n').split('\n')
           # print(self.classes)
        
        # Give the textGraph and weight files for the model
        textGraph = "/home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Mask_RCNN/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt" # The text graph file that has been tuned by the OpenCV’s DNN support group, so that the network can be loaded using OpenCV.
        modelWeights = "/home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Mask_RCNN/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" #The pre-trained weights
        
        # Load the network
        self.net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph)        
        self.net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        
        
        # Load the classes
        colorsFile = "/home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Mask_RCNN/colors.txt"
        with open(colorsFile, 'rt') as f:
            colorsStr = f.read().rstrip('\n').split('\n')
        self.colors = [] #[0,0,0]
        for i in range(len(colorsStr)):
            rgb = colorsStr[i].split(' ')
            color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
            self.colors.append(color)

    def check(self, S) : 
        if S == "person" or "bicycle" or "car" or "motorcycle" or "bus" or "train" or "truck" or "traffic light" or "stop sign" or "parking meter" or "bench" : 
            return 1
        return 0
    

    # Draw the predicted bounding box, colorize and show the mask on the image
    def drawBox(self ,classId, conf, left, top, right, bottom, classMask):
        
        if self.check(self.classes[classId]) :  
            # Draw a bounding box.
            # cv.rectangle(self.frame1, (left, top), (right, bottom), (255, 178, 50), 1)
            
            
            # # Print a label of class.
            # label = '%.2f' % conf
            # if self.classes:
            #     assert(classId < len(self.classes))
            #     label = '%s:%s' % (self.classes[classId], label)
            
            # # Display the label at the top of the bounding box
            # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # top = max(top, labelSize[1])
            # cv.rectangle(self.frame1, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            # cv.putText(self.frame1, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            
            
            
            # Resize the mask, threshold, color and apply it on the image
            classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
            mask = (classMask > self.maskThreshold)
            roi = self.frame1[top:bottom+1, left:right+1][mask]
        
            # color = self.colors[classId%len(colors)]
            # Comment the above line and uncomment the two lines below to generate different instance colors
            colorIndex = random.randint(0, len(self.colors)-1)
            color = self.colors[colorIndex]
        
            self.frame1[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
            
            
            # Draw the contours on the image
            mask = mask.astype(np.uint8)
            contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(self.frame1[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)
          
    
    # For each frame, extract the bounding box and mask for each detected object
    def postprocess(self ,boxes, masks):
        # Output size of masks is NxCxHxW where
        # N - number of detected boxes
        # C - number of classes (excluding background)
        # HxW - segmentation shape
        numClasses = masks.shape[1]
        numDetections = boxes.shape[2]
    
        frameH = self.frame2.shape[0]
        frameW = self.frame2.shape[1]
        
        temp = self.frame1.shape[0]
    
        for i in range(numDetections):
            box = boxes[0, 0, i]
            mask = masks[i]
            score = box[2]
            if score > self.maskThreshold:
                classId = int(box[1])
                
                # Extract the bounding box
                left = int(frameW * box[3])
                top = int(frameH * box[4])
                right = int(frameW * box[5])
                bottom = int(frameH * box[6])
                
                left = int ( max(0, min(left, frameW - 1)) * (temp/frameW) )
                top = int ( max(0, min(top, frameH - 1)) * (temp/frameH) )
                right = int (max(0, min(right, frameW - 1)) * (temp/frameW) )
                bottom = int (max(0, min(bottom, frameH - 1)) * (temp/frameH) )
                
                # Extract the mask for the object
                classMask = mask[classId]
    
                # Draw bounding box, colorize and show the mask on the image
                self.drawBox(classId, score, left, top, right, bottom, classMask)
                
                #storing rectangle cordinates
                self.rec_cordinates.append([0.5*(left+right),0.5*(top+bottom),right-left,bottom-top])  # storing centre of rectagle (x,y) and (w,h)
        




    def get_image(self,s1 , s2 ): 
        
            self.frame1 = copy.deepcopy(s1)
            self.frame2 = s2
            
            self.rec_cordinates = []
            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(self.frame2, swapRB=True, crop=False)
        
            # Set the input to the network
            self.net.setInput(blob)
        
            # Run the forward pass to get output from the output layers
            boxes, masks = self.net.forward(['detection_out_final', 'detection_masks'])
        
            # Extract the bounding box and mask for each of the detected objects
            self.postprocess(boxes, masks)
        
            return self.frame1 ,self.rec_cordinates 
        
    