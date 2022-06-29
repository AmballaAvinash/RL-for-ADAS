from Mask_rcnn_image import Instance_segmentation
mask = Instance_segmentation()

from Lane_detection import Lane_detector
ld = Lane_detector()

from Depth_map import Depth
depth = Depth()

import numpy as np
import cv2
from PIL import Image 
import matplotlib.pyplot as plt

import torchvision.models as models

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms

transform = transforms.Compose([
        # transforms.Resize((224,224)),       # resize the input to 224x224
        transforms.ToTensor() ,              # put the input to tensor format
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
        # the normalization is based on images from ImageNet
    ])


import copy

class Environment :
    
    def __init__(self,n,threshold,car_width,car_height ,image_shape, video_path ,action_space_f , action_space_theta):
        self.n = n
        self.threshold = threshold
        self.car_width = car_width
        self.car_height = car_height
        self.video_path = video_path
        self.image_shape = image_shape
        self.video_frames , self.video_lane_cordinates , self.video_rec_cordinates = self.get_frames()     
        self.action_space_f = action_space_f
        self.action_space_theta = action_space_theta
        self.time_count = 1
        
        
    
    def imshow(self,image):
        # image = image / 2 + 0.5     # unnormalize
        # image=Image.fromarray(image)  #changing image to PIL format
        # image=transform(image)        #transformed img shape is (3,224,224)                  
        # image = image.numpy()
        # plt.imshow(np.transpose(image, (1, 2, 0)))
        image = image.reshape(224,224,3)
        plt.imshow(image)
        plt.show()
    
    
    
    def draw_car(self, s, car_centre, r, f, theta): #https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/  
          
        image = copy.deepcopy(s)  #(say x = y , partially updating x updates y also where x and y can be lists or arrays . To avoid this problem use deep copy) 
            
        
        #plotting the car box
        # represents the top left corner of rectangle
        start_point = (int(car_centre[0]-0.5*self.car_width), int(car_centre[1]-0.5*self.car_height))           
        # represents the bottom right corner of rectangle
        end_point = (int(car_centre[0]+0.5*self.car_width), int(car_centre[1]+0.5*self.car_height)) 
        
        # Blue color in BGR
        color = (0, 0, 0)          
        # Line thickness of 2 px
        thickness = 1     
        
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
        
        
        
        #plotting the arrow
        start_point = (car_centre[0], car_centre[1])       
        # End coordinate
        end_point = (int(car_centre[0]+r*np.cos(theta*np.pi/180)) , int(car_centre[1]-r*np.sin(theta*np.pi/180))) # # since dz = -dy
        
        # Red color in BGR 
        color = (0, 0, 0)        
        # Line thickness of 9 px 
        thickness = 1
        
        image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
        
        
        
        
        # Adding predicted speed and angle text
        label1 = "Speed : "+str(f)
        label2 = "Angle : "+str(theta)
        (left, top) =  (int(car_centre[0]-0.5*self.car_width), int(car_centre[1]-0.5*self.car_height))   
        (right, bottom) = (int(car_centre[0]+0.5*self.car_width), int(car_centre[1]+0.5*self.car_height)) 
        font_scale = 0.3
        color= (0,0,0)
        cv2.putText(image , label1, (right, top), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        cv2.putText(image , label2, (right, top + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
                
        return image


    
    def intersection(self,rec_cordinates,car_centre):
        rect1 = ( (car_centre[0],car_centre[1]) , (self.car_width,self.car_height) ,0) # 0 is the angle of rectagle in clockwise
        
        output = []
        for rec in rec_cordinates:
            rect2 = ((rec[0],rec[1]) , (rec[2],rec[3]) , 0)
            r1 = cv2.rotatedRectangleIntersection(rect1, rect2)
            if r1[0]!=0 : #rect1 and rect2 are intersected or either one rectagle is present inside other
                area = cv2.contourArea(r1[1])   #area of intersection
                output.append(area/(self.car_width*self.car_height)) #amount of intersection 
                # print(100*area/(car_width*car_height)) 
          
        if len(output) ==0 :
            return 0
        
        return max(output)
             
    
    def lane_angle(self,lane_cordinates):
        theta_lane = []
        
        for i in range(len(lane_cordinates)):    
                    lane = lane_cordinates[i]
                    p1,q1 = lane[0]
                    p2,q2 = lane[-1]
                    
                    if p1==p2 :
                        temp = 90
                    else:
                        temp = np.arctan(-(q2-q1)/(p2-p1))*180/np.pi  # since dz = -dy
                    if temp < 0 :
                        theta_lane.append(180+temp)
                    else:
                        theta_lane.append(temp)
         
        
        if len(theta_lane)!=0 : 
             # taking the mean of theta_lena
             theta_lane = sum(theta_lane)/len(theta_lane)
             
             if theta_lane % 10 <= 5 : 
                theta_lane = 10*int(theta_lane/10)
             else:
                theta_lane = 10*(int(theta_lane/10)+1)
                 
        else:
             theta_lane = 90
        
        
        return theta_lane #wrt to x axis 
                
        

    def get_lane_limits(self,lane_cordinates):
               
        p_min = self.image_shape 
        p_max = 0 
        
        
        for i in range(len(lane_cordinates)):    
                lane = lane_cordinates[i]                
                
                # for first point the z cordinate (here y cordinate) must be near to image_shape
                p,q = lane[0]  # taking only the first point in every lane assuming line joining car_centre[0] and first point in a lane is paralle to x axis                 
                p_min = min(p_min,p)
                p_max = max(p_max,p)
                
        if p_min < p_max :
    
            return p_min , p_max
    
        else :
            return 0.20*self.image_shape , 0.80*self.image_shape
        
        
        
    def distance_from_lane(self,lane_cordinates,car_center,a):
               
        # find two nearest lanes to car        
        distance = {}
        done = 0
        p_min = self.image_shape 
        p_max = 0 
        
        
        for i in range(len(lane_cordinates)):    
                lane = lane_cordinates[i]                
                
                # for first point the z cordinate (here y cordinate) must be near to image_shape
                p,q = lane[0]  # taking only the first point in every lane assuming line joining car_centre[0] and first point in a lane is paralle to x axis                 
                distance[abs(p-car_center[0])] = p
                p_min = min(p_min,p)
                p_max = max(p_max,p)
                
          
        if p_min < p_max and (int(car_center[0]+0.5*self.car_width) < p_min or int(car_center[0]-0.5*self.car_width) > p_max) :  
            done = 1
        
        if len(distance)>=2:
            sorted_keys = list( sorted(distance.keys()) )
            l1 = distance[sorted_keys[0]]
            l2 = distance[sorted_keys[1]]
            
        elif len(distance)==1:
            for key in distance:
                l1 = distance[key]
                l2 = distance[key]
            
        else:
            l1 = 0
            l2 = self.image_shape 
            
            
        # find mid point off those two lanes
        l = (l1+l2)/2
                
        
        #find distance between car and midpoint
        alpha = 1
        beta = 1
        
        theta_lane = self.lane_angle(lane_cordinates)
        a = a - theta_lane
        
        return alpha*(abs(np.cos(a*np.pi/180))-abs(np.sin(a*np.pi/180))) - beta * abs(l-car_center[0])/self.image_shape  , done
        
            
    def reward_intersection(self,i, theta , f , car_centre  ,theta_action , f_action , car_centre1 ): 
                            
            
              rec_cordinates = self.video_rec_cordinates[i]
              rec_cordinates1 = self.video_rec_cordinates[np.clip(i+f_action,0,self.n-1)]
              
              max_intersection = self.intersection(rec_cordinates,car_centre)* np.cos((theta-90)*np.pi/180)
              max_intersection1 = self.intersection(rec_cordinates1,car_centre1)* np.cos((theta_action-90)*np.pi/180)
             
              alpha = 1 #action_space_f
              beta = 1 #f_action
              gamma = 0.1 #f_action
              
              
              done  = 0
                
              lane_reward , done =  self.reward_lane(i, theta , f ,car_centre  ,theta_action , f_action , car_centre1 ) 
              
              if max_intersection1 > 0.5 : # collision if intersection > 0.5
                  done = 1
                  
              if i+f_action >= self.n :
                  done = 1
                  
              if f_action == 0 and f == 0 :
                      self.time_count +=1
              else : 
                  self.time_count = 1
                      
              if self.time_count > 5 : 
                  done = 1
                  
              if (max_intersection<self.threshold): #follow the same lane to encourage stablity

                  return (alpha*(f_action-f)/self.action_space_f)  + (beta*lane_reward) - (gamma* (max_intersection1)) , done
              
                
              else :                   #dont follow that lane   

                  return -(alpha*(f_action-f)/self.action_space_f) - (beta*lane_reward) - (gamma* (max_intersection1)) , done
                   
        
            
    def reward_lane(self , i, theta , f , car_centre  ,theta_action , f_action , car_centre1 ): 
        
            lane_cordinates = self.video_lane_cordinates[i]
            lane_cordinates1 = self.video_lane_cordinates[np.clip(i+f_action,0,self.n-1)]
            
            distance , done = self.distance_from_lane(lane_cordinates,car_centre,theta)
            distance1, done1 = self.distance_from_lane(lane_cordinates1,car_centre1,theta_action)
        
            #using only distance1 as reward (absolute reward)
            return distance1,done1  # the problem with relative reward (distance1-distance) is that even if both distance and distance1 are high , we will get a low reward. 
    
    
    
    def preprocess(self, s1 , s2 ):
              # objects not detecting in 224x224 image , lane marking are good in 224x224 image
              segmentation_img , rec_cordinates  = mask.get_image(s1 , s2)  # segmentation_img is unnormalized numpy array 
              lane_img , lane_cordinates  = ld.get_image(s1 , s2) # lane_img is unnormalized numpy array               
              depth_img = depth.get_image(s1 , s2)  # depth_img is unnormalized numpy array 
              
              return segmentation_img , lane_img ,depth_img, lane_cordinates , rec_cordinates
          
        
    def get_frames(self):
        vidcap = cv2.VideoCapture(self.video_path)   
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        frames_step=int(total_frames/self.n)
        video_frames=[]
        video_rec_cordinates=[]
        video_lane_cordinates=[]
        
        for i in range(0,self.n):
             #here, we set the parameter 1 which is the frame number to the frame (i*frames_step)
             vidcap.set(1,i*frames_step)
             success,image = vidcap.read() 
             
             if image is not None :     # image shape is (720, 1280, 3)             
                # imshow(image)
                image1 = cv2.resize(image, (self.image_shape,self.image_shape), interpolation = cv2.INTER_AREA)
                image2 = cv2.resize(image, (720,720), interpolation = cv2.INTER_AREA) # for mask-rcnn and lane_detection and depth maps , taking image to higer dimension .
                          
                try :   # we might get errors due these mask rcnn or lane detection model , depth map .   
                 
                   segmentation_img ,lane_img , depth_img , lane_cordinates , rec_cordinates = self.preprocess(image1 , image2 )
                   video_frames.append([image1,segmentation_img,depth_img])                  
                   video_lane_cordinates.append(lane_cordinates)
                   video_rec_cordinates.append(rec_cordinates)
                   
                except Exception as e:
                      print("skipping the frame due to the exception " + str(e))
                  
                    
        self.n = len(video_frames)      
                           
                   
        return video_frames , video_lane_cordinates , video_rec_cordinates
    
    
    def execute_action(self, car_centre , theta_lane ,  a ,f):
        
        c = 1
        
            
        car_centre1 = [np.clip(int(car_centre[0]+c*f*np.sin((theta_lane+a)*np.pi/180)/np.sin(a*np.pi/180) ) ,0, self.image_shape-1) , car_centre[1]]
        

        return car_centre1
    
        
    def get_image(self,i,car_centre,r,f,theta,train_network):  
        
       
        i = np.clip(i,0,self.n-1)
        s = self.video_frames[i] 
            
        
        if train_network==0:  #run below two lines during testing to visualise our car
            
            display_img = self.draw_car(s[0], car_centre, r, f, theta) # due to steering 
            self.imshow(display_img) # plotting the cuurent action in the current state
            return (s[1],s[2]) , display_img
        
        return s[1],s[2]  # returning only segmenation and depth map 
        



