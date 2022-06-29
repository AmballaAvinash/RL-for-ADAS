import numpy as np
import cv2
from PIL import Image 
import matplotlib.pyplot as plt

import moviepy.video.io.ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip

    
def random_agent(action_space_f , action_space_theta):
        return  np.random.randint(0,action_space_f) , 10*(np.random.randint(0,action_space_theta)+1)
    
def intelligent_agent(action_space_f , car_centre , p ,q, theta_lane):
    
    speed = int(0.5*action_space_f)   # returning avg spped
    
    if theta_lane < 70  or (car_centre[0] < (3*p+q)/4 and theta_lane < 90 ):
        return  speed , 50
    elif theta_lane > 110 or (car_centre[0] > (p+3*q)/4 and theta_lane > 90 ):
        return  speed , 130 
    return  speed , 90 



# # to make video clips from images
# image_folder = 'driving_dataset/driving_dataset'
# l = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# j = 1
# for i in range(len(l)):
#    if i!=0 and i%50==0:
#        clip = ImageSequenceClip.ImageSequenceClip(B, fps=1)
#        clip.write_videofile("driving_dataset/videos/video_"+str(j)+".mp4")
#        B = []
#        j+=1
       
#    B.append(os.path.join(image_folder, str(i)+".jpg"))

   



#preparing the dataset from single long video
def preprocess(video_path, output_path , n ) : 
    
    vidcap = cv2.VideoCapture(video_path)   
    fps = vidcap.get(cv2.CAP_PROP_FPS)  
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames/fps
     
    print("duration {} , total frames {} , fps {}".format(duration,total_frames,fps ))
    i = 0
    c = 0
    slide = 50
    with VideoFileClip(video_path) as video:
        while i<int(total_frames):  
            # getting only n frames  
            t1 = i/fps
            t2 = min(i+n,total_frames)/fps
            new = video.subclip(t1,t2)
            new.write_videofile(output_path+str(int(i/n))+".mp4")  
            i +=n
            # i+ = slide
            c +=1
            
    return c 