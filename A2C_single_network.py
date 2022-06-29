from environment import Environment

from algorithm import AdvantageActorCriticSN

from utils import *

import numpy as np
import cv2
from PIL import Image 
import matplotlib.pyplot as plt
import os
import torchvision.models as models
import pandas as pd

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms


from sklearn.model_selection import train_test_split

# preprocessing 
input_path = "driving_dataset/videos/ex1_cropped.mp4"
output_path = "dataset/trimmed_video1_"
n = 100  # no of frames in a single video
c = preprocess(input_path , output_path , n)
c = min(c,20)# taking 20 videos 
print(c)
dataset = np.arange(0,c)
train , test = train_test_split(dataset , test_size = 0.2)



# parameters
threshold = 0.2
car_width = 40
car_height = 40
image_shape = 224
# frame_step = int(total_frames/n)



# state_space = entire_image
action_space_f = np.arange(6)  #[0,1,2,...n-1]
action_space_theta = (np.arange(17)+1)*10  # [ideadlly 0  to  180 degrees continuous but we are taking fininte discrete intervals ,  0 and 180 not included ]
 

action_space = []
for i in range(len(action_space_f)):
    for j in range(len(action_space_theta)):
        action_space.append([action_space_f[i],action_space_theta[j]])
        

#initilizing all the enviroments
ENV = []
for i in range(c):
        ENV.append(Environment(n,threshold,car_width,car_height,image_shape,output_path+str(i)+".mp4" , len(action_space_f) , len(action_space_theta)))
        
        print("Fininshed Initializing the environment {}" .format(i))





def trajectories():        

        rew=0
        rewards = []
        
       
        log_probs = []
        values = []
                
                
        #resetting the environemt
        x = np.random.randint(0,len(train))
        env = ENV[train[x]]     
        
        # random initialization .
        
        lane_cordinates = env.video_lane_cordinates[0]
        
        p , q = env.get_lane_limits(lane_cordinates)
        
        car_centre = [np.random.randint(int(p),int(q)),int(image_shape-0.5*car_height-1)]
        
        r = 20 #half of car_width
        theta = 10*(np.random.randint(0,len(action_space_theta))+1)
        f = 0  #by default f = 0
        
        i=0
        
        done = 0 
        
        while  done==0 : #runs unless done = 1
            image = env.get_image(i,car_centre,r,f,theta,train_network)
               
            p , q = env.get_lane_limits(env.video_lane_cordinates[i])
                   
            theta_lane = env.lane_angle(env.video_lane_cordinates[i])
        
            state = [image ,car_centre , f , theta , p , q , theta_lane]
            
            value ,  action , log_prob , entropy = net.select_action(state)
            
            f_action = action_space[action][0]
            theta_action = action_space[action][1]
            
            
            values.append(value)
            log_probs.append(log_prob)
            
     
            car_centre1 = env.execute_action(car_centre , theta_lane , theta_action ,f_action) 
            
            
            reward,done = env.reward_intersection(i, theta , f , car_centre  ,theta_action , f_action , car_centre1)
         
         
                
            i+=f_action       # i+=f_action + k where k is no of frames to skip.       
            car_centre = car_centre1
            theta = theta_action
            f = f_action
            
            
            rewards.append(reward)                 
            rew=rew+reward
            
        
        image = env.get_image(i,car_centre,r,f,theta,train_network)
               
        p , q = env.get_lane_limits(env.video_lane_cordinates[-1])
        
        theta_lane = env.lane_angle(env.video_lane_cordinates[-1])
        
        state = [image ,car_centre , f , theta , p , q , theta_lane]
        
        Qval, _ = net.forward(state)
        Qval = Qval.detach().numpy()[0,0]
        
            
        return values , log_probs , rewards, rew , Qval





net = AdvantageActorCriticSN(len(action_space_f) , len(action_space_theta) , image_shape , len(action_space)) 

optimizer = torch.optim.Adam(net.parameters(), lr=5e-6)

   


GAMMA = 0.99


N = 1500
reward=[]
reward_avg = []
reward_std = []
overall_loss = []


train_network = 1

if train_network == 1 : 
    # training the agent 
    print("training started")
    for t in range(N):
          

        values , log_probs , rewards, rew ,  Qval  = trajectories() 
               
        print("episode {}---->reward {}".format(t,rew))
        reward.append(rew)
        reward_avg.append(np.mean(reward[-50:]))  #last 10 rewards
        reward_std.append(np.std(reward[-50:]))  #last 10 rewards
        
        # compute Q values for  network and update actor critic
        Qvals = np.zeros_like(values)
        for i in reversed(range(len(rewards))):
            Qval = rewards[i] + GAMMA * Qval
            Qvals[i] = Qval
  

        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()
        
        overall_loss.append(ac_loss.item())
        
                         
        
        if (t+1)%100 == 0:
            
            
            
            x = np.linspace(1,t+1,t+1)
            plt.plot(x,reward,color="black")
            plt.plot(x,reward_avg,color="green")
            plt.legend(['Reward',"Mean Reward"])
            plt.xlabel('Episode')
            plt.ylabel('Reward')           
            plt.savefig('output_plots/Custom_CNN/test90.png')
            plt.show()
            
            
            plt.plot(x,overall_loss)
            plt.legend(['Loss '])
            plt.xlabel('Episode')
            plt.ylabel('Loss')            
            plt.savefig('output_plots/Custom_CNN/loss90.png')
            plt.show()
        
        
            # saving to .csv 
            dic = {'Episode': x, 'Reward': reward, 'Mean Reward': reward_avg , "Std" : reward_std} 
             
            df = pd.DataFrame(dic)
              
            # saving the dataframe
            df.to_csv('output_plots/Custom_CNN/CSV/A2C_single_network_ex1.csv')
            
            
            #saving the model
            torch.save(net, "saved_model_A2C_SN.pt")
        
    



else:

    
    #testing the agent on entire video
    
    #loading the model
    net = torch.load("saved_model_A2C_SN.pt")
    
    
    
    test_reward = []
    for x in range(len(test)):
        
        env = ENV[test[x]]
        
        # random initialization at starting frame
        lane_cordinates = env.video_lane_cordinates[0]
        
        p , q = env.get_lane_limits(lane_cordinates)
        
        car_centre = [np.random.randint(int(p),int(q)),int(image_shape-0.5*car_height-1)]
    
        r = 20   
        theta = 10*(np.random.randint(0,action_space_theta)+1)
        f = 0  #by default f = 0 , car at rest
        
        i=0
        rew = 0        
        done = 0
        while done==0: #runs unless done = 1
        
            image , display_img = env.get_image(i,car_centre,r,f,theta,train_network)
            
            p , q = env.get_lane_limits(env.video_lane_cordinates[i])
        
            theta_lane = env.lane_angle(env.video_lane_cordinates[i])
        
            state = [image ,car_centre , f , theta , p , q , theta_lane]
            
            value ,  action , log_prob , entropy = net.select_action(state)
            
            f_action = action_space[action][0]
            theta_action = action_space[action][1]
            
            
            print("speed {}  ,  streering angle {}".format(f_action , theta_action))                
            
            car_centre1 = env.execute_action(car_centre , theta_lane , theta_action ,f_action) 
                                                    
            reward,done = env.reward_intersection(i, theta , f , car_centre  ,theta_action , f_action , car_centre1)
         
          
                     
            i+=f_action
                            
            car_centre = car_centre1
            theta = theta_action
            f = f_action
                                 
            rew=rew+reward
        
       
        print("reward for test video {} is {}".format(x,rew))
        test_reward.append(rew)
        
        
    print("Overall test reward for all the videos is {}".format(sum(test_reward)/len(test_reward)))
    
    
    


# testing on entire video

#loading the model
train_network = 0
net = torch.load("saved_model_A2C_SN.pt")

 # random initialization at starting frame
lane_cordinates = ENV[0].video_lane_cordinates[0]

p , q = ENV[0].get_lane_limits(lane_cordinates)

car_centre = [np.random.randint(int(p),int(q)),int(image_shape-0.5*car_height-1)]
                    

r = 20        
theta = 10*(np.random.randint(0,action_space_theta)+1)
f = 0  #by default f = 0 , car at rest
rew = 0   



# make output a video and display speed and angle on the top of the car box
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
cwd = os.getcwd()
output_file_name = os.path.join(cwd,"ex1_output_A2C_SN.mp4")
height, width, channels = (224,224,3)
out = cv2.VideoWriter(output_file_name, fourcc, 20.0, (width, height))


for x in range(c):
    
   env = ENV[x]
   i=0
   done = 0 
   
   while  done==0 : #runs unless done = 1

        image , display_img = env.get_image(i,car_centre,r,f,theta,train_network)        
        out.write(display_img)
               
        p , q = env.get_lane_limits(env.video_lane_cordinates[i])
    
        theta_lane = env.lane_angle(env.video_lane_cordinates[i])
        
        state = [image ,car_centre , f , theta , p , q , theta_lane]
        
        
        value ,  action , log_prob , entropy = net.select_action(state)
        
        f_action = action_space[action][0]
        theta_action = action_space[action][1]
            
        
        print("speed {}  ,  streering angle {}".format(f_action , theta_action))                
        
        car_centre1 = env.execute_action(car_centre , theta_lane , theta_action ,f_action) 
                                                
        reward,done = env.reward_intersection(i, theta , f , car_centre  ,theta_action , f_action , car_centre1)
     
        # if done == 1 : # done = 1 if agent collides with object or if agent moves out of road
        #     reward = -5
                
        i+=f_action                            
        car_centre = car_centre1
        theta = theta_action
        f = f_action
                             
        rew=rew+reward
    
out.release()
  
print("Overall test reward for all the entire video is {}".format(rew))
   


