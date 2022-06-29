from environment import Environment

from algorithm import AdvantageActorSV , AdvantageCriticSV

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
action_space_f = 6  #[1,2,...n-1]
action_space_theta = 17  # [ideadlly 0  to  180 degrees continuous but we are taking fininte discrete intervals ,  0 and 180 not included ]
 


    

#initilizing all the enviroments
ENV = []
for i in range(c):
        ENV.append(Environment(n,threshold,car_width,car_height,image_shape,output_path+str(i)+".mp4" , action_space_f , action_space_theta))
        
        print("Fininshed Initializing the environment {}" .format(i))








def trajectories():        

        rew=0
        rewards = []
    
        f_log_probs = []
        theta_log_probs = []
        
        values = []
        
                
        #resetting the environemt
        x = np.random.randint(0,len(train))
        env = ENV[train[x]]     
        
        # random initialization .
        
        lane_cordinates = env.video_lane_cordinates[0]
        
        p , q = env.get_lane_limits(lane_cordinates)
        
        car_centre = [np.random.randint(int(p),int(q)),int(image_shape-0.5*car_height-1)]
        
        r = 20 #half of car_width
        theta = 10*(np.random.randint(0,action_space_theta)+1)
        f = 0  #by default f = 0
        
        i=0
        
        done = 0 
        
        while  done==0 : #runs unless done = 1 or -1
            image = env.get_image(i,car_centre,r,f,theta,train_network)
               
            p , q = env.get_lane_limits(env.video_lane_cordinates[i])
                   
            theta_lane = env.lane_angle(env.video_lane_cordinates[i])
        
            state = [image ,car_centre , f , theta , p , q , theta_lane]
            
            
            
            f_action , f_log_prob , f_entropy = net_speed.select_action(state)
            
            theta_action , theta_log_prob , theta_entropy = net_angle.select_action(state)
            theta_action =  10*(theta_action+1)
            
            f_log_probs.append(f_log_prob)
            
            theta_log_probs.append(theta_log_prob)
            
            
            value = net_value.forward(state)            
            values.append(value)
            
           
        
         
            
            car_centre1 = env.execute_action(car_centre , theta_lane , theta_action ,f_action) 
            
            
            reward,done = env.reward_intersection(i, theta , f , car_centre  ,theta_action , f_action , car_centre1)
         
                        
            # if done == 1 : # if agent reached end of video
            #     reward = 1
            # elif done == -1 : # done = -1 if agent collides with object or if agent moves out of road
            #     reward = -1
                
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
                                    
        
        Qval = net_value.forward(state)
       
        return values , f_log_probs , theta_log_probs , rewards , rew , Qval 






net_speed = AdvantageActorSV(action_space_f , action_space_theta , image_shape , "speed")     # define the network
net_angle = AdvantageActorSV(action_space_f , action_space_theta , image_shape , "angle") 
net_value =  AdvantageCriticSV(action_space_f , action_space_theta , image_shape )

optimizer_speed = torch.optim.Adam(net_speed.parameters(), lr=5e-6 )
optimizer_angle = torch.optim.Adam(net_angle.parameters(), lr=5e-6 )
optimizer_value = torch.optim.Adam(net_value.parameters(), lr=5e-6 )
   

GAMMA = 0.99

        

N = 1500
reward=[]
reward_avg = []
reward_std = []
overall_loss_speed = []
overall_loss_angle = []
overall_loss_value = []

train_network = 1

if train_network == 1 : 
    # training the agent 
    print("training started")
    for t in range(N):
          

        values , f_log_probs , theta_log_probs , rewards , rew , Qval  = trajectories() 
               
        print("episode {}---->reward {}".format(t,rew))
        reward.append(rew)
        reward_avg.append(np.mean(reward[-50:]))  #last 10 rewards
        reward_std.append(np.std(reward[-50:]))  #last 10 rewards
        
        # compute Q values for critic
        Qvals = np.zeros_like(values)
        for i in reversed(range(len(rewards))):
            Qval = rewards[i] + GAMMA * Qval
            Qvals[i] = Qval
  
        values = torch.stack(values)
        Qvals = torch.stack(Qvals.tolist())
        advantage = Qvals - values
        
        # actor speed
        f_log_probs = torch.stack(f_log_probs)
        f_actor_loss = (-f_log_probs * advantage.detach()).mean()        
        optimizer_speed.zero_grad()
        f_actor_loss.backward()
        optimizer_speed.step()        
        overall_loss_speed.append(f_actor_loss.item())
        
                
        # actor angle
        theta_log_probs = torch.stack(theta_log_probs)               
        theta_actor_loss = (-theta_log_probs * advantage.detach()).mean()
        optimizer_angle.zero_grad()
        theta_actor_loss.backward()
        optimizer_angle.step()        
        overall_loss_angle.append(theta_actor_loss.item())
                
        #value critic 
        critic_loss = 0.5 * advantage.pow(2).mean()
        optimizer_value.zero_grad()
        critic_loss.backward()
        optimizer_value.step()        
        overall_loss_value.append(critic_loss.item())
        
        
    
        # print("loss_speed  {}  , loss angle {} ".format(loss_speed.item() , loss_angle.item()))
        
        if (t+1)%100 == 0:
            
            x = np.linspace(1,t+1,t+1)
            plt.plot(x,reward,color="black")
            plt.plot(x,reward_avg,color="green")
            plt.legend(['Reward',"Mean Reward"])
            plt.xlabel('Episode')
            plt.ylabel('Reward')           
            plt.savefig('output_plots/Custom_CNN/test92.png')
            plt.show()
            
            
            plt.plot(x,overall_loss_speed)
            plt.plot(x,overall_loss_angle)
            plt.plot(x,overall_loss_value)
            plt.legend(['Loss speed',"Loss angle","Loss value"])
            plt.xlabel('Episode')
            plt.ylabel('Loss')            
            plt.savefig('output_plots/Custom_CNN/loss92.png')
            plt.show()
        
        
            
            # saving to .csv 
            dic = {'Episode': x, 'Reward': reward, 'Mean Reward': reward_avg , "Std" : reward_std} 
             
            df = pd.DataFrame(dic)
              
            # saving the dataframe
            df.to_csv('output_plots/Custom_CNN/CSV/A2C_Single_V_ex1.csv')
            
            
            #saving the model
            torch.save(net_speed, "saved_model_A2C_speed.pt")
            torch.save(net_angle, "saved_model_A2C_angle.pt")
            torch.save(net_value, "saved_model_A2C_value.pt")
        
            


else:

    
    #testing the agent on entire video
    
    #loading the model
    net_speed = torch.load("saved_model_A2C_speed.pt")
    net_angle = torch.load("saved_model_A2C_angle.pt")
    net_value = torch.load("saved_model_A2C_value.pt")
    
    
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
                        
                                
            f_action , f_log_prob , f_entropy = net_speed.select_action(state)
            
            theta_action , theta_log_prob , theta_entropy = net_angle.select_action(state)
            theta_action =  10*(theta_action+1)
                            
            
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
        
       
        print("reward for test video {} is {}".format(x,rew))
        test_reward.append(rew)
        
        
    print("Overall test reward for all the videos is {}".format(sum(test_reward)/len(test_reward)))
    
    
    
    



# testing on entire video
    
#loading the model
train_network = 0
net_speed = torch.load("saved_model_A2C_speed.pt")
net_angle = torch.load("saved_model_A2C_angle.pt")
net_value = torch.load("saved_model_A2C_value.pt")

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
output_file_name = os.path.join(cwd,"ex1_output_A2C_SV.mp4")
height, width, channels = (224,224,3)
out = cv2.VideoWriter(output_file_name, fourcc, 20.0, (width, height))



for x in range(c):
    
   env = ENV[x]
   i=0
   done = 0 
   
   while  done==0 : #runs unless done = 1

        image , display_img = env.get_image(i,car_centre,r,f,theta,train_network)        
        out.write(display_img)
        
        
        theta_lane = env.lane_angle(env.video_lane_cordinates[i])
        
        state = [image ,car_centre , f , theta , p , q , theta_lane]


        theta_lane = env.lane_angle(env.video_lane_cordinates[i])
                    
                            
        f_action , f_log_prob , f_entropy = net_speed.select_action(state)
            
        theta_action , theta_log_prob , theta_entropy = net_angle.select_action(state)
        theta_action =  10*(theta_action+1)
                        
        
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



