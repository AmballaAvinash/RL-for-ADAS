#link : https://github.com/ucbdrive/pyTORCS#install-torcs-and-pytorcs

import numpy as np
import matplotlib.pyplot as plt
import cv2
import gym
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd



from py_TORCS.py_TORCS import torcs_env



          
env = torcs_env() #torcs_envs(num = 1, game_config=game_config, isServer = 0, continuous=True, resize=False)       
game_config = '/home/avinash/Desktop/projects/Self_driving_car/TORCS/py_TORCS/game_config/michigan.xml'
env.init(game_config=game_config, isServer=0, continuous=False, resize=False)
env.reset()
# info=env.get_info() #return info about the state such as speed etc 
# s=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"],info["pos"][0],info["pos"][1],info["pos"][2]])
# print(s)



class ActorCritic(torch.nn.Module):
   def __init__(self , action_space , image_shape):
        super(ActorCritic, self).__init__()
        
        self.action_space = action_space
        self.image_shape = image_shape
        
        self.conv1 = torch.nn.Conv2d(3, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)

   
        
        self.critic_linear1 = torch.nn.Linear(500+4, 100)
        self.critic_linear2 = torch.nn.Linear(100, 1)

        self.actor_linear1 = torch.nn.Linear(500+4, 100)
        self.actor_linear2 = torch.nn.Linear(100, self.action_space)
        
        
        
    
    
   def forward(self, s):
        
        
        # do normalization for input images before passing 

        image = cv2.resize(s[0], (self.image_shape,self.image_shape), interpolation = cv2.INTER_AREA)/255       
        x = torch.from_numpy(image.reshape(3,self.image_shape,self.image_shape)).float().unsqueeze(0)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(-1, 64 *10* 10) # flattening and also -1 means batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
 
        
        
        # print(s[1]) # s[1] is numpy array
                
        y = torch.from_numpy(s[1]).unsqueeze(0)
        
        x = torch.cat((x , y.float()) , 1)
               
        
        value = F.relu(self.critic_linear1(x))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(x))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist


   def select_action(self, s):
        value, policy_dist = self.forward(s)
        value = value.detach().numpy()[0,0]
        dist = policy_dist.detach().numpy() 

        action = np.random.choice(self.action_space, p=np.squeeze(dist))
        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))

        return value , action , log_prob , entropy



   



def trajectories():
        
            # plt.imshow(image)

            rew=0
            rewards = []
            
            log_probs = []
            values = []
                    
            image = env.reset() #returns an image 
            info=env.get_info() #return info about the state such as speed etc 
            sensor_info=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"]])
            
            state=[image,sensor_info]                                
            while True : # done = 0
                    value ,  action , log_prob , entropy  = net.select_action(state)
                    
                    values.append(value)
                    log_probs.append(log_prob)
                                                       
                    image1, reward, done, info = env.step(action) 
                    
                    sensor_info1=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"]])
            
                    state1=[image1,sensor_info1]
            
                    state = state1
                    rewards.append(reward)     
                    rew += reward
                    
                    if done:
                               
                        Qval, _ = net.forward(state)
                        Qval = Qval.detach().numpy()[0,0]
                             
                        break
            
                
            return values , log_probs , rewards , rew , Qval
        
    
        

#state_space = 
action_space = 9 # we are predicting only accel, brake , steering with 9 combinations
sensor_info_size = 4
image_shape = 224


net = ActorCritic(action_space,image_shape) # we are predicting only steering angle.
optimizer = torch.optim.Adam(net.parameters(), lr=5e-6 )
     

gamma=0.99
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
      
        values , log_probs , rewards , rew , Qval = trajectories() 
                
        print("episode {}---->reward {}".format(t,rew))
        reward.append(rew)
        reward_avg.append(np.mean(reward[-50:]))  #last 50 rewards
        reward_std.append(np.std(reward[-50:]))  #last 10 rewards
        
  
        
        # compute Q values and update actor critic
        Qvals = np.zeros_like(values)
        for i in reversed(range(len(rewards))):
            Qval = rewards[i] + gamma * Qval
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
        
        
        
        if (t+1)%100 == 0 :
            
            x = np.linspace(1,t+1,t+1)
            plt.plot(x,reward,color="black")
            plt.plot(x,reward_avg,color="green")
            plt.legend(['Reward',"Mean Reward"])
            plt.xlabel('Episode')
            plt.ylabel('Reward')           
            plt.savefig('output_plots/test7.png')
            plt.show()
            
            
            plt.plot(x,overall_loss)
            plt.legend(['Loss'])
            plt.xlabel('Episode')
            plt.ylabel('Loss')            
            plt.savefig('output_plots/loss7.png')
            plt.show()
        
        
            # saving to .csv 
            dic = {'Episode': x, 'Reward': reward, 'Mean Reward': reward_avg , "Std" : reward_std} 
             
            df = pd.DataFrame(dic)
              
            # saving the dataframe
            df.to_csv('output_plots/CSV/A2C_single_network_TORCS.csv')
            
            
            #saving the model
            torch.save(net, "saved_model_A2C_SN.pt")
        

env.close()





