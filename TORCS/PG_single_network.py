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
# env.close()
# info=env.get_info() #return info about the state such as speed etc 
# s=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"],info["pos"][0],info["pos"][1],info["pos"][2]])
# print(s)



class PolicyNetwork(torch.nn.Module): # network is defined per image not per batch so if we give batch of images then we will get batch of outputs 
    def __init__(self , action_space , image_shape):
        super(PolicyNetwork, self).__init__()
        
        self.action_space = action_space
        self.image_shape = image_shape
        
        
        self.conv1 = torch.nn.Conv2d(3, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)

        self.fc3 = torch.nn.Linear(500+4, 100)
        self.fc4 = torch.nn.Linear(100, self.action_space)   
        
    
        
               

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
        
    
               
        # no embedding
        # print(s[1]) # s[1] is numpy array               
        y = torch.from_numpy(s[1]).unsqueeze(0)
        
        
        x = torch.cat((x , y.float()) , 1)
                
        z = F.relu(self.fc3(x))
        z = F.softmax(self.fc4(z))
        
        
        return z


    def select_action(self,s):  # s is an numpy image
                                     
        a = net.forward(s).squeeze(0)
        
        # sampling from distribution (try using np.random.choice)
        temp = 0
        y=torch.rand(1)
        for i in range(self.action_space):
            if temp<y and y<= temp + a[i] :
                action = i
                prob = -torch.log(a[i])
                
                break 
            
            else :
                temp = temp + a[i]
            
            
        return [action , prob]
 
   



def trajectories():
        
        for i_episode in range(K):
                        
                              
                # plt.imshow(image)
                g=[]
                T=[]
                P=[] # list of tensors
                rew=0
                        
                image = env.reset() #returns an image 
                info=env.get_info() #return info about the state such as speed etc 
                sensor_info=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"]])
                
                state=[image,sensor_info]                                
                while True : # done = 0
                        action , p  = net.select_action(state)
                        
                                               
                        image1, reward, done, info = env.step(action) 
                        
                        sensor_info1=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"]])
                
                        state1=[image1,sensor_info1]
                
                        state = state1
                        rew += reward
                        
                        # T.append(state)
                        # T.append(action)
                        P.append(p)
                        g.append(reward)
                    
                        
                        if done:
                            # reward = -5
                            break
                
                
            
                y1=[]
                for j in range(len(g)):
                  y2=0
                  for k in range(j,len(g)):
                    y2=y2+ pow(gamma,k-j)*g[k]
                  y1.append(y2)
                    
                G.append(y1)
                Traj.append(T)
                P1.append(P)
                R.append(rew)
            
    
        
        

#state_space = 
action_space = 9 # we are predicting only accel, brake , steering with 9 combinations
sensor_info_size = 4
image_shape = 224
net = PolicyNetwork(action_space,image_shape) # we are predicting only steering angle.
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
        K = 2  # change to 5 
        Traj=[]
        G=[]
        P1=[]
        R=[]
        trajectories()   # input x and predict based on x
        print("episode {}---->reward {}".format(t,sum(R)/float(K)))
    
        reward.append(sum(R)/float(K))
        reward_avg.append(np.mean(reward[-50:]))
        reward_std.append(np.std(reward[-50:]))  #last 10 rewards
                    
                    
        loss=[]
        for i in range(K):
          mean=sum(G[i])/len(G[i])
          for j in range(len(G[i])):
             loss.append(P1[i][j]*(G[i][j]-mean)/float(K))
            
        loss=sum(loss)
        overall_loss.append(loss.item())
        
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        
        
        
        
        # print("loss {} ".format(loss.item()))
        
        if (t+1)%100 == 0 :
            
            x = np.linspace(1,t+1,t+1)
            plt.plot(x,reward,color="black")
            plt.plot(x,reward_avg,color="green")
            plt.legend(['Reward',"Mean Reward"])
            plt.xlabel('Episode')
            plt.ylabel('Reward')           
            plt.savefig('output_plots/test8.png')
            plt.show()
            
            
            plt.plot(x,overall_loss)
            plt.legend(['Loss'])
            plt.xlabel('Episode')
            plt.ylabel('Loss')            
            plt.savefig('output_plots/loss8.png')
            plt.show()
        
            # saving to .csv 
            dic = {'Episode': x, 'Reward': reward, 'Mean Reward': reward_avg , "Std" : reward_std} 
             
            df = pd.DataFrame(dic)
              
            # saving the dataframe
            df.to_csv('output_plots/CSV/PG_single_network_TORCS.csv')
            
            #saving the model
            torch.save(net, "saved_model_PG_SN.pt")
        

env.close()





