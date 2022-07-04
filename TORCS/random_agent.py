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

   
def random_agent(action_space):
        return np.random.randint(0,action_space)
    
def intelligent_agent(info, continuous=False):
    if info['angle'] > 0.2 or (info['trackPos'] < -2 and info['angle'] > 0):
        return np.array([0.5, 1]) if continuous else 0
    elif info['angle'] < -0.2 or (info['trackPos'] > 2 and info['angle'] < 0):
        return np.array([0.5, -1]) if continuous else 2
    return np.array([0.5, 0]) if continuous else 1


          
env = torcs_env() #torcs_envs(num = 1, game_config=game_config, isServer = 0, continuous=True, resize=False)       
game_config = '/home/avinash/Desktop/projects/Self_driving_car/TORCS/py_TORCS/game_config/michigan.xml'
env.init(game_config=game_config, isServer=0, continuous=False, resize=False)
env.reset()
# info=env.get_info() #return info about the state such as speed etc 
# s=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"],info["pos"][0],info["pos"][1],info["pos"][2]])
# print(s)



def trajectories():
        
        for i_episode in range(K):
                        
                              
                # plt.imshow(image)
               
                rew=0
                        
                image = env.reset() #returns an image 
                info=env.get_info() #return info about the state such as speed etc 
                sensor_info=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"]])
                
                state=[image,sensor_info]                                
                while True : # done = 0
                
                        #random 
                        action  = random_agent(action_space)
                       
                        image1, reward, done, info = env.step(action) 
                        
                        sensor_info1=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"]])
                
                        state1=[image1,sensor_info1]
                
                        state = state1
                        rew += reward
                                       
                        
                        if done:
                            # reward = -5
                            break
                

                R.append(rew)
            
    
        
        

#state_space = 
action_space = 9 # we are predicting only accel, brake , steering with 9 combinations
sensor_info_size = 4
image_shape = 224

N = 1500
reward=[]
reward_avg = []
reward_std = []

train_network = 1

if train_network == 1 : 
    # training the agent 
    for t in range(N):
        K = 1

        R=[]
        trajectories()   # input x and predict based on x
        print("episode {}---->reward {}".format(t,sum(R)/float(K)))
    
        reward.append(sum(R)/float(K))
        reward_avg.append(np.mean(reward[-50:]))
        reward_std.append(np.std(reward[-50:]))  #last 10 rewards
 
      
        # print("loss {} ".format(loss.item()))
        
        if (t+1)%100 == 0 :
            
            x = np.linspace(1,t+1,t+1)
            plt.plot(x,reward,color="black")
            plt.plot(x,reward_avg,color="green")
            plt.legend(['Reward',"Mean Reward"])
            plt.xlabel('Episode')
            plt.ylabel('Reward')           
            plt.savefig('output_plots/test9.png')
            plt.show()
            
            
            # saving to .csv 
            dic = {'Episode': x, 'Reward': reward, 'Mean Reward': reward_avg , "Std" : reward_std} 
             
            df = pd.DataFrame(dic)
              
            # saving the dataframe
            df.to_csv('output_plots/CSV/random_agent_TORCS.csv')
            
            

env.close()





