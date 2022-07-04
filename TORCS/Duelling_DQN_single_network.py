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



class QNetwork(torch.nn.Module): # network is defined per image not per batch so if we give batch of images then we will get batch of outputs 
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

   
        
        self.fc_value = torch.nn.Linear(500+60, 100)
        self.value = torch.nn.Linear(100, 1)
        
        self.fc_adv = torch.nn.Linear(500+4, 100)
        self.adv = torch.nn.Linear(100,self.action_space)
        
        
               

    def forward(self, state):
        
        for i in range(len(state)):
             
            s = state[i]
            
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
            
            value = F.relu(self.fc_value(x))
            adv = F.relu(self.fc_adv(x))
    
            value = self.value(value)
            adv = self.adv(adv)
    
            advAverage = torch.mean(adv, dim=1, keepdim=True)
            Q = value + adv - advAverage
            
            
            if i==0:
                Q1 = Q
            else:
                Q1 = torch.cat((Q1,Q),0)
            
         return Q1
            


     def select_action(self, s):
        with torch.no_grad():
            Q = self.forward(s)
            action_index = torch.argmax(Q, dim=1) 
        return action_index.item()
 
   



def trajectories():
                            
        rew=0
        global epsilon
        global begin_learn
        global learn_steps
       
                
        image = env.reset() #returns an image 
        info=env.get_info() #return info about the state such as speed etc 
        sensor_info=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"]])
        
        state=[image,sensor_info]                                
        while True : # done = 0
        
                p = np.random.rand()
                if p < epsilon:
                    action = np.random.randint(0,action_space)
                    
                else:
                    action = onlineQNetwork.select_action([state])        
                                            
                                       
                image1, reward, done, info = env.step(action) 
                
                sensor_info1=np.array([info["speed"],info["angle"],info["trackPos"],info["trackWidth"]])
        
                state1=[image1,sensor_info1]
        
                state = state1
                rew += reward
                
                # T.append(state)
                # T.append(action)
                P.append(p)
                g.append(reward)
            
                memory_replay.add((state, state1, action, reward, done))
                
                
               # updatng newtork 
                loss = 0
                if memory_replay.size() > 128:
                                   
                    if begin_learn is False:
                        print('learn begin!')
                        begin_learn = True
                    learn_steps += 1
                    if learn_steps % UPDATE_STEPS == 0:
                        targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
                    batch = memory_replay.sample(BATCH, False)
                    batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)
                    
        
                    batch_action = torch.FloatTensor(batch_action).unsqueeze(1)
                    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
                    batch_done = torch.FloatTensor(batch_done).unsqueeze(1)
        
                    with torch.no_grad():
                        onlineQ_next = onlineQNetwork(batch_next_state)
                        targetQ_next = targetQNetwork(batch_next_state)
                        online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
                        y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, online_max_action.long())
        
    
                    loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
                    if epsilon > FINAL_EPSILON:
                        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                        
                    
                if done:
                    # reward = -5
                    break
    
        return loss  , rew
        

class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()








#state_space = 
action_space = 9 # we are predicting only accel, brake , steering with 9 combinations
sensor_info_size = 4
image_shape = 224


REPLAY_MEMORY = 5000
memory_replay = Memory(REPLAY_MEMORY)

onlineQNetwork = QNetwork(action_space, image_shape ) 
targetQNetwork = QNetwork(action_space, image_shape ) 
targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=5e-6)

    



GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
BATCH = 16
UPDATE_STEPS = 4

epsilon = INITIAL_EPSILON
learn_steps = 0
begin_learn = False



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
        
        loss , rew = trajectories() 
        print("episode {}---->reward {}".format(t,rew))
        reward.append(rew)
        reward_avg.append(np.mean(reward[-50:]))  #last 10 rewards
        reward_std.append(np.std(reward[-50:]))  #last 10 rewards
        
        overall_loss.append(loss)
                    
    
        
        # print("loss {} ".format(loss.item()))
        
        if (t+1)%100 == 0:
            
            x = np.linspace(1,t+1,t+1)
            plt.plot(x,reward,color="black")
            plt.plot(x,reward_avg,color="green")
            plt.legend(['Reward',"Mean Reward"])
            plt.xlabel('Episode')
            plt.ylabel('Reward')           
            plt.savefig('output_plots/Custom_CNN/test87.png')
            plt.show()
            
            
            plt.plot(x,overall_loss)
            plt.legend(['Loss'])
            plt.xlabel('Episode')
            plt.ylabel('Loss')            
            plt.savefig('output_plots/Custom_CNN/loss87.png')
            plt.show()
        
                   
            # saving to .csv 
            dic = {'Episode': x, 'Reward': reward, 'Mean Reward': reward_avg , "Std" : reward_std} 
             
            df = pd.DataFrame(dic)
              
            # saving the dataframe
            df.to_csv('output_plots/CSV/Duelling_DQN_single_network_TORCS.csv')
        
            #saving the model
            torch.save(net, "saved_model_Duelling_DQN_SN.pt")
        

env.close()





