# Dueling DQN

import numpy as np
import cv2
from PIL import Image 
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from collections import deque



# Policy Gradients

class PolicyNetwork(torch.nn.Module): # network is defined per image not per batch so if we give batch of images then we will get batch of outputs 
    def __init__(self , action_space_f ,action_space_theta , image_shape , flag ):
        super(PolicyNetwork, self).__init__()
        
        self.image_shape = image_shape
        
        if flag == "speed" :
            self.action_space = action_space_f
        elif flag == "angle":
            self.action_space = action_space_theta
        

        ## pretrained models
        # self.vgg16 =   models.vgg16(pretrained=True)
        # self.inceptionv3 = models.inception_v3(pretrained=True)
        
        # evaluation mode
        #self.vgg16.eval()
        # self.inceptionv3.eval() 
        
        # feature_extract = True #  If feature_extract = False, the model is finetuned and all model parameters are updated. If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
        # self.set_parameter_requires_grad(self.inceptionv3, feature_extract)   
        
        
        self.conv1 = torch.nn.Conv2d(6, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        # self.gap = torch.nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)
        # self.fc3 = torch.nn.Linear(500, 100) 

        self.fc4 = torch.nn.Linear(500+60,100)        
        self.fc5 = torch.nn.Linear(100, self.action_space)
        
        
        
        self.d = 10
        self.emd_size = 10
        self.W = torch.randn((self.d,self.emd_size) , requires_grad = True)
        self.b = torch.zeros((self.d,1), requires_grad = True)        
        self.V = torch.randn((self.d,1), requires_grad = True)  
         
        self.embedding1 = torch.nn.Embedding(image_shape, self.emd_size)
        self.embedding2 = torch.nn.Embedding(action_space_f, self.emd_size)
        self.embedding3 = torch.nn.Embedding(action_space_theta, self.emd_size)
        
        
        

    def forward(self, s):
        
        # do normalization for input images before passing 

        segmentation_img = s[0][0]/255
        depth_img = s[0][1]/255
        
        image = np.concatenate([segmentation_img,depth_img],2)
        # image = segmentation_img
        
        x = torch.from_numpy(image.reshape(6,self.image_shape,self.image_shape)).float().unsqueeze(0)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # # x shape is batch_size*32*10*10
        # x = self.gap(x)
        # # x shape is batch_size*32*1*1
        
        x = x.view(-1, 64 *10* 10) # flattening and also -1 means batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))

        
        # x = self.inceptionv3(x)
        # x.shape is (1,1000)
              
        
        # attetnion
        # e = torch.mm(self.V.T ,torch.tanh(self.W*s[1][0]/self.image_shape + self.b))
                          
        # e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(self.W*s[2]/5 + self.b))) , 1)
        
        # e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(self.W*s[3]/180  + self.b))) , 1)        
        
        # e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(self.W*s[4]/self.image_shape  + self.b))) , 1)     
        
        # e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(self.W*s[5]/self.image_shape  + self.b))) , 1)  
        
        # a = 500*torch.softmax(e,1).squeeze(0)
        
        # y = torch.tensor([[ a[0]*s[1][0]/self.image_shape , a[1]*s[2]/5 , a[2]*s[3]/180 , a[3]*s[4]/self.image_shape , a[4]*s[5]/self.image_shape]] )
               
        
        # embedding
        # state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
        # state_f = self.embedding2(torch.tensor([int(s[2])]))  
        # state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
        
        
        # no embedding
        # state_f = torch.tensor([[int(s[2]-1)]])
        # state_theta = torch.tensor([[int(0.1*s[3])]])
        # state_car_centre = torch.tensor([[int(s[1][0])]])
        
        # y = torch.cat([state_f,state_theta,state_car_centre],1)
        
       

        # attetnion on embedding 
        state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
        state_f = self.embedding2(torch.tensor([int(s[2])]))
        state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
        state_left_cor = self.embedding1(torch.tensor([int(s[4])]))
        state_right_cor = self.embedding1(torch.tensor([int(s[5])]))
        theta_lane_cor = self.embedding3(torch.tensor([int(0.1*s[6]-1)]))
        

        e = torch.mm(self.V.T ,torch.tanh(torch.mm(self.W,state_car_centre.T) + self.b))
                          
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_f.T) + self.b))) , 1)
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_theta.T) + self.b))) , 1)        
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_left_cor.T)  + self.b))) , 1)     
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_right_cor.T)  + self.b))) , 1) 
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,theta_lane_cor.T)  + self.b))) , 1) 
        
        a = torch.softmax(e,1).squeeze(0)
        
        y = torch.cat([ a[0]*state_car_centre , a[1]*state_f , a[2]*state_theta , a[3]*state_left_cor , a[4]*state_right_cor , a[5]*theta_lane_cor ] , 1)
               

        x = torch.cat((x , y.float()) , 1)
            
        z = F.relu(self.fc4(x))
        z = F.softmax(self.fc5(z))
        
        
        return z


    def select_action(self,s):  # s is an numpy image
            
                       
            a = self.forward(s).squeeze(0)
            
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
                
                
            return action , prob   
 
            
            
            


# Policy Gradients using Single Network

class PolicyNetworkSN(torch.nn.Module): # network is defined per image not per batch so if we give batch of images then we will get batch of outputs 
    def __init__(self , action_space_f ,action_space_theta , image_shape , action_space ):
        super(PolicyNetworkSN, self).__init__()
        
        self.image_shape = image_shape
                
        self.action_space = action_space

        
        self.conv1 = torch.nn.Conv2d(6, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)

        self.fc4 = torch.nn.Linear(500+60,100)        
        self.fc5 = torch.nn.Linear(100, self.action_space)
        
        
        
        self.d = 10
        self.emd_size = 10
        self.W = torch.randn((self.d,self.emd_size) , requires_grad = True)
        self.b = torch.zeros((self.d,1), requires_grad = True)        
        self.V = torch.randn((self.d,1), requires_grad = True)  
         
        self.embedding1 = torch.nn.Embedding(image_shape, self.emd_size)
        self.embedding2 = torch.nn.Embedding(action_space_f, self.emd_size)
        self.embedding3 = torch.nn.Embedding(action_space_theta, self.emd_size)
        
        
        

    def forward(self, s):
        
        # do normalization for input images before passing 

        segmentation_img = s[0][0]/255
        depth_img = s[0][1]/255
        
        image = np.concatenate([segmentation_img,depth_img],2)
        # image = segmentation_img
        
        x = torch.from_numpy(image.reshape(6,self.image_shape,self.image_shape)).float().unsqueeze(0)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        

        
        x = x.view(-1, 64 *10* 10) # flattening and also -1 means batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
       
        
        # attetnion on embedding 
        state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
        state_f = self.embedding2(torch.tensor([int(s[2])]))
        state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
        state_left_cor = self.embedding1(torch.tensor([int(s[4])]))
        state_right_cor = self.embedding1(torch.tensor([int(s[5])]))
        theta_lane_cor = self.embedding3(torch.tensor([int(0.1*s[6]-1)]))
        

        e = torch.mm(self.V.T ,torch.tanh(torch.mm(self.W,state_car_centre.T) + self.b))
                          
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_f.T) + self.b))) , 1)
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_theta.T) + self.b))) , 1)        
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_left_cor.T)  + self.b))) , 1)     
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_right_cor.T)  + self.b))) , 1) 
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,theta_lane_cor.T)  + self.b))) , 1) 
        
        a = torch.softmax(e,1).squeeze(0)
        
        y = torch.cat([ a[0]*state_car_centre , a[1]*state_f , a[2]*state_theta , a[3]*state_left_cor , a[4]*state_right_cor , a[5]*theta_lane_cor ] , 1)
               

        x = torch.cat((x , y.float()) , 1)
            
        z = F.relu(self.fc4(x))
        z = F.softmax(self.fc5(z))
        
        
        return z


    def select_action(self,s):  # s is an numpy image
            
                       
            a = self.forward(s).squeeze(0)
            
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
                
                
            return action , prob   
 
            
            
            
            




# A2C
class AdvantageActorCritic(torch.nn.Module):
    def __init__(self,  action_space_f ,action_space_theta , image_shape , flag):
        super(AdvantageActorCritic, self).__init__()
        
        self.image_shape = image_shape
        
        if flag == "speed" :
            self.action_space = action_space_f
        elif flag == "angle":
            self.action_space = action_space_theta
        
        # self.action_space = action_space
    
        
        # old architecture
        self.conv1 = torch.nn.Conv2d(6, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        
        
        # new architecture
        # self.conv1 = torch.nn.Conv2d(6, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        # self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        # self.conv2 = torch.nn.Conv2d(8, 16, 5)
        # self.conv3 = torch.nn.Conv2d(16, 32, 5)
        # self.conv4 = torch.nn.Conv2d(32, 64, 5)
        # self.conv5 = torch.nn.Conv2d(64, 128, 5)
        # self.conv6 = torch.nn.Conv2d(128, 256, 5)
        # self.conv7 = torch.nn.Conv2d(256, 256, 5)
        
        
        
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)

   
        
        self.critic_linear1 = torch.nn.Linear(500+60, 100)
        self.critic_linear2 = torch.nn.Linear(100, 1)

        self.actor_linear1 = torch.nn.Linear(500+60, 100)
        self.actor_linear2 = torch.nn.Linear(100, self.action_space)
        
        self.d = 10
        self.emd_size = 10
        self.W = torch.randn((self.d,self.emd_size) , requires_grad = True)
        self.b = torch.zeros((self.d,1), requires_grad = True)        
        self.V = torch.randn((self.d,1), requires_grad = True)  
         
        self.embedding1 = torch.nn.Embedding(image_shape, self.emd_size)
        self.embedding2 = torch.nn.Embedding(action_space_f, self.emd_size)
        self.embedding3 = torch.nn.Embedding(action_space_theta, self.emd_size)
        
    
    
    def forward(self, s):
        
        
        # do normalization for input images before passing 

        segmentation_img = s[0][0]/255
        depth_img = s[0][1]/255
        
        image = np.concatenate([segmentation_img,depth_img],2)
        # image = segmentation_img
        
        x = torch.from_numpy(image.reshape(6,self.image_shape,self.image_shape)).float().unsqueeze(0)
            
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        
        # x = F.relu(self.conv1(x))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = F.relu(self.conv3(x))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = F.relu(self.conv5(x))
        # x = self.pool(F.relu(self.conv6(x)))
        # x = self.pool(F.relu(self.conv7(x)))
        
        
        
        x = x.view(-1, 64 *10* 10) # flattening and also -1 means batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
 
    
        # attention 
        # e = torch.mm(self.V.T ,torch.tanh(self.W*s[1][0]/self.image_shape + self.b))
                          
        # e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(self.W*s[2]/5 + self.b))) , 1)
        
        # e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(self.W*s[3]/180  + self.b))) , 1)        
        
        # e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(self.W*s[4]/self.image_shape  + self.b))) , 1)     
        
        # e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(self.W*s[5]/self.image_shape  + self.b))) , 1)  
        
        # a = 500*torch.softmax(e,1).squeeze(0)
        
        # y = torch.tensor([[ a[0]*s[1][0]/self.image_shape , a[1]*s[2]/5 , a[2]*s[3]/180 , a[3]*s[4]/self.image_shape , a[4]*s[5]/self.image_shape]] )
        
        
        # Embedding
        # state_f = self.embedding2(torch.tensor([int(s[2])]))
        # state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
        # state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
        
        
        # no embedding
        # state_f = torch.tensor([[int(s[2])]])
        # state_theta = torch.tensor([[int(0.1*s[3]-1)]])
        # state_car_centre = torch.tensor([[int(s[1][0])]])
               
        # y = torch.cat([state_f,state_theta,state_car_centre],1)
        
        
        
        # attetnion on embedding 
        state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
        state_f = self.embedding2(torch.tensor([int(s[2])]))
        state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
        state_left_cor = self.embedding1(torch.tensor([int(s[4])]))
        state_right_cor = self.embedding1(torch.tensor([int(s[5])]))
        theta_lane_cor = self.embedding3(torch.tensor([int(0.1*s[6]-1)]))

        e = torch.mm(self.V.T ,torch.tanh(torch.mm(self.W,state_car_centre.T) + self.b))
                          
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_f.T) + self.b))) , 1)
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_theta.T) + self.b))) , 1)        
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_left_cor.T)  + self.b))) , 1)     
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_right_cor.T)  + self.b))) , 1)  
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,theta_lane_cor.T)  + self.b))) , 1) 

        a = torch.softmax(e,1).squeeze(0)
        
        y = torch.cat([ a[0]*state_car_centre , a[1]*state_f , a[2]*state_theta , a[3]*state_left_cor , a[4]*state_right_cor ,a[5]*theta_lane_cor] , 1)


        
        x = torch.cat((x , y.float()) , 1)
               
        
        
        # value (Critic) network
        value = F.relu(self.critic_linear1(x))
        value = self.critic_linear2(value)
        
        # policy (Actor)network
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



        
    
    

# A2C with Single Network 
class AdvantageActorCriticSN(torch.nn.Module):
    def __init__(self,  action_space_f ,action_space_theta , image_shape , action_space):
        super(AdvantageActorCriticSN, self).__init__()
        
        self.image_shape = image_shape
    
        self.action_space = action_space
    
        
        # old architecture
        self.conv1 = torch.nn.Conv2d(6, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        
          
        
        
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)

   
        
        self.critic_linear1 = torch.nn.Linear(500+60, 100)
        self.critic_linear2 = torch.nn.Linear(100, 1)

        self.actor_linear1 = torch.nn.Linear(500+60, 100)
        self.actor_linear2 = torch.nn.Linear(100, self.action_space)
        
        
        self.d = 10
        self.emd_size = 10
        self.W = torch.randn((self.d,self.emd_size) , requires_grad = True)
        self.b = torch.zeros((self.d,1), requires_grad = True)        
        self.V = torch.randn((self.d,1), requires_grad = True)  
         
        self.embedding1 = torch.nn.Embedding(image_shape, self.emd_size)
        self.embedding2 = torch.nn.Embedding(action_space_f, self.emd_size)
        self.embedding3 = torch.nn.Embedding(action_space_theta, self.emd_size)
        
    
    
    def forward(self, s):
        
        
        # do normalization for input images before passing 

        segmentation_img = s[0][0]/255
        depth_img = s[0][1]/255
        
        image = np.concatenate([segmentation_img,depth_img],2)
        # image = segmentation_img
        
        x = torch.from_numpy(image.reshape(6,self.image_shape,self.image_shape)).float().unsqueeze(0)
            
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
     
        
        x = x.view(-1, 64 *10* 10) # flattening and also -1 means batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
 
        
        # attetnion on embedding 
        state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
        state_f = self.embedding2(torch.tensor([int(s[2])]))
        state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
        state_left_cor = self.embedding1(torch.tensor([int(s[4])]))
        state_right_cor = self.embedding1(torch.tensor([int(s[5])]))
        theta_lane_cor = self.embedding3(torch.tensor([int(0.1*s[6]-1)]))

        e = torch.mm(self.V.T ,torch.tanh(torch.mm(self.W,state_car_centre.T) + self.b))
                          
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_f.T) + self.b))) , 1)
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_theta.T) + self.b))) , 1)        
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_left_cor.T)  + self.b))) , 1)     
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_right_cor.T)  + self.b))) , 1)  
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,theta_lane_cor.T)  + self.b))) , 1) 

        a = torch.softmax(e,1).squeeze(0)
        
        y = torch.cat([ a[0]*state_car_centre , a[1]*state_f , a[2]*state_theta , a[3]*state_left_cor , a[4]*state_right_cor ,a[5]*theta_lane_cor] , 1)


        
        x = torch.cat((x , y.float()) , 1)
               
        
        
        # value (Critic) network
        value = F.relu(self.critic_linear1(x))
        value = self.critic_linear2(value)
        
        # policy (Actor)network
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



        
    




# A2C with single Value finction V 
class AdvantageActorSV(torch.nn.Module):
    def __init__(self,  action_space_f ,action_space_theta , image_shape , flag):
        super(AdvantageActorSV, self).__init__()
        
        self.image_shape = image_shape
        
        if flag == "speed" :
            self.action_space = action_space_f
        elif flag == "angle":
            self.action_space = action_space_theta
        
        # self.action_space = action_space
    
        
        # old architecture
        self.conv1 = torch.nn.Conv2d(6, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        
        
     
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)

   
        self.actor_linear1 = torch.nn.Linear(500+60, 100)
        self.actor_linear2 = torch.nn.Linear(100, self.action_space)
        
        self.d = 10
        self.emd_size = 10
        self.W = torch.randn((self.d,self.emd_size) , requires_grad = True)
        self.b = torch.zeros((self.d,1), requires_grad = True)        
        self.V = torch.randn((self.d,1), requires_grad = True)  
         
        self.embedding1 = torch.nn.Embedding(image_shape, self.emd_size)
        self.embedding2 = torch.nn.Embedding(action_space_f, self.emd_size)
        self.embedding3 = torch.nn.Embedding(action_space_theta, self.emd_size)
        
    
    
    def forward(self, s):
        
        
        # do normalization for input images before passing 

        segmentation_img = s[0][0]/255
        depth_img = s[0][1]/255
        
        image = np.concatenate([segmentation_img,depth_img],2)
        # image = segmentation_img
        
        x = torch.from_numpy(image.reshape(6,self.image_shape,self.image_shape)).float().unsqueeze(0)
            
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
    
        
        
        x = x.view(-1, 64 *10* 10) # flattening and also -1 means batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        

        
        # attetnion on embedding 
        state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
        state_f = self.embedding2(torch.tensor([int(s[2])]))
        state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
        state_left_cor = self.embedding1(torch.tensor([int(s[4])]))
        state_right_cor = self.embedding1(torch.tensor([int(s[5])]))
        theta_lane_cor = self.embedding3(torch.tensor([int(0.1*s[6]-1)]))

        e = torch.mm(self.V.T ,torch.tanh(torch.mm(self.W,state_car_centre.T) + self.b))
                          
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_f.T) + self.b))) , 1)
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_theta.T) + self.b))) , 1)        
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_left_cor.T)  + self.b))) , 1)     
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_right_cor.T)  + self.b))) , 1)  
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,theta_lane_cor.T)  + self.b))) , 1) 

        a = torch.softmax(e,1).squeeze(0)
        
        y = torch.cat([ a[0]*state_car_centre , a[1]*state_f , a[2]*state_theta , a[3]*state_left_cor , a[4]*state_right_cor ,a[5]*theta_lane_cor] , 1)


        
        x = torch.cat((x , y.float()) , 1)
               

        # policy (Actor)network
        policy_dist = F.relu(self.actor_linear1(x))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return  policy_dist


    def select_action(self, s):
        policy_dist = self.forward(s)
        dist = policy_dist.detach().numpy() 

        action = np.random.choice(self.action_space, p=np.squeeze(dist))
        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))

        return action , log_prob , entropy


class AdvantageCriticSV(torch.nn.Module):
    def __init__(self,  action_space_f ,action_space_theta , image_shape):
        super(AdvantageCriticSV, self).__init__()
        
        self.image_shape = image_shape
        
        # self.action_space = action_space
    
        
        # old architecture
        self.conv1 = torch.nn.Conv2d(6, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        
        
     
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)


        self.critic_linear1 = torch.nn.Linear(500+60, 100)
        self.critic_linear2 = torch.nn.Linear(100, 1)


        self.d = 10
        self.emd_size = 10
        self.W = torch.randn((self.d,self.emd_size) , requires_grad = True)
        self.b = torch.zeros((self.d,1), requires_grad = True)        
        self.V = torch.randn((self.d,1), requires_grad = True)  
         
        self.embedding1 = torch.nn.Embedding(image_shape, self.emd_size)
        self.embedding2 = torch.nn.Embedding(action_space_f, self.emd_size)
        self.embedding3 = torch.nn.Embedding(action_space_theta, self.emd_size)
        
    
    
    def forward(self, s):
        
        
        # do normalization for input images before passing 

        segmentation_img = s[0][0]/255
        depth_img = s[0][1]/255
        
        image = np.concatenate([segmentation_img,depth_img],2)
        # image = segmentation_img
        
        x = torch.from_numpy(image.reshape(6,self.image_shape,self.image_shape)).float().unsqueeze(0)
            
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
    
        
        
        x = x.view(-1, 64 *10* 10) # flattening and also -1 means batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        

        # attetnion on embedding 
        state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
        state_f = self.embedding2(torch.tensor([int(s[2])]))
        state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
        state_left_cor = self.embedding1(torch.tensor([int(s[4])]))
        state_right_cor = self.embedding1(torch.tensor([int(s[5])]))
        theta_lane_cor = self.embedding3(torch.tensor([int(0.1*s[6]-1)]))

        e = torch.mm(self.V.T ,torch.tanh(torch.mm(self.W,state_car_centre.T) + self.b))
                          
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_f.T) + self.b))) , 1)
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_theta.T) + self.b))) , 1)        
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_left_cor.T)  + self.b))) , 1)     
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_right_cor.T)  + self.b))) , 1)  
        
        e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,theta_lane_cor.T)  + self.b))) , 1) 

        a = torch.softmax(e,1).squeeze(0)
        
        y = torch.cat([ a[0]*state_car_centre , a[1]*state_f , a[2]*state_theta , a[3]*state_left_cor , a[4]*state_right_cor ,a[5]*theta_lane_cor] , 1)


        
        x = torch.cat((x , y.float()) , 1)
               
        

        # value (Critic) network
        value = F.relu(self.critic_linear1(x))
        value = self.critic_linear2(value)
    
        return  value[0,0]


            
            
# duelling DQN

class QNetwork(torch.nn.Module):
    def __init__(self, action_space_f ,action_space_theta , image_shape , flag ):
        super(QNetwork, self).__init__()
        
        # fine tuning vgg16 with UCMerced to evaluate the effectiveness and robustness of CNN features . To see whether vgg16 can capture aerial features (because plant images are taken from drone in real life scenarios, but in plant village data set those are not drone images)
        
        self.image_shape = image_shape
        
        if flag == "speed" :
            self.action_space = action_space_f
        elif flag == "angle":
            self.action_space = action_space_theta
        
        # self.action_space = action_space
    
        
        # old architecture
        self.conv1 = torch.nn.Conv2d(6, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        
                
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)

   
        
        self.fc_value = torch.nn.Linear(500+60, 100)
        self.value = torch.nn.Linear(100, 1)
        
        self.fc_adv = torch.nn.Linear(500+60, 100)
        self.adv = torch.nn.Linear(100,self.action_space)
        
        self.d = 10
        self.emd_size = 10
        self.W = torch.randn((self.d,self.emd_size) , requires_grad = True)
        self.b = torch.zeros((self.d,1), requires_grad = True)        
        self.V = torch.randn((self.d,1), requires_grad = True)  
         
        self.embedding1 = torch.nn.Embedding(image_shape, self.emd_size)
        self.embedding2 = torch.nn.Embedding(action_space_f, self.emd_size)
        self.embedding3 = torch.nn.Embedding(action_space_theta, self.emd_size)
        
        
                
    def forward(self, state):   
        
        
         for i in range(len(state)):
             
            s = state[i]
            
            # do normalization for input images before passing 
    
            segmentation_img = s[0][0]/255
            depth_img = s[0][1]/255
            
            image = np.concatenate([segmentation_img,depth_img],2)
            # image = segmentation_img
            
            x = torch.from_numpy(image.reshape(6,self.image_shape,self.image_shape)).float().unsqueeze(0)
                
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            
    
            
            x = x.view(-1, 64 *10* 10) # flattening and also -1 means batch size
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
           
            
            # attetnion on embedding 
            state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
            state_f = self.embedding2(torch.tensor([int(s[2])]))
            state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
            state_left_cor = self.embedding1(torch.tensor([int(s[4])]))
            state_right_cor = self.embedding1(torch.tensor([int(s[5])]))
            theta_lane_cor = self.embedding3(torch.tensor([int(0.1*s[6]-1)]))
            
    
            e = torch.mm(self.V.T ,torch.tanh(torch.mm(self.W,state_car_centre.T) + self.b))
                              
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_f.T) + self.b))) , 1)
            
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_theta.T) + self.b))) , 1)        
            
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_left_cor.T)  + self.b))) , 1)     
            
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_right_cor.T)  + self.b))) , 1) 
            
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,theta_lane_cor.T)  + self.b))) , 1) 
            
            a = torch.softmax(e,1).squeeze(0)
            
            y = torch.cat([ a[0]*state_car_centre , a[1]*state_f , a[2]*state_theta , a[3]*state_left_cor , a[4]*state_right_cor , a[5]*theta_lane_cor ] , 1)
               
        
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




            
# duelling DQN single Q 

class QNetworkSN(torch.nn.Module):
    def __init__(self, action_space_f ,action_space_theta , image_shape , action_space ):
        super(QNetworkSN, self).__init__()
        
        # fine tuning vgg16 with UCMerced to evaluate the effectiveness and robustness of CNN features . To see whether vgg16 can capture aerial features (because plant images are taken from drone in real life scenarios, but in plant village data set those are not drone images)
        
        self.image_shape = image_shape
        
        self.action_space = action_space
    
        
        # old architecture
        self.conv1 = torch.nn.Conv2d(6, 8, 5) # 3 is due to colour image(depth) , 6 is no of filters , and each filter size is 5*5 
        self.pool = torch.nn.MaxPool2d(2, 2) # 2*2  size  with stride 2 
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.conv4 = torch.nn.Conv2d(32, 64, 5)
        
                
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 2000)
        self.fc2 = torch.nn.Linear(2000, 500)

   
        
        self.fc_value = torch.nn.Linear(500+60, 100)
        self.value = torch.nn.Linear(100, 1)
        
        self.fc_adv = torch.nn.Linear(500+60, 100)
        self.adv = torch.nn.Linear(100,self.action_space)
        
        self.d = 10
        self.emd_size = 10
        self.W = torch.randn((self.d,self.emd_size) , requires_grad = True)
        self.b = torch.zeros((self.d,1), requires_grad = True)        
        self.V = torch.randn((self.d,1), requires_grad = True)  
         
        self.embedding1 = torch.nn.Embedding(image_shape, self.emd_size)
        self.embedding2 = torch.nn.Embedding(action_space_f, self.emd_size)
        self.embedding3 = torch.nn.Embedding(action_space_theta, self.emd_size)
        
        
                
    def forward(self, state):   
        
        
         for i in range(len(state)):
             
            s = state[i]
            
            # do normalization for input images before passing 
    
            segmentation_img = s[0][0]/255
            depth_img = s[0][1]/255
            
            image = np.concatenate([segmentation_img,depth_img],2)
            # image = segmentation_img
            
            x = torch.from_numpy(image.reshape(6,self.image_shape,self.image_shape)).float().unsqueeze(0)
                
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            
    
            
            x = x.view(-1, 64 *10* 10) # flattening and also -1 means batch size
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
           
            
            # attetnion on embedding 
            state_car_centre = self.embedding1(torch.tensor([int(s[1][0])]))
            state_f = self.embedding2(torch.tensor([int(s[2])]))
            state_theta = self.embedding3(torch.tensor([int(0.1*s[3]-1)]))
            state_left_cor = self.embedding1(torch.tensor([int(s[4])]))
            state_right_cor = self.embedding1(torch.tensor([int(s[5])]))
            theta_lane_cor = self.embedding3(torch.tensor([int(0.1*s[6]-1)]))
            
    
            e = torch.mm(self.V.T ,torch.tanh(torch.mm(self.W,state_car_centre.T) + self.b))
                              
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_f.T) + self.b))) , 1)
            
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_theta.T) + self.b))) , 1)        
            
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_left_cor.T)  + self.b))) , 1)     
            
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,state_right_cor.T)  + self.b))) , 1) 
            
            e = torch.cat(( e, torch.mm( self.V.T ,torch.tanh(torch.mm(self.W,theta_lane_cor.T)  + self.b))) , 1) 
            
            a = torch.softmax(e,1).squeeze(0)
            
            y = torch.cat([ a[0]*state_car_centre , a[1]*state_f , a[2]*state_theta , a[3]*state_left_cor , a[4]*state_right_cor , a[5]*theta_lane_cor ] , 1)
               
        
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







