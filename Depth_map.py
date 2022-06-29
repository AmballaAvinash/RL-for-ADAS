#https://github.com/nianticlabs/monodepth2
# run /home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Depth_estimation/Monocular/test_simple.py --image_path /home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Depth_estimation/Monocular/test_image.jpg --model_name mono+stereo_640x192

from __future__ import absolute_import, division, print_function


import sys
sys.path.insert(1, '/home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Depth_estimation/Monocular')



import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2

import torch
from torchvision import transforms, datasets


from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder
from layers import disp_to_depth

STEREO_SCALE_FACTOR = 5.4


class Depth():    
    def get_image(self , image1 , image2):
        """Function to predict for a single image or folder of images
        """
        
        if torch.cuda.is_available() :
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
        model_path = os.path.join("/home/avinash/Desktop/projects/Self_driving_car/Udacity_self_driving_car_simulator/self_driving_car_RL/Depth_estimation/Monocular", "mono+stereo_640x192")
    
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")
        
        
        # LOADING PRETRAINED MODEL
        encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
    
        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()
    
        
        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)
    
        depth_decoder.to(device)
        depth_decoder.eval()

    
        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            
                # Load image and preprocess
                input_image = pil.fromarray(np.uint8(image2)).convert('RGB')
                original_width, original_height = input_image.size
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    
                # PREDICTION
                input_image = input_image.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)
    
                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)
    
                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    
                np_img = cv2.resize(colormapped_im, (image1.shape[0],image1.shape[1]), interpolation = cv2.INTER_AREA)
                return np_img
            
