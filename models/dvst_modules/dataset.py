from torch.utils.data import Dataset
import random
import os
# import torch.nn.functional as F
import torch
import numpy as np
from einops import rearrange, repeat
import math
import json
import traceback
from typing import List
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from transformers import CLIPImageProcessor
import pickle
from PIL import Image
from tqdm import tqdm
from skimage import color
import cv2

def convert_H_channel(img_rbg):
    img_hed = color.rgb2hed(img_rbg)
    img_hed[:, :, 1] = 0  # 将E通道置零
    img_hed[:, :, 2] = 0  # 将D通道置零
    modified_rgb_image = color.hed2rgb(img_hed)
    modified_rgb_image = (modified_rgb_image * 255.).astype("uint8")
    return modified_rgb_image

class VideosIterableDataset(Dataset):
    def __init__(
        self,  
        data_dir_pre,
        group_size = 4, # how much images use for chosing patch
        video_length = 16,
        img_resolution = [1024, 1024],
        resolution=[512, 512],
        pose_dir_name = None,
        target_dir_name = None,
        source_from_self = False,
        blur_ref_image = False,
    ):
        self.clip_image_processor = CLIPImageProcessor.from_pretrained("./weights/dvst_pretrained")
        self.resolution = resolution
        self.img_resolution = img_resolution
        self.group_size = group_size
        self.video_length = video_length
        self.source_from_self = source_from_self
        self.blur_ref_image = blur_ref_image
        assert self.video_length % self.group_size == 0, \
            "group_size must be devided by video_length"
        self.select_list = []
        self.image_names = {}
        data_dirs = os.listdir(os.path.join(data_dir_pre, target_dir_name))
        for item in data_dirs:
            source_dir = os.path.join(data_dir_pre, target_dir_name, item)
            pose_dir = os.path.join(data_dir_pre, pose_dir_name, item)
            self.select_list.append((source_dir, pose_dir, item))
            self.image_names[item] = os.listdir(pose_dir)
            
        self.select_list = self.select_list * (1000000 // len(self.select_list))
        random.shuffle(self.select_list)
        
    def __len__(self, ):
        return len(self.select_list)
    
    def blur(self, img):
        img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (51, 51), sigmaX=0, sigmaY=0)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def __getitem__(self, index):
        source_dir, pose_dir, item_name = self.select_list[index]
        image_names = self.image_names[item_name]
        select_names = random.sample(image_names, self.group_size + (1 if not self.source_from_self else 0))
        if self.source_from_self:
            select_names.append(select_names[-1])
        random_x = random.sample(range(self.img_resolution[0] - self.resolution[0]), self.video_length + 1)
        random_y = random.sample(range(self.img_resolution[1] - self.resolution[1]), self.video_length + 1)
        random_use_idx = 0
        
        pixel_values = []
        pixel_values_pose = []
        pixel_values_pose_h = []
        for select_name in select_names[:-1]:
            ori_source_img = Image.open(os.path.join(source_dir, select_name)).convert("RGB")
            ori_pose_img = Image.open(os.path.join(pose_dir, select_name)).convert("RGB")

            ori_source_img = np.array(ori_source_img)
            ori_pose_img = np.array(ori_pose_img)
            for i in range(self.video_length // self.group_size):
                x, y = random_x[random_use_idx], random_y[random_use_idx]
                source_img = ori_source_img[x: x + self.resolution[0], y: y + self.resolution[1]]
                pose_img = ori_pose_img[x: x + self.resolution[0], y: y + self.resolution[1]]
                pose_img_h = convert_H_channel(pose_img)
                pixel_values.append(source_img)
                pixel_values_pose.append(pose_img)
                pixel_values_pose_h.append(pose_img_h)
                random_use_idx += 1 
                
        pixel_values_ref_img = os.path.join(source_dir, select_names[-1])
        pixel_values_ref_img = Image.open(pixel_values_ref_img).convert("RGB").resize((self.resolution[0], self.resolution[1]))
        pixel_values_ref_img = np.array(pixel_values_ref_img)
        
        pixel_values.append(pixel_values_ref_img)
        pixel_values, pixel_values_ref_img = pixel_values[:-1], pixel_values[-1]
        if self.blur_ref_image:
            pixel_values_ref_img = self.blur(pixel_values_ref_img)
        
        clip_image = self.clip_image_processor.preprocess((np.asarray(
                    pixel_values_ref_img) / 255.0).astype(np.float32),
                    return_tensors="pt",
                    do_rescale=False).pixel_values.squeeze(0)
        
        pixel_values = torch.tensor(np.array(pixel_values))
        pixel_values_pose = torch.tensor(np.array(pixel_values_pose))
        pixel_values_pose_h = torch.tensor(np.array(pixel_values_pose_h))
        pixel_values = rearrange(pixel_values, "f h w c -> f c h w")
        pixel_values_pose = rearrange(pixel_values_pose, "f h w c -> f c h w")
        pixel_values_pose_h = rearrange(pixel_values_pose_h, "f h w c -> f c h w")
        pixel_values_ref_img = torch.tensor(pixel_values_ref_img)
        pixel_values_ref_img = rearrange(pixel_values_ref_img, "h w c -> c h w")
        return dict(
            pixel_values = pixel_values,
            pixel_values_pose = pixel_values_pose,
            pixel_values_ref_img = pixel_values_ref_img,
            pixel_values_pose_h = pixel_values_pose_h,
            clip_image = clip_image,
        )
        
