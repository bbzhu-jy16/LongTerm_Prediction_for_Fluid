# -*- coding: utf-8 -*-
"""
Created on Wed Sep 8 11:44:07 2021

@author: zhujy
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

#Density Prediction Model
def load_density_input(density_path,frame0):
    print("Loading Frame 0 Density")
    density_file=glob.glob(density_path+"/*.npz")
    density_data=[]
    for filename in tqdm(density_file):
        data=np.load(filename,allow_pickle=True)
        data=np.squeeze(data['data'])
        density_data.append(data[frame0])
    return density_data

def load_density_output(density_path,frame0,frame_num):
    print("Loading Density Ground Truth")
    density_file=glob.glob(density_path+"/*.npz")
    density_data=[]
    for filename in tqdm(density_file):
        data=np.load(filename,allow_pickle=True)
        data=np.squeeze(data['data'])
        density_data.append(data['data'][frame0+1:frame0+1+frame_num])
    return density_data

def load_velocity_frame(velocity_path, frame0, frame_num,dim):
    print("Loading Velocity")
    vel_file=glob.glob(velocity_path+"/*.npz")
    vel_data=[]
    for filename in tqdm(vel_file):
        data=np.load(filename,allow_pickle=True)
        data=np.squeeze(data['data'])
        vel_data.append(data[frame0:frame0+frame_num,:,:,0:dim])
    return vel_data

#Velocity Prediction Model
def load_velocity(velocity_path, frame0, frame_num, dim):
    print("Loading Velocity")
    vel_file=glob.glob(velocity_path+"/*.npz")
    vel_data=[]
    for filename in tqdm(vel_file):
        data=np.load(filename,allow_pickle=True)
        data=np.squeeze(data['data'])
        vel_data.append(data[frame0:frame0+frame_num,:,:,0:dim])
    return vel_data