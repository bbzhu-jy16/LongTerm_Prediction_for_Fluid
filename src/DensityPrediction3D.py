# -*- coding: utf-8 -*-
"""
Created on Wed Sep 8 11:44:07 2021

@author: zhujy
"""

import numpy as np
import tensorflow as tf
import os
import time
import glob
import math
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
from dataload.dataloader import load_density_input,load_density_output,load_velocity_frame
from util.loss_function import density_loss
from util.arguments import Density3D_parse
from util.metrics import psnr_den_3D, cos_den_3D
from model.prediction_model import DensityPrediction3D

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, BatchNormalization, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, LSTM, Embedding, add, Reshape, Multiply, concatenate
from tensorflow.keras.layers import LeakyReLU, ELU, PReLU, ThresholdedReLU
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def train(args,model):
    res=args.resolution
    frame0=args.frame0
    frame_num=args.frame_num

    density_data = load_density_input(args.density_datapath, frame0, frame_num)
    vel_data = load_velocity_frame(args.velocity_datapath, frame0, frame_num)
    density_GT = load_density_output(args.density_datapath, frame0, frame_num)

    vel_data = np.asarray(vel_data) #(frame_num,64,64,64,3)
    density_GT=np.asarray(density_GT) #(frame_num,64,64,64)
    density_data = np.asarray(density_data) #(64,64,64)
    density_data_input = np.zeros((density_data.shape[0],frame_num,res,res,res)) 
    for i in range(frame_num):
        density_data_input[:,i,:,:,0]=density_data
    
    model.compile(optimizer='adam',loss=density_loss)

    checkpointer = ModelCheckpoint(os.path.join(args.model_path,'3D_Long_density_prediction64_{epoch:03d}.h5'),verbose=1,save_weights_only=False,save_best_only=False)
    Density_Prediction_training = model.fit([density_data_input, vel_data], density_GT, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, validation_split=0.111,callbacks=[checkpointer])

    plt.plot(Density_Prediction_training.history['loss'])
    plt.plot(Density_Prediction_training.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','validation'],loc='upper right')
    plt.show()
    
    return model

def test(args, model):
    frame0=args.frame0
    filenames=glob.glob(args.density_test_path+"/*.npz")
    res=args.resolution
    err_all=[]
    psnr_all=[]
    cos_all=[]
    time_all=[]
    frame_num=30
    for filename in tqdm(filenames):
        density_test = np.load(filename, allow_pickle=True)
        density_test = density_test['data']
        velocity_test = np.load(args.velocity_test_path+"/"+filename[-10:],allow_pickle=True)
        velocity_test = velocity_test['data'][frame0:frame0+frame_num,:,:,:,0:3]
        velocity_test = velocity_test.reshape(1,frame_num,res,res,res,3)
        density_test0 = density_test[frame0-1,:,:,:]*2-1
        density_test0 = density_test0.reshape(res,res,res)
        density_test_truth = density_test[frame0:frame0+frame_num,:,:,:]
        density_test_input = np.zeros((1,frame_num,res,res,res)) 
        for i in range(frame_num):
            density_test_input[0,i,:,:,:]=density_test0
        test_input = [density_test_input, velocity_test]
        begin_time=time.time()
        outputs_test = model.predict(test_input, batch_size=1)
        end_time=time.time()
        time_all.append(end_time-begin_time)
        outputs_test = np.squeeze(outputs_test)
        outputs_test = outputs_test+1
        outputs_test = outputs_test/2
        np.savez("args.test_save_path"+filename[-10:],data=outputs_test)
        err=0
        for i in range(frame_num):
            for j in range(res):
                for k in range(res):
                    for l in range(res):
                        #err += np.abs(outputs_test[i,j,k]-density_test_truth[i,j,k])
                        err += (outputs_test[i,j,k,l]-density_test_truth[i,j,k,l])*(outputs_test[i,j,k,l]-density_test_truth[i,j,k,l])
        err /= frame_num*res*res*res
        err_all.append(err)
        psnr_frame = psnr_den_3D(density_test_truth, outputs_test)
        psnr_all.append(psnr_frame)
        cos_all.append(cos_den_3D(density_test_truth, outputs_test))
    
    time_all=np.asarray(time_all)
    print("Mean Inference Time:", np.mean(time_all)/frame_num)
    err_all=np.asarray(err_all)
    print("Test Mean Absolute Error:", np.mean(err_all))
    psnr_all=np.asarray(psnr_all)
    print("Mean PSNR:", np.mean(psnr_all))
    cos_all=np.asarray(cos_all)
    print("Mean Cosine Similarity:", np.mean(cos_all))


def main():
    print("Main Function Running")
    args=Density3D_parse()
    model=DensityPrediction3D(args.resolution,args.frame_num)
    mode=args.mode
    if mode=="train":
        train(args,model)
    elif mode=="test":
        model=load_model(args.model_path+"/"+args.model_name)
        test(args,model)