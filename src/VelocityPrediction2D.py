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

from dataload.dataloader import load_velocity
from util.loss_function import velocity_loss_2D
from util.arguments import Velocity2D_parse
from util.metrics import psnr_vel_2D, cos_vel_2D
from model.prediction_model import VelocityPrediction2D

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Lambda, Conv1D, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, Flatten, concatenate, ELU, PReLU, ThresholdedReLU
from keras.layers import Dense, Input, Activation, Dropout, LSTM, Embedding, add, LeakyReLU, recurrent, Reshape, Conv3D, Conv3DTranspose, Multiply
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.utils import get_custom_objects
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def train(args,model):
    vel_data = load_velocity(args.velocity_train_path,args.frame0,args.frame_num,2)
    vel_data = np.asarray(vel_data)
    
    frame_input=args.frame_input
    frame_output=args.frame_num-args.frame_input

    model.compile(optimizer='adam',loss=velocity_loss_2D)
    checkpointer = ModelCheckpoint(os.path.join(args.model_path,'Long_Velocity_Prediction_{epoch:03d}.h5'),verbose=1,save_weights_only=False,save_best_only=False)
    training = model.fit(vel_data[:,0:frame_input],vel_data[:,frame_input:frame_input+frame_output],epochs=50,batch_size=64,shuffle=True,validation_split=0.111,callbacks=[checkpointer])

    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','validation'],loc='upper right')
    plt.show()

    return model

def test(args, model):
    frame_input = args.frame_input
    frame_output = args.frame_num-args.frame_input
    res=args.resolution
    vel_file=glob.glob(args.velocity_test_path+"/*.npz")
    err_all=[]
    psnr_all=[]
    cos_all=[]
    time_all=[]
    for filename in tqdm(vel_file):
        data=np.load(filename,allow_pickle=True)
        #test_data=scale*data['data'][:,:,:,0,0:2]
        test_data=data['data'][:,:,:,0:2]
        test_input = test_data[0:frame_input].reshape(1,frame_input,res,res,2)
        test_gt = test_data[frame_input:frame_input+frame_output]
        begin_time=time.time()
        test_output = model.predict(test_input, batch_size=1)
        end_time=time.time()
        time_all.append(end_time-begin_time)
        test_output = np.squeeze(test_output)
        np.savez(args.test_save_path+filename[-10:],data=test_output)
        err = 0 
        for i in range(frame_output):
            for j in range(res):
                for k in range(res):
                    for l in range(2):
                        #err += np.abs(test_output[i,j,k,l]-test_gt[i,j,k,l])
                        err += (test_output[i,j,k,l]-test_gt[i,j,k,l])*(test_output[i,j,k,l]-test_gt[i,j,k,l])
        err = err/(frame_output*res*res)
        err_all.append(err)
        psnr_frame = psnr_vel_2D(test_gt, test_output)
        psnr_all.append(psnr_frame)
        cos_all.append(cos_vel_2D(test_gt, test_output))
        
    time_all = np.asarray(time_all)
    print("Mean Inference Time:", np.mean(time_all)/frame_output)
    err_all = np.asarray(err_all)
    print("Mean Test Error:", np.mean(err_all))
    psnr_all=np.asarray(psnr_all)
    print("Mean PSNR:", np.mean(np.mean(psnr_all)))
    cos_all=np.asarray(cos_all)
    print("Mean Cosine Similarity:", np.mean(cos_all))

def main():
    print("Main Function Running")
    args=Velocity2D_parse()
    model=VelocityPrediction2D(args.resolution,args.frame_input,args.frame_num-args.frame_input)
    mode=args.mode
    if mode=="train":
        train(args,model)
    elif mode=="test":
        model=load_model(args.model_path+"/"+args.model_name)
        test(args,model)

if __name__=="__main__":
    main()