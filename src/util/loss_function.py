# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 12:10:07 2021

@author: zhujy
"""

import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Lambda, Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, BatchNormalization, MaxPooling2D, Flatten
from keras.layers import Dense, Input, Activation, Dropout, LSTM, Embedding, add, recurrent, Reshape, Multiply, concatenate
from keras.layers import LeakyReLU, ELU, PReLU, ThresholdedReLU
from keras.callbacks import LearningRateScheduler
from keras.utils import get_custom_objects
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

res=64

def density_loss(y_true,y_pred,alpha=0.5):
    norm_loss = K.mean(K.abs(y_true-y_pred))
    weights_loss = K.mean(K.abs(Multiply()([y_true-y_pred,K.abs(y_true)])))

    loss = alpha*norm_loss+(1-alpha)*weights_loss

    return loss

def bnd_loss_2d(x):
    M=N=res
    boundary_x=K.square(x[:,:,0,:,0])+K.square(x[:,:,N-1,:,0])
    boundary_y=K.square(x[:,:,:,0,1])+K.square(x[:,:,:,M-1,1])
    boundary_loss = K.mean(boundary_x+boundary_y)
    
    return boundary_loss
    
def div_loss_2d(x):
    M=N=res
    div_loss=K.mean(K.square(1/2*(x[:,:,2:M,1:N-1,1]-x[:,:,0:M-2,1:N-1,1])+1/2*(x[:,:,1:M-1,2:N,0]-x[:,:,1:M-1,0:N-2,0])))
    return div_loss

def velocity_loss_2D(y_true, y_pred, alpha=0.5, beta=0.2, gamma=0.2):
    weights_loss = K.mean(K.abs(Multiply()([y_true-y_pred,K.abs(y_true)])))
    norm_loss = K.mean(K.abs(y_true-y_pred))
    
    boundary_loss = Lambda(bnd_loss_2d)(y_pred)
    divergency_loss = Lambda(div_loss_2d)(y_pred)
    
    loss = alpha*norm_loss+beta*weights_loss+gamma*divergency_loss+(1-alpha-beta-gamma)*boundary_loss
    
    return loss 

def bnd_loss_3d(x):
    M=N=K=res
    boundary_x=K.square(x[:,:,0,:,:,0])+K.square(x[:,:,N-1,:,:,0])
    boundary_y=K.square(x[:,:,:,0,:,1])+K.square(x[:,:,:,M-1,:,1])
    boundary_z=K.square(x[:,:,:,:,0,2])+K.square(x[:,:,:,:,K-1,2])
    boundary_loss = K.mean(boundary_x+boundary_y+boundary_z)
    
    return boundary_loss

def div_loss_3d(x):#(batch_size,frame_input,sizex,sizey,sizez,dim)
    M=N=K=res
    div_loss=K.mean(K.square(1/2*(x[:,:,2:M,1:N-1,1:K-1,1]-x[:,:,0:M-2,1:N-1,1:K-1,1])+1/2*(x[:,:,1:M-1,2:N,1:K-1,0]-x[:,:,1:M-1,0:N-2,1:K-1,0])+1/2*(x[:,:,1:M-1,1:N-1,2:K,2]-x[:,:,1:M-1,1:N-1,0:K-2,2])))
    return div_loss

def velocity_loss_3D(y_true, y_pred, alpha=0.5, beta=0.2, gamma=0.2):
    weights_loss = K.mean(K.abs(Multiply()([y_true-y_pred,K.abs(y_true)])))
    norm_loss = K.mean(K.abs(y_true-y_pred))
    
    boundary_loss = Lambda(bnd_loss_3d)(y_pred)
    divergency_loss = Lambda(div_loss_3d)(y_pred)
    
    loss = alpha*norm_loss+beta*weights_loss+gamma*divergency_loss+(1-alpha-beta-gamma)*boundary_loss
    
    return loss 