# -*- coding: utf-8 -*-
"""
Created on Wed Sep 8 11:55:12 2021

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

def DensityPrediction2D(res,frame_num):
    #parameters
    feature_multiplier = 8
    surface_kernel_size = 4
    kernel_size = 4
    
    #MODEL
    Input_density = Input(shape=(frame_num,res,res,1))
    conv11 = Conv3D(filters=feature_multiplier, 
                   kernel_size=surface_kernel_size,
                   strides=(1, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(Input_density)
    conv11 = BatchNormalization()(conv11, training=False)
    conv11 = LeakyReLU(alpha=0.2)(conv11)
    #conv11 = Reshape((frame_num,int(res/2),int(res/2),4))(Input_density)
    
    conv22 = Conv3D(filters=feature_multiplier*2, 
                   kernel_size=kernel_size,
                   strides=(1, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv11)
    conv22 = BatchNormalization()(conv22, training=False)
    conv22 = LeakyReLU(alpha=0.2)(conv22)
    
    conv33 = Conv3D(filters=feature_multiplier*4,
                   kernel_size=kernel_size,
                   strides=(1, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv22)
    conv33 = BatchNormalization()(conv33, training=False)
    conv33 = LeakyReLU(alpha=0.2)(conv33)
    
    conv44 = Conv3D(filters=feature_multiplier*8,
                   kernel_size=kernel_size,
                   strides=(1, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv33)
    conv44 = BatchNormalization()(conv44, training=False)
    conv44 = LeakyReLU(alpha=0.2)(conv44)
    
    density_output = Reshape((frame_num,int(res*res/4)))(conv44)
    
    Input_velocity = Input(shape=(frame_num,res,res,2))
    conv1 = Conv3D(filters=feature_multiplier, 
                   kernel_size=surface_kernel_size,
                   strides=(1, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(Input_velocity)
    conv1 = BatchNormalization()(conv1, training=False)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    #conv1 = Reshape((frame_num,int(res/2),int(res/2),8))(Input_velocity)
    
    conv2 = Conv3D(filters=feature_multiplier*2, 
                   kernel_size=kernel_size,
                   strides=(1, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv1)
    conv2 = BatchNormalization()(conv2, training=False)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    
    conv3 = Conv3D(filters=feature_multiplier*4,
                   kernel_size=kernel_size,
                   strides=(1, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv2)
    conv3 = BatchNormalization()(conv3, training=False)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    
    conv4 = Conv3D(filters=feature_multiplier*8,
                   kernel_size=kernel_size,
                   strides=(1, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv3)
    conv4 = BatchNormalization()(conv4, training=False)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    
    velocity_output = Reshape((frame_num,int(res*res/4)))(conv4)
    
    hidden_layer = add([density_output, velocity_output])
    hidden_layer = LSTM(int(res*res/4), input_shape=(frame_num,int(res*res/4)), return_sequences=True,
                        activation='softsign', recurrent_activation='hard_sigmoid',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal', bias_initializer='zeros')(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer, training=False)
    hidden_layer = Reshape((frame_num,2,2,int(res*res/16)))(hidden_layer)
  
    deconv5 = Conv3DTranspose(filters=feature_multiplier*8,
                               kernel_size=kernel_size,
                               strides=(1,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(hidden_layer)
    deconv5 = BatchNormalization()(deconv5, training=False)
    deconv5 = LeakyReLU(alpha=0.2)(deconv5)
    
    deconv4 = Conv3DTranspose(filters=feature_multiplier*4,
                               kernel_size=kernel_size,
                               strides=(1,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv5)
    deconv4 = BatchNormalization()(deconv4, training=False)
    deconv4 = LeakyReLU(alpha=0.2)(deconv4)
    
    deconv3 = Conv3DTranspose(filters=feature_multiplier*2,
                               kernel_size=kernel_size,
                               strides=(1,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv4)
    deconv3 = BatchNormalization()(deconv3, training=False)
    deconv3 = LeakyReLU(alpha=0.2)(deconv3)
    
    deconv2 = Conv3DTranspose(filters=feature_multiplier*1,
                               kernel_size=kernel_size,
                               strides=(1,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv3)
    deconv2 = BatchNormalization()(deconv2, training=False)
    deconv2 = LeakyReLU(alpha=0.2)(deconv2)
    
    deconv1 = Conv3DTranspose(filters=1,
                               kernel_size=kernel_size,
                               strides=(1,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv2)
    deconv1 = BatchNormalization()(deconv1, training=False)
    deconv1 = LeakyReLU(alpha=0.2)(deconv1)
    
    outputs = Reshape((frame_num,res,res))(deconv1)
    
    Density_Prediction = Model(inputs=[Input_density, Input_velocity], outputs=outputs)
    Density_Prediction.summary()

    return Density_Prediction

def DensityPrediction3D(res,frame_num):
    #parameters
    feature_multiplier = 8
    surface_kernel_size = 4
    kernel_size = 2
    
    #MODEL
    Input_density = Input(shape=(frame_num,res,res,res))
    density_reshape = Reshape((res,res,res,frame_num))(Input_density)

    conv11 = Conv3D(filters=feature_multiplier*frame_num, 
                   kernel_size=surface_kernel_size,
                   strides=(2, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(density_reshape)
    conv11 = BatchNormalization()(conv11, training=False)
    conv11 = LeakyReLU(alpha=0.2)(conv11)
    
    conv22 = Conv3D(filters=feature_multiplier*frame_num*2, 
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv11)
    conv22 = BatchNormalization()(conv22, training=False)
    conv22 = LeakyReLU(alpha=0.2)(conv22)
    
    conv33 = Conv3D(filters=feature_multiplier*frame_num*4,
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv22)
    conv33 = BatchNormalization()(conv33, training=False)
    conv33 = LeakyReLU(alpha=0.2)(conv33)
    
    conv44 = Conv3D(filters=feature_multiplier*frame_num*8,
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv33)
    conv44 = BatchNormalization()(conv44, training=False)
    conv44 = LeakyReLU(alpha=0.2)(conv44)
    
    conv55 = Conv3D(filters=feature_multiplier*frame_num*16,
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv44)
    conv55 = BatchNormalization()(conv55, training=False)
    conv55 = LeakyReLU(alpha=0.2)(conv55)
    
    density_output = Reshape((frame_num,int(res*res*res/256)))(conv55)
    
    Input_velocity = Input(shape=(frame_num,res,res,res,3))

    conv1 = Reshape((res,res,res,frame_num*3))(Input_velocity)

    conv1 = Conv3D(filters=feature_multiplier*frame_num, 
                   kernel_size=surface_kernel_size,
                   strides=(2, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv1)
    conv1 = BatchNormalization()(conv1, training=False)
    conv1 = Activation('linear')(conv1)
    
    conv2 = Conv3D(filters=feature_multiplier*frame_num*2, 
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv1)
    conv2 = BatchNormalization()(conv2, training=False)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    
    conv3 = Conv3D(filters=feature_multiplier*frame_num*4,
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv2)
    conv3 = BatchNormalization()(conv3, training=False)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    
    conv4 = Conv3D(filters=feature_multiplier*frame_num*8,
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv3)
    conv4 = BatchNormalization()(conv4, training=False)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    
    conv5 = Conv3D(filters=feature_multiplier*frame_num*16,
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv4)
    conv5 = BatchNormalization()(conv5, training=False)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    
    velocity_output = Reshape((frame_num,int(res*res*res/256)))(conv5)
    
    #hidden_layer = concatenate([density_output, velocity_output],axis=2)
    hidden_layer = add([density_output, velocity_output])
    hidden_layer = LSTM(int(res*res*res/256), input_shape=(frame_num,int(res*res*res/256)), return_sequences=True,
                        activation='softsign', recurrent_activation='hard_sigmoid',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal', bias_initializer='zeros')(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer, training=False)
    hidden_layer = Dense(int(res*res*res/256), activation=None)(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer, training=False)
    hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer) 
    hidden_layer = Reshape((2,2,2,frame_num*int(res*res*res/2048)))(hidden_layer)
    
    deconv5 = Conv3DTranspose(filters=feature_multiplier*frame_num*8,
                               kernel_size=kernel_size,
                               strides=(2,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(hidden_layer)
    deconv5 = BatchNormalization()(deconv5, training=False)
    deconv5 = LeakyReLU(alpha=0.2)(deconv5)
    
    deconv4 = Conv3DTranspose(filters=feature_multiplier*frame_num*4,
                               kernel_size=kernel_size,
                               strides=(2,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv5)
    deconv4 = BatchNormalization()(deconv4, training=False)
    deconv4 = LeakyReLU(alpha=0.2)(deconv4)
    
    deconv3 = Conv3DTranspose(filters=feature_multiplier*frame_num*2,
                               kernel_size=kernel_size,
                               strides=(2,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv4)
    deconv3 = BatchNormalization()(deconv3, training=False)
    deconv3 = LeakyReLU(alpha=0.2)(deconv3)
    
    deconv2 = Conv3DTranspose(filters=feature_multiplier*frame_num*1,
                               kernel_size=kernel_size,
                               strides=(2,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv3)
    deconv2 = BatchNormalization()(deconv2, training=False)
    deconv2 = LeakyReLU(alpha=0.2)(deconv2)
    
    deconv1 = Conv3DTranspose(filters=frame_num*1,
                               kernel_size=surface_kernel_size,
                               strides=(2,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv2)
    deconv1 = BatchNormalization()(deconv1, training=False)
    #deconv1 = LeakyReLU(alpha=0.2)(deconv1)

    deconv1 = Activation('linear')(deconv1)
    
    outputs = Reshape((frame_num,res,res,res))(deconv1)

    Density_Prediction = Model(inputs=[Input_density, Input_velocity], outputs=outputs)
    Density_Prediction.summary()

    return Density_Prediction

def VelocityPrediction2D(res,frame_input,frame_output):
    #parameters
    feature_multiplier = 8
    surface_kernel_size = 4
    kernel_size = 2
    
    #model
    Inputs = Input(shape=(frame_input,res,res,2))
    conv1 = Conv3D(filters=feature_multiplier*1, 
                    kernel_size=surface_kernel_size,
                    strides=(1, 2, 2), padding='same', 
                    activation=None,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=None)(Inputs)
    conv1 = Activation('linear')(conv1)
    
    conv2 = Conv3D(filters=feature_multiplier*2, 
                   kernel_size=kernel_size,
                   strides=(1, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv1)
    conv2 = BatchNormalization()(conv2, training=False)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv3D(filters=feature_multiplier*4,
                   kernel_size=kernel_size,
                   strides=(1, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv2)
    conv3 = BatchNormalization()(conv3, training=False)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    
    conv4 = Conv3D(filters=feature_multiplier*8,
                   kernel_size=kernel_size,
                   strides=(1, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv3)
    conv4 = BatchNormalization()(conv4, training=False)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    
    hidden_layer = Reshape((frame_input,int(res*res/4)))(conv4)
    hidden_layer = LSTM(int(res*res/4), input_shape=(frame_input,int(res*res/4)), return_sequences=True,
                        activation='tanh', recurrent_activation='hard_sigmoid',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal', bias_initializer='zeros')(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer, training=False)
    hidden_layer = Reshape((int(res*res/4),frame_input))(hidden_layer)
    hidden_layer = Conv1D(filters=frame_output,
                          kernel_size=2,
                          strides=1,
                          activation='relu',
                          padding='same')(hidden_layer)
    hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer, training=False)
    hidden_layer = Reshape((frame_output,int(res/16),int(res/16),64))(hidden_layer)
    
    deconv4 = Conv3DTranspose(filters=feature_multiplier*4,
                               kernel_size=kernel_size,
                               strides=(1,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(hidden_layer)
    deconv4 = BatchNormalization()(deconv4, training=False)
    deconv4 = LeakyReLU(alpha=0.2)(deconv4)

    deconv3 = Conv3DTranspose(filters=feature_multiplier*2,
                               kernel_size=kernel_size,
                               strides=(1,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv4)
    deconv3 = BatchNormalization()(deconv3, training=False)
    deconv3 = LeakyReLU(alpha=0.2)(deconv3)
    
    deconv2 = Conv3DTranspose(filters=feature_multiplier*1,
                               kernel_size=kernel_size,
                               strides=(1,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv3)
    deconv2 = BatchNormalization()(deconv2, training=False)
    deconv2 = LeakyReLU(alpha=0.2)(deconv2)

    deconv1 = Reshape((frame_output,res,res,2))(deconv2)
    
    outputs = Activation('linear')(deconv1)
    
    velocity_prediction = Model(inputs=Inputs, outputs=outputs)
    velocity_prediction.summary()

    return velocity_prediction

def VelocityPrediction3D(res,frame_input,frame_output):
    #parameters
    feature_multiplier = 16
    surface_kernel_size = 4
    kernel_size = 2
    
    #model
    Inputs = Input(shape=(frame_input,res,res,res,3))
    conv1 = Reshape((res,res,res,3*frame_input))(Inputs)
    conv1 = Conv3D(filters=feature_multiplier*frame_input*1, 
                   kernel_size=surface_kernel_size,
                   strides=(2, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv1)
    #conv1 = BatchNormalization()(conv1, training=False)
    conv1 = Activation('linear')(conv1)
    
    conv2 = Conv3D(filters=feature_multiplier*frame_input*2, 
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same', 
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv1)
    conv2 = BatchNormalization()(conv2, training=False)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    
    conv3 = Conv3D(filters=feature_multiplier*frame_input*4,
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv2)
    conv3 = BatchNormalization()(conv3, training=False)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Dropout(0.2)(conv3)
    
    conv4 = Conv3D(filters=feature_multiplier*frame_input*8,
                   kernel_size=kernel_size,
                   strides=(2, 2, 2), padding='same',
                   activation=None,
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=None)(conv3)
    conv4 = BatchNormalization()(conv4, training=False)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    
    hidden_layer = Reshape((int(res*res*res/32),frame_input))(conv4)
    hidden_layer = Conv1D(filters=frame_output,
                          kernel_size=2,
                          strides=1,
                          activation='relu',
                          padding='same')(hidden_layer)
    #hidden_layer = Dense(frame_output, activation=None)(hidden_layer)
    hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer, training=False)
    hidden_layer = Reshape((frame_output,int(res*res*res/32)))(hidden_layer)
    hidden_layer = LSTM(int(res*res*res/32), input_shape=(frame_output,int(res*res*res/32)), return_sequences=True,
                        activation='softsign', recurrent_activation='hard_sigmoid',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal', bias_initializer='zeros')(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer, training=False)
    #hidden_layer = Dense(int(res*res/4), activation=None)(hidden_layer)
    #hidden_layer = BatchNormalization()(hidden_layer, training=False)
    hidden_layer = Reshape((4,4,4,frame_output*int(res*res*res/2048)))(hidden_layer)
    
    deconv4 = Conv3DTranspose(filters=feature_multiplier*frame_output*4,
                               kernel_size=kernel_size,
                               strides=(2,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(hidden_layer)
    deconv4 = BatchNormalization()(deconv4, training=False)
    deconv4 = LeakyReLU(alpha=0.2)(deconv4)

    deconv3 = Conv3DTranspose(filters=feature_multiplier*frame_output*2,
                               kernel_size=kernel_size,
                               strides=(2,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv4)
    deconv3 = BatchNormalization()(deconv3, training=False)
    deconv3 = LeakyReLU(alpha=0.2)(deconv3)
    deconv3 = Dropout(0.2)(deconv3)
    
    deconv2 = Conv3DTranspose(filters=feature_multiplier*frame_output,
                               kernel_size=kernel_size,
                               strides=(2,2,2), padding='same',
                               activation=None,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=None)(deconv3)
    deconv2 = BatchNormalization()(deconv2, training=False)
    deconv2 = LeakyReLU(alpha=0.2)(deconv2)
    
    deconv1 = Conv3DTranspose(filters=3*frame_output,
                              kernel_size=surface_kernel_size,
                              strides=(2,2,2), padding='same',
                              activation=None,
                              kernel_initializer='glorot_uniform',
                              kernel_regularizer=None)(deconv2)
    deconv1 = BatchNormalization()(deconv1, training=False)
    #deconv1 = LeakyReLU(alpha=0.2)(deconv1)
    deconv1 = Reshape((frame_output,res,res,res,3))(deconv1)
    
    outputs = Activation('linear')(deconv1)
    
    velocity_prediction = Model(inputs=Inputs, outputs=outputs)
    velocity_prediction.summary()

    return velocity_prediction