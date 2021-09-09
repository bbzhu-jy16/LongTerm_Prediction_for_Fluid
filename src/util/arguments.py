# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 12:15:51 2021

@author: zhujy
"""

import argparse

def Density2D_parse():
    parser=argparse.ArgumentParser(description="DensityPrediction2D")
    parser.add_argument('--mode','-m',choices=["train","test"],default="test",help='Program Mode')
    parser.add_argument('--resolution','-r',type=int,default=64,help='data resolution')
    parser.add_argument('--frame0','-f0',type=int,default=10,help='frame0')
    parser.add_argument('--frame_num','-fn',type=int,default=150,help='frame num')
    parser.add_argument('--epochs','-e',type=int,default=40,help="epoch num")
    parser.add_argument('--batch_size','-bs',type=int,default=32,help="batch size")
    parser.add_argument('--density_train_path','-n',default="../datasets/2D_gas64_200/density_train")
    parser.add_argument('--density_test_path','-n',default="../datasets/2D_gas64_200/density_valid")
    parser.add_argument('--velocity_train_path','-n',default="../datasets/2D_gas64_200/velocity_train")
    parser.add_argument('--velocity_test_path','-n',default="../datasets/2D_gas64_200/velocity_valid")
    parser.add_argument('--model_path','-n',default="../density_model",help="path to save model")
    parser.add_argument('--model_name','-n',default="Long_Density_Prediction64.h5")
    parser.add_argument('--test_save_path','-n',default="../predictions/2D_long_den64/")
    parser.add_argument('--scale','-s',type=int,default=1,help="scale parameter")

    args=parser.parse_args()
    
    del parser
    
    return args

def Density3D_parse():
    parser=argparse.ArgumentParser(description="DensityPrediction3D")
    parser.add_argument('--mode','-m',choices=["train","test"],default="test",help='Program Mode')
    parser.add_argument('--resolution','-r',type=int,default=64,help='data resolution')
    parser.add_argument('--epochs','-e',type=int,default=40,help="epoch num")
    parser.add_argument('--batch_size','-bs',type=int,default=4,help="batch size")
    parser.add_argument('--frame0','-f0',type=int,default=30,help='frame0')
    parser.add_argument('--frame_num','-fn',type=int,default=30,help='frame num')
    parser.add_argument('--density_train_path','-n',default="../datasets/3D_smoke_obstacle/density_train")
    parser.add_argument('--density_test_path','-n',default=".../datasets/3D_smoke_obstacle/density_valid")
    parser.add_argument('--velocity_train_path','-n',default="../datasets/3D_smoke_obstacle/velocity_train")
    parser.add_argument('--velocity_test_path','-n',default="../datasets/3D_smoke_obstacle/velocity_valid")
    parser.add_argument('--model_path','-n',default="../density_model",help="path to save model")
    parser.add_argument('--model_name','-n',default="3D_Long_Density_Prediction64.h5")
    parser.add_argument('--test_save_path','-n',default="../predictions/3D_long_den64/")
    parser.add_argument('--scale','-s',type=int,default=1,help="scale parameter")
    
    args=parser.parse_args()
    
    del parser
    
    return args

def Velocity2D_parse():
    parser=argparse.ArgumentParser(description="VelocityPrediction2D")
    parser.add_argument('--mode','-m',choices=["train","test"],default="test",help='Program Mode')
    parser.add_argument('--types','-t',choices=["gas","liquid"],default="gas",help='Fluid Type')
    parser.add_argument('--resolution','-r',type=int,default=64,help='data resolution')
    parser.add_argument('--epochs','-e',type=int,default=40,help="epoch num")
    parser.add_argument('--batch_size','-bs',type=int,default=32,help="batch size")
    parser.add_argument('--frame0','-f0',type=int,default=10,help='frame0')
    parser.add_argument('--frame_input','-fi',type=int,default=30,help='frame input')
    parser.add_argument('--frame_num','-fn',type=int,default=50,help='frame num')
    parser.add_argument('--scale','-s',type=int,default=1,help="scale parameter")
    parser.add_argument('--velocity_train_path','-n',default="../datasets/2D_gas64_200/velocity_train")
    parser.add_argument('--velocity_test_path','-n',default="../datasets/2D_gas64_200/velocity_valid")
    parser.add_argument('--model_path','-n',default="../velocity_model",help="path to save model")
    parser.add_argument('--model_name','-n',default="Long_Velocity_Prediction64.h5")
    parser.add_argument('--test_save_path','-n',default="../predictions/2D_long_vel64/")

    args=parser.parse_args()
    
    del parser
    
    return args

def Velocity3D_parse():
    parser=argparse.ArgumentParser(description="VelocityPrediction3D")
    parser.add_argument('--mode','-m',choices=["train","test"],default="test",help='Program Mode')
    parser.add_argument('--types','-t',choices=["gas","liquid"],default="gas",help='Fluid Type')
    parser.add_argument('--resolution','-r',type=int,default=64,help='data resolution')
    parser.add_argument('--epochs','-e',type=int,default=40,help="epoch num")
    parser.add_argument('--batch_size','-bs',type=int,default=4,help="batch size")
    parser.add_argument('--frame0','-f0',type=int,default=30,help='frame0')
    parser.add_argument('--frame_input','-fi',type=int,default=30,help='frame input')
    parser.add_argument('--frame_num','-fn',type=int,default=30,help='frame num')
    parser.add_argument('--scale','-s',type=int,default=1,help="scale parameter")
    parser.add_argument('--velocity_train_path','-n',default="../datasets/3D_smoke_obstacle/velocity_train")
    parser.add_argument('--velocity_test_path','-n',default="../datasets/3D_smoke_obstacle/velocity_valid")
    parser.add_argument('--model_path','-n',default="../velocity_model",help="path to save model")
    parser.add_argument('--model_name','-n',default="3D_Long_Velocity_Prediction64.h5")
    parser.add_argument('--test_save_path','-n',default="../predictions/3D_long_vel64/")

    args=parser.parse_args()
    
    del parser
    
    return args
