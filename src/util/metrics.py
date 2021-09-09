# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 12:10:07 2021

@author: zhujy
"""

import numpy as np
import tensorflow as tf
import os
import time
import glob
import cv2
import math
import argparse
import pywt
from tqdm import tqdm

def psnr_den_2D(gt, pre):
    frames=gt.shape[0]
    gt=np.squeeze(gt)
    pre=np.squeeze(pre)
    sizex=gt.shape[1]
    sizey=gt.shape[2]
    psnr=np.zeros(frames,)
    for frame in range(frames):
        mse=0
        for i in range(sizex):
            for j in range(sizey):
                mse=mse+(gt[frame,i,j]-pre[frame,i,j])*(gt[frame,i,j]-pre[frame,i,j])
        mse=mse/(sizex*sizey)
        maxi=np.max(np.max(np.abs(gt[frame])))
        scale=255/maxi
        #psnr[frame]=10*math.log10(maxi*maxi/mse)
        psnr[frame]=10*math.log10(255*255/(mse*scale))
    return psnr

def cos_den_2D(gt, pre):
    frames=gt.shape[0]
    gt=np.squeeze(gt)
    pre=np.squeeze(pre)
    sizex=gt.shape[1]
    sizey=gt.shape[2]
    cos_sim=np.zeros(frames,)
    for frame in range(frames):
        sum_gt=0
        sum_pre=0
        gt_pre=0
        for i in range(sizex):
            for j in range(sizey):
                sum_gt=sum_gt+gt[frame,i,j]*gt[frame,i,j]
                sum_pre=sum_pre+pre[frame,i,j]*pre[frame,i,j]
                gt_pre=gt_pre+gt[frame,i,j]*pre[frame,i,j]
        cos_sim[frame]=gt_pre/math.sqrt(sum_gt)/math.sqrt(sum_pre)
    return cos_sim

def psnr_den_3D(gt, pre):
    frames=gt.shape[0]
    gt=np.squeeze(gt)
    pre=np.squeeze(pre)
    sizex=gt.shape[1]
    sizey=gt.shape[2]
    sizez=gt.shape[3]
    psnr=np.zeros(frames,)
    for frame in range(frames):
        mse=0
        for i in range(sizex):
            for j in range(sizey):
                for k in range(sizez):
                    mse=mse+(gt[frame,i,j,k]-pre[frame,i,j,k])*(gt[frame,i,j,k]-pre[frame,i,j,k])
        mse=mse/(sizex*sizey*sizez)
        maxi=np.max(np.max(np.max(np.abs(gt[frame]))))
        psnr[frame]=10*math.log10(255*maxi/mse)
    return psnr

def cos_den_3D(gt, pre):
    frames=gt.shape[0]
    gt=np.squeeze(gt)
    pre=np.squeeze(pre)
    sizex=gt.shape[1]
    sizey=gt.shape[2]
    sizez=gt.shape[3]
    cos_sim=np.zeros(frames,)
    for frame in range(frames):
        sum_gt=0
        sum_pre=0
        gt_pre=0
        for i in range(sizex):
            for j in range(sizey):
                for k in range(sizez):
                    sum_gt=sum_gt+gt[frame,i,j,k]*gt[frame,i,j,k]
                    sum_pre=sum_pre+pre[frame,i,j,k]*pre[frame,i,j,k]
                    gt_pre=gt_pre+gt[frame,i,j,k]*pre[frame,i,j,k]
        cos_sim[frame]=gt_pre/math.sqrt(sum_gt)/math.sqrt(sum_pre)
    return cos_sim

def psnr_vel_2D(gt,pre):
    frames=gt.shape[0]
    gt=np.squeeze(gt)
    pre=np.squeeze(pre)
    sizex=gt.shape[1]
    sizey=gt.shape[2]
    psnr=np.zeros(frames,)
    for frame in range(frames):
        maxi=0
        for i in range(sizex):
            for j in range(sizey):
                if maxi< math.sqrt(gt[frame,i,j,0]*gt[frame,i,j,0]+gt[frame,i,j,1]*gt[frame,i,j,1]):
                    maxi = math.sqrt(gt[frame,i,j,0]*gt[frame,i,j,0]+gt[frame,i,j,1]*gt[frame,i,j,1])
        mse=0
        for dim in range(2):
            for i in range(sizex):
                for j in range(sizey):
                    mse=mse+(gt[frame,i,j,dim]-pre[frame,i,j,dim])*(gt[frame,i,j,dim]-pre[frame,i,j,dim])
        mse=mse/(sizex*sizey)
        psnr[frame]=10*math.log10(255*maxi/mse)
    return psnr

def cos_vel_2D(gt, pre):
    frames=gt.shape[0]
    gt=np.squeeze(gt)
    pre=np.squeeze(pre)
    sizex=gt.shape[1]
    sizey=gt.shape[2]
    cos_sim=np.zeros(frames,)
    for frame in range(frames):
        sum_gt=0
        sum_pre=0
        gt_pre=0
        for i in range(sizex):
            for j in range(sizey):
                sum_gt=sum_gt+gt[frame,i,j,0]*gt[frame,i,j,0]+gt[frame,i,j,1]*gt[frame,i,j,1]
                sum_pre=sum_pre+pre[frame,i,j,0]*pre[frame,i,j,0]+pre[frame,i,j,1]*pre[frame,i,j,1]
                gt_pre+=math.sqrt(gt[frame,i,j,0]*gt[frame,i,j,0]+gt[frame,i,j,1]*gt[frame,i,j,1])*math.sqrt(pre[frame,i,j,0]*pre[frame,i,j,0]+pre[frame,i,j,1]*pre[frame,i,j,1])
                #gt_pre+=gt[frame,i,j,0]*pre[frame,i,j,0]+gt[frame,i,j,1]*pre[frame,i,j,1]
        cos_sim[frame]=gt_pre/math.sqrt(sum_gt)/math.sqrt(sum_pre)
    return cos_sim

def psnr_vel_3D(gt,pre):
    frames=gt.shape[0]
    gt=np.squeeze(gt)
    pre=np.squeeze(pre)
    sizex=gt.shape[1]
    sizey=gt.shape[2]
    sizez=gt.shape[3]
    psnr=np.zeros(frames,)
    for frame in range(frames):
        maxi=0
        for i in range(sizex):
            for j in range(sizey):
                for k in range(sizez):
                    if maxi< math.sqrt(gt[frame,i,j,k,0]*gt[frame,i,j,k,0]+gt[frame,i,j,k,1]*gt[frame,i,j,k,1]+gt[frame,i,j,k,2]*gt[frame,i,j,k,2]):
                        maxi = math.sqrt(gt[frame,i,j,k,0]*gt[frame,i,j,k,0]+gt[frame,i,j,k,1]*gt[frame,i,j,k,1]+gt[frame,i,j,k,2]*gt[frame,i,j,k,2])
        mse=0
        for dim in range(3):
            for i in range(sizex):
                for j in range(sizey):
                    for k in range(sizez):
                        mse=mse+(gt[frame,i,j,k,dim]-pre[frame,i,j,k,dim])*(gt[frame,i,j,k,dim]-pre[frame,i,j,k,dim])
        mse=mse/(sizex*sizey*sizez)
        psnr[frame]=10*math.log10(255*maxi/mse)
    return psnr

def cos_vel_3D(gt, pre):
    frames=gt.shape[0]
    gt=np.squeeze(gt)
    pre=np.squeeze(pre)
    sizex=gt.shape[1]
    sizey=gt.shape[2]
    sizez=gt.shape[3]
    cos_sim=np.zeros(frames,)
    for frame in range(frames):
        sum_gt=0
        sum_pre=0
        gt_pre=0
        for i in range(sizex):
            for j in range(sizey):
                for k in range(sizez):
                    sum_gt=sum_gt+gt[frame,i,j,k,0]*gt[frame,i,j,k,0]+gt[frame,i,j,k,1]*gt[frame,i,j,k,1]+gt[frame,i,j,k,2]*gt[frame,i,j,k,2]
                    sum_pre=sum_pre+pre[frame,i,j,k,0]*pre[frame,i,j,k,0]+pre[frame,i,j,k,1]*pre[frame,i,j,k,1]+pre[frame,i,j,k,2]*pre[frame,i,j,k,2]
                    gt_pre+=math.sqrt(gt[frame,i,j,k,0]*gt[frame,i,j,k,0]+gt[frame,i,j,k,1]*gt[frame,i,j,k,1]+gt[frame,i,j,k,2]*gt[frame,i,j,k,2])*math.sqrt(pre[frame,i,j,k,0]*pre[frame,i,j,k,0]+pre[frame,i,j,k,1]*pre[frame,i,j,k,1]+pre[frame,i,j,k,2]*pre[frame,i,j,k,2])
                #gt_pre+=gt[frame,i,j,0]*pre[frame,i,j,0]+gt[frame,i,j,1]*pre[frame,i,j,1]
        cos_sim[frame]=gt_pre/math.sqrt(sum_gt)/math.sqrt(sum_pre)
    return cos_sim