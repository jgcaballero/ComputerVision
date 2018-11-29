# Segmantation based on connected components 
# Programmed by Olac Fuentes
# Last modified November 19, 2018

import numpy as np
import cv2
import pylab as plt
import time
import random

def find(i):
    if S[i] <0:
        return i
    s = find(S[i]) 
    S[i] = s #Path compression
    return s

def union(i,j,thr):
    # Joins two regions if they are similar
    # Keeps track of size and mean color of regions
    ri =find(i)
    rj = find(j)
    if (ri!= rj):
        d =  sum_pixels[ri,:]/count[ri] - sum_pixels[rj,:]/count[rj]
        diff = np.sqrt(np.sum(d*d))
        if diff < thr:	
            S[rj] = ri
            count[ri]+=count[rj]
            count[rj]=0
            sum_pixels[ri,:]+=sum_pixels[rj,:]
            sum_pixels[rj,:]=0
                  
def initialize(I):
    rows = I.shape[0]
    cols = I.shape[1]   
    S=np.zeros(rows*cols).astype(np.int)-1
    count=np.ones(rows*cols).astype(np.int)       
    sum_pixels = np.copy(I).reshape(rows*cols,3)      
    return S, count, sum_pixels        

def connected_components_segmentation(I,thr):
    rows = I.shape[0]
    cols = I.shape[1]   
    for p in range(S.shape[0]):
        if p%cols < cols-1:  # If p is not in the last column
            union(p,p+1,thr) # p+1 is the pixel to the right of p  
        if p//cols < rows-1: # If p is not in the last row   
            union(p,p+cols,thr) # p+cols is the pixel to below p  

thr=0.2
I  =  (cv2.imread('quijote_lr.jpg',1)/255)
#I  =  (cv2.imread('capetown.jpg',1)/255)
#I  =  (cv2.imread('mayon.jpg',1)/255)
cv2.imshow('Original',I)

rows = I.shape[0]
cols = I.shape[1]   
S, count, sum_pixels = initialize(I)
connected_components_segmentation(I,thr)

print('Regions found: ',np.sum(S==-1))
print('Size 1 regions found: ',np.sum(count==1))

rand_cm = np.random.random_sample((rows*cols, 3))
seg_im_mean = np.zeros((rows,cols,3))
seg_im_rand = np.zeros((rows,cols,3))
for r in range(rows-1):
    for c in range(cols-1):
        f = find(r*cols+c)
        seg_im_mean[r,c,:] = sum_pixels[f,:]/count[f]
        seg_im_rand[r,c,:] = rand_cm[f,:]
            
cv2.imshow('Segmentation 1 - using mean colors',seg_im_mean)
cv2.imshow('Segmentation 2 - using random colors',seg_im_rand)
