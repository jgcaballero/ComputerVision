#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:31:37 2018

@author: josecaballero
"""

import numpy as np
import cv2
from PIL import Image
import pylab as plt
#plt.switch_backend('TKAgg')

#✔️
def real_value_indexing(): #✔️
    
    r0 = 1
    r1 = 1
    c0 = 2
    c1 = 2
    
    counter = 0
    iterations = 0;
    
    a = np.array([[1,2,3],
                  [4,5,6], 
                  [7,8,9]])


    for y in range (r0, c0+1):
        for x in range (r1, c1+1):
            counter += a[y,x]
            iterations += 1
            print(a[y,x])  
    
    print(counter/iterations)
    
def triangle(img):    
    triangle2 = np.full_like(img, 0)
        
    row = img.shape[0]
    col = img.shape[1]
    slope = round(img.shape[1]/img.shape[0])
    
    for y in range (1,row):
        for x in range (1,col):
            if(x == col):
                triangle2[y,x] = img[y,x]
                continue
            else:
                triangle2[y,x] = img[y,x]
                img[y,x] = 0
        col -= slope

    return img, triangle2

def transform(H,fp):
    # Transforming point fp according to H
    # Convert to homogeneous coordinates if necessary
    if fp.shape[0]==2:
          t = np.dot(H,np.vstack((fp,np.ones(fp.shape[1]))))
    else:
        t = np.dot(H,fp)
    return t[:2]
  
def demo():
    img = cv2.imread('images/banner_small.jpg',1)
    triangle1, triangle2 = triangle(img)
    #cv2.imshow('image',triangle1)
    #cv2.imshow('image2',triangle2)
    
    im2 = np.array(triangle2, dtype=np.uint8)
    plt.figure(1)
    plt.imshow(triangle2)
    plt.show()
    
    source_im = np.array(Image.open('images/tennis.jpg'), dtype=np.uint8)
    plt.figure(2)
    plt.imshow(source_im)
    plt.show()
    
    x = [0,0,im2.shape[0]-1]
    y = [0,im2.shape[1]-1,0]
    fp = np.vstack((x,y))
    
    #print("Click destination points, top-left, top-tight, and bottom-left corners")
    tp = np.asarray(plt.ginput(n=3), dtype=np.float).T
    tp = tp[[1,0],:]
    print(fp)
    print(tp)
    
    #Using pseudoinverse
    # Generating homogeneous coordinates
    fph = np.vstack((fp,np.ones(fp.shape[1])))
    tph = np.vstack((tp,np.ones(tp.shape[1])))
    H = np.dot(tph,np.linalg.pinv(fph))
    
    print((transform(H,fp)+.5).astype(np.int))
    
    #Generating pixel coordinate locations
    ind = np.arange(im2.shape[0]*im2.shape[1])
    row_vect = ind//im2.shape[1]
    col_vect = ind%im2.shape[1]
    coords = np.vstack((row_vect,col_vect))
    
    new_coords = transform(H,coords).astype(np.int)
    target_im = source_im
    target_im[new_coords[0],new_coords[1],:] = im2[coords[0],coords[1],:]

    plt.figure(3)
    plt.imshow(target_im)
    plt.show()

    

demo()

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
   cv2.destroyAllWindows()
   
#real_value_indexing()