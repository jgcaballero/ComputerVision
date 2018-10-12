#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:31:37 2018

@author: josecaballero
"""

import numpy as np
import cv2

#image, r0, c0, r1, c1
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
    #0,0 0,1 0,2
    #1,0 1,1
    #2,0   
    #        0,2
    #    1,1 1,2
    #2.0 2,1 2,2
    
    img2 = np.array([[1,2,3,4,5,6,7,8,9,10],
                  [1,2,3,4,5,6,7,8,9,10],
                  [1,2,3,4,5,6,7,8,9,10]])
    
    triangle2 = np.full_like(img, 0)
        
    row = img.shape[0]
    col = img.shape[1]
    print(img)
    
    print('row', row)
    print('col', col)
    
    for y in range (1,row):
        for x in range (1,col):
            #print(y,x)
            #print(img[y,x])
            #triangle2[y,x] = img[y,x]
            if(x == col):
                triangle2[y,x] = img[y,x]
                continue
            else:
                triangle2[y,x] = img[y,x]
                img[y,x] = 0
        col -= round(img.shape[1]/img.shape[0])
        print('res', round(img.shape[1]/img.shape[0]))

        
        #print('col', col)
        #print('row', row)
        
        
    print('==')
    print(img)
    print('==')
    print(triangle2)
    
    #return triangle2
    return img, triangle2
  
def wat():
    print("wat")
    img = cv2.imread('images/banner_small.jpg',1)
    #triangle()
    triangle1, triangle2 = triangle(img)
    #triangle2 = triangle(img)
    cv2.imshow('image',triangle1)
    cv2.imshow('image2',triangle2)
    

wat()

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
   cv2.destroyAllWindows()
   
#real_value_indexing()