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
    
def triangle():
    #0,0 0,1 0,2
    #1,0 1,1
    #2,0   
    #        0,2
    #    1,1 1,2
    #2.0 2,1 2,2
    a = np.array([[1,2,3],
                  [4,5,6], 
                  [7,8,9]])
    
    triangle2 = np.full_like(a, 0)
        
    row = round((a.shape[0]/2)+.5)
    col = round((a.shape[1]/2)+.5)
    
    for y in range (0,row+1):
        for x in range (0,col+1):
            print(a[y,x])
            triangle2[y,x] = a[y,x]
            if(x == col):
                print('DIS', a[y][x])
            else:
                a[y,x] = 0
        col -= 1
        
        
    print('==')
    print(a)
    print('==')
    print(triangle2)
    
triangle()
#real_value_indexing()