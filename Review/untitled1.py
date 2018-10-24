"""
Warping using a homography matrix T
Uses loops for clarity, but it's slooow
Programmed by Olac Fuentes
Last modified October 10, 2018
"""


import numpy as np
import cv2
import time

start = time.time()
source_im = cv2.imread('images/cat.jpg',1)
cv2.imshow('Source image',source_im)
cv2.waitKey(1)  
dest_im = np.zeros(source_im.shape, dtype=np.uint8)
                 
theta = np.radians(80)
dr = -30
dc = -45
c, s = np.cos(theta), np.sin(theta)
T0 = np.array(((c,-s, 0), (s, c, 0), (0, 0, 1))) # Rotate
T1 = np.array(((1,0, dr), (0, 1, dc), (0, 0, 1)))# Translate
T2 = np.array(((2, 0, 0), (0, 1, 0), (0, 0, 1))) # Scale
T3 = np.array(((2,0, dr), (0, 2, dc), (0, 0, 1)))# Scale and Translate
T4 = np.matmul(np.array(((c,-s, 0), (s, c, 0), (0, 0, 1))), T1) # Sequence of operations
T5 = np.matmul(np.array(((1,.5, 0), (0, 1, 0), (0, 0, 1))), T1) # Sequence of operations, shearing
T6 = np.matmul(np.array(((1,-.1, 0), (.5, 1, 0), (0, 0, 1))), T1) # Sequence of operations, shearing

T = np.matmul(T1,T0)
#T = np.matmul(T0,T1)
T = T4

max_row = source_im.shape[0]-1
max_col = source_im.shape[1]-1

def f(r,c):
    d = np.dot(T,[r,c,1])
    rs = int(d[0]+.5) #Add 1/2, then truncate, which is equivalent to rounding
    cs = int(d[1]+.5)
    return rs, cs

for r in range(dest_im.shape[0]):
    for c in range(dest_im.shape[1]):
        rs, cs = f(r,c)
        if(rs<0 or cs<0 or rs>max_row or cs>max_col): # Out of bounds pixels are black
            dest_im[r,c,:] = [0,0,0]
        else:
            dest_im[r,c,:] = source_im[rs,cs,:]
        
cv2.imshow('Destination image loop',dest_im)
cv2.waitKey(1)     
   
elapsed_time = time.time()-start
print('Elapsed time: {0:.2f} '.format(elapsed_time))   
print("Source origin:",f(0,0))
#cv2.destroyAllWindows()