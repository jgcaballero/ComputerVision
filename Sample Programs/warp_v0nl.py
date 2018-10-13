"""
Warping using a homography matrix T
Uses operations on arrays for efficiency
Programmed by Olac Fuentes
Last modified October 10, 2018
"""

import numpy as np
import cv2
import time

start = time.time()
source_im = cv2.imread('tennis.jpg',1)
cv2.imshow('Source image',source_im)
cv2.waitKey(1)  
dest_im = np.zeros(source_im.shape, dtype=np.uint8)
                 
theta = np.radians(10)
dr = -300
dc = -450
c, s = np.cos(theta), np.sin(theta)
T0 = np.array(((c,-s, 0), (s, c, 0), (0, 0, 1))) # Rotate
T1 = np.array(((1,0, dr), (0, 1, dc), (0, 0, 1)))# Translate
T2 = np.array(((2, 0, 0), (0, 1, 0), (0, 0, 1))) # Scale
T3 = np.array(((2,0, dr), (0, 2, dc), (0, 0, 1)))# Scale and Translate
T4 = np.matmul(np.array(((c,-s, 0), (s, c, 0), (0, 0, 1))), T1) # Sequence of operations
T5 = np.matmul(np.array(((1,.5, 0), (0, 1, 0), (0, 0, 1))), T1) # Sequence of operations, shearing
T6 = np.matmul(np.array(((1,-.1, 0), (.5, 1, 0), (0, 0, 1))), T1) # Sequence of operations, shearing

#T = np.matmul(T1,T0)
#T = np.matmul(T0,T1)
T = T6

max_row = source_im.shape[0]-1
max_col = source_im.shape[1]-1
source_im[0]=0
source_im[max_row]=0         
source_im[:,0]=0
source_im[:,max_col]=0 

def f(r,c):
    d = np.dot(T,[r,c,1])
    rs = int(d[0]+.5) #Add 1/2, then truncate, which is equivalent to rounding
    cs = int(d[1]+.5)
    return rs, cs

def f(Cd):
    # Transforming point fp according to H
    # Convert to homogeneous coordinates if necessary
    if Cd.shape[0]==2:
          t = np.dot(T,np.vstack((Cd,np.ones(Cd.shape[1]))))
    else:
        t = np.dot(T,Cd)
    return (t[:2]+.5).astype(np.int)

ind = np.arange(source_im.shape[0]*source_im.shape[1])
row_vect = ind//source_im.shape[1]
col_vect = ind%source_im.shape[1]
Cd = np.vstack((row_vect,col_vect))

Cs = f(Cd)

Cs[Cs<0] = 0
Cs[0,Cs[0]>max_row] = max_row             
Cs[1,Cs[1]>max_col] = max_col      
      
dest_im = source_im[Cs[0],Cs[1],:].reshape(source_im.shape[0],source_im.shape[1],3)
        
cv2.imshow('Destination image no loop',dest_im)
cv2.waitKey(1)     
   
elapsed_time = time.time()-start
print('Elapsed time: {0:.2f} '.format(elapsed_time))   
print("Source origin:",Cd[:,0])
#cv2.destroyAllWindows()
