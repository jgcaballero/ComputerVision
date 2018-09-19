"""
Convolution example
Programmed by Olac Fuentes
Last modified September 18, 2018
"""
import numpy as np
import cv2

def convolve(image,kernel):
    result = np.zeros((image.shape[0]-kernel.shape[0]+1,image.shape[1]-kernel.shape[1]+1))
    rk = kernel.shape[0]    
    ck = kernel.shape[1]
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            result[r,c] = np.sum(image[r:r+rk,c:c+ck]*kernel)
    return result

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255
 
k = np.array([[1,-1],[1,-1]])
res = convolve(frame,k)
cv2.imshow('image',frame) 
cv2.imshow('convolution result',np.abs(res)) 
