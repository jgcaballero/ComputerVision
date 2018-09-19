# Ghost images
# Programmed by OLac Fuentes
# Last modified September 10, 2018
# Make sure first image captures background only
import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
start = time.time()
count=0
ret, frame0 = cap.read()
frame0=frame0//2
while(True):
    ret, frame = cap.read()
    count+=1
    cv2.imshow('frame',frame//2+frame0) #
    #cv2.imshow('frame',diff)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
elapsed_time = time.time()-start
print('Capture speed: {0:.2f} frames per second'.format(count/elapsed_time))   
cap.release()
cv2.destroyAllWindows()
