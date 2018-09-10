import numpy as np
import cv2
import time
import math
from datetime import datetime


cap = cv2.VideoCapture(0)
start = time.time()
count=0

def correct_ilumination(my_image):    
    height, width = my_image.shape
    result = np.zeros(my_image.shape, my_image.dtype)
    
    alpha = 1.0 # Simple contrast control
    beta = 0    # Simple brightness control
    
    for y in range(my_image.shape[0]):
        for x in range(my_image.shape[1]):
            result[y,x] = np.clip(((my_image[y,x] - beta)/(alpha-beta)),0,255)
    return result        
 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # 1. Displays the gray level version of the image.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Displays the negative of the gray level version of the img.
    #neg_gray = cv2.bitwise_not(gray)
   
    # 3. Display the mirrored version of the original color image.
    #mirror = cv2.flip(frame, flipCode=1)

    # 4. Display the original color image upside down.
    #flip180 = cv2.flip(frame, flipCode=-1)

    # 5. Write to a file one frame every n seconds, where n is a user-supplied parameter.
    #n = 8
    #elapsed_time = time.time()-start
    #floored_time = math.floor(elapsed_time)
    #if floored_time % n == 0:
        # print(floored_time)
    #    cv2.imwrite("frame%d.jpg" % floored_time, frame)     # save frame as JPEG file      

    # 6. Display an illumination-corrected version of the gray level version of the image. To do this, map the
    # highest intensity found in the image to 1 and the lowest intensity to 0. Let max(I) and min(I) be the
    # highest and lowest intensities in image I, then the corrected image C is given by
    # C[i][j] = I[i][j] − min(I) / max(I) − min(I)
    #corrected_gray = correct_ilumination(gray)
    
    # 7. Build a motion detector. Your program should write to a file one frame every n seconds for k seconds
    #after motion is detected, where n and k are user-supplied parameters. It is up to you to decide what
    #constitutes motion and how to detect it.
    frame1 = gray
    frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    frame3 = cv2.absdiff(frame1, frame2)
    
    maxNum = np.amax(frame3)
    #print("MAX")
    print(maxNum)
    
    if maxNum > 35:
        print("Motion Detected!")
        
 

    
    count+=1
 
    # Display the resulting frame
    # Pass parameter to change image e.g gray instead of frame.
    # cv2.imshow('frame',frame)
    cv2.imshow('frame',frame1)
    cv2.imshow('frame',frame2)
    # cv2.imshow('frame',neg_gray)
    # cv2.imshow('frame',mirror)
    # cv2.imshow('frame',flip180)
    #cv2.imshow('frame',corrected_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count%30==0:
        print(np.max(frame),np.min(frame))
    
elapsed_time = time.time()-start
print('Capture speed: {0:.2f} frames per second'.format(count/elapsed_time))   



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()