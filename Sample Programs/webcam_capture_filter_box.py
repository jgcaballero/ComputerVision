import numpy as np
import cv2
import time
from scipy.stats import rankdata

 
cap = cv2.VideoCapture(0)
start = time.time()
count=0
box_size = 10
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255
    gray_frame = frame
    count+=1
    kernel = np.ones((box_size,box_size))/(box_size*box_size)
    gray_frame_f = np.abs(cv2.filter2D(gray_frame,-1,kernel))
    res = np.hstack((gray_frame,gray_frame_f))
    cv2.imshow('frame',res) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
elapsed_time = time.time()-start
# When everything done, release the capture
print('Capture speed: {0:.2f} frames per second'.format(count/elapsed_time))   
cap.release()
cv2.destroyAllWindows()
