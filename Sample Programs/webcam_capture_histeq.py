import numpy as np
import cv2
import time
 
cap = cv2.VideoCapture(0)
start = time.time()
count=0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    count+=1
    
    gray_frame_heq = cv2.equalizeHist(gray_frame)
    res = np.hstack((gray_frame,gray_frame_heq))
    cv2.imshow('frame',res) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
elapsed_time = time.time()-start
# When everything done, release the capture
print('Capture speed: {0:.2f} frames per second'.format(count/elapsed_time))   
cap.release()
cv2.destroyAllWindows()
