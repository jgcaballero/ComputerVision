import numpy as np
import cv2
from matplotlib import pyplot as plt


cat = cv2.imread('images/cat.jpg',0)
cheetah = cv2.imread('images/cheetah.jpg', 0)
city = cv2.imread('images/city.jpg', 0)
deer = cv2.imread('images/deer.jpg', 0)
dog = cv2.imread('images/dog.jpg', 0)
husky = cv2.imread('images/husky.jpg', 0)
leopard = cv2.imread('images/leopard.jpg', 0)
ny = cv2.imread('images/ny.jpg', 0)
rose = cv2.imread('images/rose.jpg', 0)
tricycle = cv2.imread('images/tricycle.jpg', 0)

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#kernel = np.ones((5,5),np.float32)/25

dst = cv2.filter2D(rose,-1,kernel)
#equ = cv2.equalizeHist(rose)
res = np.hstack((rose,dst))
   
#cv2.imshow('image',cat)
cv2.imshow('result',res) 
#cv2.imshow('result',equ) 


k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    #cv2.imwrite('images/messigray.png',img)
    cv2.destroyAllWindows()