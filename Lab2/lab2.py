import numpy as np
import cv2
from matplotlib import pyplot as plt


cheetah = cv2.imread('images/cheetah.jpg', 0)
city = cv2.imread('images/city.jpg', 0)
deer = cv2.imread('images/deer.jpg', 0)
leopard = cv2.imread('images/leopard.jpg', 0)
ny = cv2.imread('images/ny.jpg', 0)
tricycle = cv2.imread('images/tricycle.jpg', 0)


# Cat fix using a box filter approach using a kernel of 3 by 3 and averaging the pixels
def fix_cat():
    cat = cv2.imread('images/cat.jpg',0)
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(cat,-1,kernel)
    res = np.hstack((cat,dst))
    cv2.imshow('result',res) 


#Cheetah
#Ctiy
#Deer
    
#Dog
def fix_dog():
    dog = cv2.imread('images/dog.jpg', 0) #histogram equalization
    equ = cv2.equalizeHist(dog)
    res = np.hstack((dog,equ))
    cv2.imshow('result',res) 

#Husky
def fix_husky():
    husky = cv2.imread('images/husky.jpg', 0) #histogram equalization
    equ = cv2.equalizeHist(husky)
    res = np.hstack((husky,equ))
    cv2.imshow('result',res) 

#leopard
#ny
    
#rose
def fix_rose():
    rose = cv2.imread('images/rose.jpg') #median blur to fix salt and pepper
    blur = cv2.medianBlur(rose,5)
    res = np.hstack((rose,blur))
    cv2.imshow('result',res) 

#tricycle


fix_husky()
   
#cv2.imshow('image',rose)
#cv2.imshow('result',res) 
#cv2.imshow('result',equ) 


k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    #cv2.imwrite('images/messigray.png',img)
    cv2.destroyAllWindows()