import numpy as np
import cv2
from PIL import Image

#1A
def grayscale():
    image = cv2.imread('images/cat.jpg')
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    cv2.imshow('result',gray_img) 
    
#2B
def negative():
    image = cv2.imread('images/cat.jpg')
    my_neg = 255 - image
    neg =  cv2.bitwise_not(image)
    cv2.imshow('neg',neg) 
    cv2.imshow('my_neg',my_neg) 
    
#2C
def scaled():
    print('scaled')
    
#2D
def cropped():
    image = cv2.imread('images/doggo.png')

    rows= image.shape[0]
    cols = image.shape[1]
    
    new_row = rows//2
    new_col = cols//2
    dst = np.empty([new_row,new_col])
    
    print(image[new_row:rows-new_row//2,new_col:cols-new_col//2])
    dst = image[new_row//2:rows-new_row//2,new_col//2:cols-new_col//2]
    
    cv2.imshow('crop',dst) 
    cv2.imshow('neg',image) 

#1G    
def gauss_filter():
    image = cv2.imread('images/cat.jpg')
    blur = cv2.GaussianBlur(image,(5,5),0)
    cv2.imshow('crop',image) 
    cv2.imshow('boxd',blur) 

#1H
def box_filter():
    image = cv2.imread('images/cat.jpg',0)
    box_size = 10
    kernel = np.ones((box_size,box_size))/(box_size*box_size)
    gray_frame_f = np.abs(cv2.filter2D(image,-1,kernel))
    cv2.imshow('crop',image) 
    cv2.imshow('boxd',gray_frame_f) 

#Number 4    
def is_same_picture():
    img1 = np.array(Image.open('images/cat.jpg'), dtype=np.uint8)
    img2 = np.array(Image.open('images/dog.jpg'), dtype=np.uint8)

    if(np.array_equal(img1, img2)):
        print('identical')
    else:
        print('NOT identical')

gauss_filter()

cv2.waitKey(0)
cv2.destroyAllWindows()