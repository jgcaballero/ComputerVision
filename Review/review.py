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
    
def mirror():
    image = cv2.imread('images/cat.jpg')
    mirror = image[::1,::-1] #mirror
    upside = image[::-1,::1] #
    upsideMirror = image[::-1,::-1] #-x -y axis
    identical = image[::] #identical

    cv2.imshow('image',image) 
    cv2.imshow('mirror',mirror) 
    cv2.imshow('upside',upside) 
    cv2.imshow('identical',identical) 
    cv2.imshow('upsideMirror',upsideMirror) 
    
def map_to_01_range():
    img = cv2.imread('images/cat.jpg') 
    img_01 = img-np.min(img)
    img_01 = img_01/np.max(img_01)
    cv2.imshow('image',img) 
    cv2.imshow('img_01',img_01) 
        
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
    
def half_size():
    img = cv2.imread('images/doggo.png')
    img = img / 255
    # Ignore last row/column if size is odd
    rows = 2*(img.shape[0]//2)
    cols = 2*(img.shape[1]//2)
    hs = img[np.arange(0,rows,2)] + img[np.arange(1,rows,2)] 
    hs = hs[:,np.arange(0,cols,2)] + hs[:,np.arange(1,cols,2)]    
    hs = hs/4
    
    cv2.imshow('img',img) 
    cv2.imshow('hs',hs) 
    
def convolve(image,kernel):
    result = np.zeros((image.shape[0]-kernel.shape[0]+1,image.shape[1]-kernel.shape[1]+1))
    row = kernel.shape[0]    
    col = kernel.shape[1]
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            result[r,c] = np.sum(image[r:r+row,c:c+col]*kernel)
    return result

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

half_size()

cv2.waitKey(0)
cv2.destroyAllWindows()