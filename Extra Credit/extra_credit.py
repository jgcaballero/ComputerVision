import numpy as np
import cv2
import pylab as plt


#a)
def grayscale():
    image = cv2.imread('images/quijote_lr.jpg')
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    cv2.imshow('result',gray_img) 
    
#b)
def rotate():
    image2 = cv2.imread('images/quijote_lr.jpg')
    upside = image2[::-1,::1]/255

    row = upside.shape[0]
    col = upside.shape[1]
    
    rotated = np.zeros((col,row,3))
    
    print('row',upside.shape[0])
    print('col',upside.shape[1])
    
    
    for y in range(upside.shape[0]):
        for x in range(upside.shape[1]):
            rotated[x,y] = upside[y,x]
            
    print(rotated)
    #cv2.imshow('result',image2) 
    cv2.imshow('rotated',rotated) 
   
#c)
def box_filter():
    image = cv2.imread('images/quijote_lr.jpg')
    box_size = 10
    kernel = np.ones((box_size,box_size))/(box_size*box_size)
    blur = np.abs(cv2.filter2D(image,-1,kernel))
    #cv2.imshow('crop',image) 
    #cv2.imshow('boxd',gray_frame_f) 
    res = np.hstack((image,blur))
    cv2.imshow('result',res) 
    
#d)
def enlarge():
    image = cv2.imread('images/quijote_lr.jpg')
    new_row = image.shape[0]
    new_col = image.shape[1]*2    
    enlarge = np.zeros((new_row, new_col,3))

    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if(x != image.shape[1] - 2):
                enlarge[y,x*2] = image[y,x]
                                
    for y in range(enlarge.shape[0]):
        for x in range(enlarge.shape[1]-2):
            px0 = enlarge[y,x]
            px1 = enlarge[y,x+2]
            new_px = (px0 + px1)//2
            enlarge[y,x+1] = new_px

    cv2.imshow('enlarge',enlarge/255) 
    
#e)
def edges():
    image = cv2.imread('images/quijote_lr.jpg',-1)
    kernel_t = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    edgez =cv2.filter2D(image,-1,kernel_t)
    
    res = np.hstack((image,edgez))
    cv2.imshow('edges',res) 


    

    
enlarge()
    
cv2.waitKey(0)
cv2.destroyAllWindows()