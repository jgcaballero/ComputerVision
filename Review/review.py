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
    img = cv2.imread('images/cat.jpg')
    img1 = img - np.min(img)
    img1 = img1 / np.max(img1)
    return img1

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
    #gauss_blur = np.array([[1,2,1], [2,4,2], [1,2,1]])/16
    #dst = cv2.filter2D(deer,-1,kernel)
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

#Number 3
def swap_RB():
    img = cv2.imread('images/cat.jpg',0)
    temp = img[:,:,0]
    img[:,:,0] = img[:,:,2]
    img[:,:,2] = temp

    #Optimal Way
    return img[:,:,[2,1,0]]


#Number 4    
def is_same_picture():
    img1 = np.array(Image.open('images/cat.jpg'), dtype=np.uint8)
    img2 = np.array(Image.open('images/dog.jpg'), dtype=np.uint8)

    if(np.array_equal(img1, img2)):
        print('identical')
    else:
        print('NOT identical')

#Number 5
def histogram_bars(img, n):
    img = cv2.imread('images/cat.jpg',0)
    h = np.zeros(n).astype(np.int)
    b = (img*n).astype(np.int)
    for i in range(n):
        h[i] = np.sum(b==i)
    h[n-1] += np.sum(b==n)
    return h

#6A
def array_warp(I,W):
    rows = I.shape[0]
    cols = I.shape[1]
    row_mat, col_mat = coord_mat(rows,cols)
    row_mat == row_mat + W[:,:,0]
    col_mat == col_mat + W[:,:,1]

#6B
def point_warp(rows,cols, p,q,k):
    row_mat, col_mat = coord_mat(rows,cols)
    W = np.zeros((rows,col,2))
    dist_r = row_mat - q[0]
    dist_c = col_mat - q[1]
    dist = -np.sqrt(dist_r*dist_r + dist_c*dist_c)/k
    W[:,:,0] = np.exp(dist)*(p[0] - q[0])
    W[:,:,1] = np.exp(dist)*(p[1] - q[1])
    W = (W+0.5).astype(np.int)
    return W

#Number 7
def knn(Xtrain, Ytrain, k, Xtest):
    Ytest = np.zeros((Ytrain.shape[0],Xtest.shape[1]))
    for i in range(Xtest.shape[1]):
        diff = Xtrain - Xtest[i]
        diff = diff * diff
        diff = np.sum(diff, axis = 0)
        diff = np.sqrt(diff)
        n = np.argsort(diff)[:k]
        w = 1/diff[n]  
        w = w/np.sum(w)
        Ytest[:,i] = Ytrain[:,n]*w
    return Ytest



gauss_filter()

#real value
#census transformation
#interpolation
#sort in ascending order warping?

cv2.waitKey(0)
cv2.destroyAllWindows()