import numpy as np
import cv2
from PIL import Image
import pylab as plt
import math


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
    gauss_blur = np.array([[1,2,1], [2,4,2], [1,2,1]])/16
    dst = cv2.filter2D(image,-1,gauss_blur)
    res = np.hstack((image,dst))
    cv2.imshow('result',res) 
  

#1H
def box_filter():
    image = cv2.imread('images/cat.jpg',0)
    box_size = 10
    kernel = np.ones((box_size,box_size))/(box_size*box_size)
    box = np.array([[1,1,1], [1,1,1], [1,1,1]])/9
    gray_frame_f = np.abs(cv2.filter2D(image,-1,box))
    #cv2.imshow('crop',image) 
    #cv2.imshow('boxd',gray_frame_f) 
    res = np.hstack((image,gray_frame_f))
    cv2.imshow('result',res) 
    
def threshold():
    gray_frame = cv2.imread('images/ny.jpg',0)
    thr_frame = (gray_frame>np.median(gray_frame)).astype(np.uint8)*255
    res = np.hstack((gray_frame,thr_frame))
    cv2.imshow('frame',res)
    
def threshold2():
    img = cv2.imread('images/ny.jpg',0)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(img[y,x] > 128):
                img[y,x] = 255
            else:
                img[y,x] = 0
            
    cv2.imshow('frame',img) 

                        

def edges():
    image = cv2.imread('images/ny.jpg')
    kernel_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernel_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    #np.array([[-1,-1],[1,1]])
    edges = np.abs(cv2.filter2D(image,-1,kernel_v))+np.abs(cv2.filter2D(image,-1,kernel_h))
    res = np.hstack((image,edges))
    cv2.imshow('edges',res) 

def hist2():
    img = cv2.imread('images/cat.jpg',0)
    flat = img.flatten() 
    n = 4
    bar_hist = np.zeros(n)
    raange = 255 // n

    for x in range(len(flat)):
        px = flat[x]
        for i in range(1,n+1):
            if(px <= raange*i):
                bar_hist[i-1] += 1
                break
                
    print(bar_hist)
    
def hist3():
    I = cv2.imread('images/cat.jpg',0)
    n = 4
    plt.hist(I.ravel(),n,[0,256])
    plt.show()
    
#Number 4    
def is_same_picture():
    img1 = np.array(Image.open('images/cat.jpg'), dtype=np.uint8)
    img2 = np.array(Image.open('images/dog.jpg'), dtype=np.uint8)

    if(np.array_equal(img1, img2)):
        print('identical')
    else:
        print('NOT identical')
        
#CV BGR
#PIL RGB
def swap_RB(img):
    img =  img[:,:,[2,1,0]]
        
#Generate pixel coordinates in the destination image         
def coord_mat(rows,cols):
    ind = np.arange(rows*cols)
    row_mat  = (ind//cols).reshape((rows,cols))
    col_mat  = (ind%cols).reshape((rows,cols))
    return row_mat, col_mat
        
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

def integral():
    I = np.ones((3,3))
    print(I)
    S = np.zeros((I.shape[0],I.shape[1]))
    S[0:,0:] = np.cumsum(I,axis =1)
    S = np.cumsum(S,axis = 0)
    print('----------------')
    print(S)
    return I, S

def sum_region(I,r0,c0,r1,c1):
    return np.sum(I[r0:r1+1,c0:c1+1])

def sum_region_integral(S,r0,c0,r1,c1):
    return S[r1+1,c1+1] - S[r1+1,c0] - S[r0,c1+1] + S[r0,c0]

def hor_grad(I):
    kernel = np.ones((1,2))
    kernel[0,0]=-1
    return cv2.filter2D(I,-1,kernel,borderType=cv2.BORDER_REPLICATE)

def ver_grad(I):    
    kernel = np.ones((2,1)) 
    kernel[0,0]=-1
    return cv2.filter2D(I,-1,kernel,borderType=cv2.BORDER_REPLICATE)

def grad_mag(vg,hg):    
    return np.sqrt(vg*vg+hg*hg)

def grad_angle(vg,hg): 
    t = np.arctan2(vg,hg)*180/math.pi
    t[t<0] = t[t<0] + 360
    return t

def hog(gm,ga,bars): 
    hist = np.zeros(bars)
    assigned_bar = (ga//(360/bars)+.5).astype(np.int)
    for b in range(bars):
        hist[b] = np.sum(gm[assigned_bar==b])

def hist_of_grad():
    I = np.random.randint(10, size=24).reshape((4,6)).astype(np.float)
    
    #calculate horizontal gradient
    hg = hor_grad(I)
    
    #calculate vertical gradient
    vg = ver_grad(I)
    
    print('hg', hg)
    print('==========================')
    print('vg', vg)
    
    #calculate magniture of gradients (square root)
    gm = grad_mag(vg,hg)
    
    #gettubg angle of gradients with some obscure formula
    ga = grad_angle(vg,hg)
    
    print(gm)
    print('==========================')
    print(ga)
    
    
    h = hog(gm,ga,8)
    print(h)
    print('sum', np.sum(h))
    print('sum2', np.sum(gm))

I, S = integral()

print(sum_region(I,0,0,2,2))
print(sum_region_integral(S,0,0,2,2))

hist_of_grad()


cv2.waitKey(0)
cv2.destroyAllWindows()