import numpy as np
import cv2
from matplotlib import pyplot as plt

# ✔️  Cat fix using a box filter approach using a kernel of 3 by 3 and averaging the pixels
def fix_cat():
    cat = cv2.imread('images/cat.jpg',0)
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(cat,-1,kernel)
    res = np.hstack((cat,dst))
    cv2.imshow('result',res) 


#Cheetah ✔️
def fix_cheetah():
    cheetah = cv2.imread('images/cheetah.jpg', 0)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    dst = cv2.filter2D(cheetah,-1,kernel)
    res = np.hstack((cheetah,dst))
    cv2.imshow('result',res) 
    
#City
def fix_city():
    city = cv2.imread('images/city.jpg')
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    dst = cv2.filter2D(city,-1,kernel)
    res = np.hstack((city,dst))
    cv2.imshow('result',res) 
    
#Deer
def fix_deer():
    deer = cv2.imread('images/deer.jpg', 0)
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(deer,-1,kernel)
    #blur = cv2.medianBlur(equ,5)
    res = np.hstack((deer,dst))

    cv2.imshow('result',res) 
    
#Dog ✔️ 
def fix_dog():
    dog = cv2.imread('images/dog.jpg') #histogram equalization
    color = cv2.cvtColor(dog, cv2.COLOR_BGR2YUV)
    color[:,:,0] = cv2.equalizeHist(color[:,:,0])
    img_output = cv2.cvtColor(color, cv2.COLOR_YUV2BGR)
    gauss_blur2 = np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4],[1,4,6,4,1]])/256
    dst = cv2.filter2D(img_output,-1,gauss_blur2)
    res = np.hstack((dog,dst))
    cv2.imshow('result',res) 

#Husky ✔️ 
def fix_husky():
    husky = cv2.imread('images/husky.jpg') #histogram equalization
    color = cv2.cvtColor(husky, cv2.COLOR_BGR2YUV)
    color[:,:,0] = cv2.equalizeHist(color[:,:,0])
    img_output = cv2.cvtColor(color, cv2.COLOR_YUV2BGR)
    #gauss_blur = np.array([[1,2,1], [2,4,2], [1,2,1]])/16
    #gauss_blur2 = np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4],[1,4,6,4,1]])/256
    dst = cv2.filter2D(img_output,-1,img_output)
    res = np.hstack((husky,dst))
    cv2.imshow('result',res) 

#leopard✔️
def fix_leopard():
    leopard = cv2.imread('images/leopard.jpg', 0)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    dst = cv2.filter2D(leopard,-1,kernel)
    res = np.hstack((leopard,dst))
    cv2.imshow('result',res) 
    
#ny✔️
def fix_ny():
    ny = cv2.imread('images/ny.jpg')
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    dst = cv2.filter2D(ny,-1,kernel)
    res = np.hstack((ny,dst))
    cv2.imshow('result',res) 
    
#rose ✔️
def fix_rose():
    rose = cv2.imread('images/rose.jpg') #median blur to fix salt and pepper
    color = cv2.cvtColor(rose, cv2.COLOR_BGR2YUV)
    color[:,:,0] = median_filter(color[:,:,0])
    img_output = cv2.cvtColor(color, cv2.COLOR_YUV2BGR)
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(img_output,-1,kernel)
    #med = cv2.medianBlur(rose,5)
    res = np.hstack((rose,dst))
    cv2.imshow('result',res) 
    
def median_filter(img):
    arr = np.zeros(9)
    result = np.zeros(img.shape, img.dtype)

    for y in range(1,img.shape[0]-1):
        for x in range(1,img.shape[1]-1):
            arr[0] = img[y-1,x-1]
            arr[1] = img[y,x-1]
            arr[2] = img[y+1,x-1]
            arr[3] = img[y-1,x]
            arr[4] = img[y,x]
            arr[5] = img[y+1,x]
            arr[6] = img[y-1,x+1]
            arr[7] = img[y,x+1]
            arr[8] = img[y+1,x+1]

            arr.sort()
            result[y,x]=arr[4]
            
    return result
            

#tricycle
def fix_tricycle():
    tricycle = cv2.imread('images/tricycle.jpg')
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    dst = cv2.filter2D(tricycle,-1,kernel)
    res = np.hstack((tricycle,dst))
    cv2.imshow('result',res) 
    
rose = cv2.imread('images/rose.jpg') #median blur to fix salt and pepper
fix_rose()



k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    #cv2.imwrite('images/messigray.png',img)
    cv2.destroyAllWindows()