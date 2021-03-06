import numpy as np
import cv2
import time

# ✔️  Cat fix using a box filter approach using a kernel of 3 by 3 and averaging the pixels
def fix_cat():
    cat = cv2.imread('images/cat.jpg')
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
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
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
    #dst = cv2.filter2D(img_output,-1,img_output)
    res = np.hstack((husky,img_output))
    cv2.imshow('result',res) 

#leopard✔️
def fix_leopard():
    leopard = cv2.imread('images/leopard.jpg')
    color = cv2.cvtColor(leopard, cv2.COLOR_BGR2YUV)
    color[0,:,:] = cv2.equalizeHist(color[0,:,:])
    img_output = cv2.cvtColor(color, cv2.COLOR_YUV2BGR)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    dst = cv2.filter2D(img_output,-1,kernel)
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
    
def increase_image():
    cat = cv2.imread('images/cat.jpg')
    cat_bw = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)/255.0
    inc = increase_img(cat_bw)
    result = inc_interpol(inc)
    result2 = inc_cols(result)
    print(result2)
    cv2.imshow("resize",result2)
    
def increase_img(img):
    col = len(img)*2
    row = len(img[0])*2
    og = len(img[0])   
    result = np.zeros(shape=(col,row))
    
    for y in range(0,img.shape[0]):        
        for x in range(0,img.shape[1]):
            if(x == og-1):
                result[y*2,row-1] = img[y,x]
            else:
                #print(img[y,x])
                #arr[0] = img[y,x]
                #arr[1] = img[y,x+1]
                result[y*2,x*2]=img[y,x]
                
    #(result)
    return result
    
def inc_interpol(img):
    #print(img)
    arr = np.zeros(3)
    length = len(img[0])

    
    for y in range(1,img.shape[0],-1): 
        for x in range(1,img.shape[1],-1):
            if(x != length-1):
                arr[0] = img[y,x]
                arr[1] = img[y,x+2]
                #arr[2] = img[y+2,x]
                resX = apply_linear_interpolation(x+1,x,arr[0],x+2,arr[1])
                #resY = apply_linear_interpolation(x,x,arr[0],x,arr[2])
                #print(resY)
                img[y,x+1] = resX
                #[y+1,x] = resY
                #print(arr)
            else:
                pass
                #print('ignore last val!')
                
    #print(img)
    return img


def inc_cols(img):
    #print('fuuuuuuuuuuuuuuuu')
    #print(img)
    arr = np.zeros(2)
    length = len(img[0])
    
    for y in range(0,img.shape[0],2): 
        for x in range(0,img.shape[1]):
            if(y != length-1):
                #print(y,x)
                arr[0] = img[y,x]
                arr[1] = img[y+1,x]
                resY = apply_linear_interpolation(x,x,arr[0],x,arr[1])
                #print('RESY IS : ' , resY)
                img[y+1,x] = resY
                #print(arr)
            else:
                pass
                #print('ignore last val!')
            
    return img
    #print(img)
                

            

            
def apply_linear_interpolation (x, x1, y1, x2, y2):
    #print(x, x1, y1, x2, y2)
    isZero = x2 - x1
    if(isZero == 0 ):
        #print('ZEROOOOOOOOOOOOOOOOOOOOOES')
        y = (y1 + y2)//2
       # print(y)
    else:
        #print('YYYYYYYY THOOOO')
        y = y1 + (x - x1)*((y2-y1)//(x2-x1))
    
    return y

    
def decrease_image():
    cat = cv2.imread('images/cat.jpg',0)
    dec = decrease_img(cat, 2)
    cv2.imshow('result',dec)

def decrease_img(img_to_resize, box_size):
    img = img_to_resize/255
    arr = np.zeros(box_size*box_size)
    col = len(img)//box_size
    row = len(img[0])//box_size
    result = np.zeros(shape=(col,row))

    for y in range(0,img.shape[0],box_size):        
        for x in range(0,img.shape[1],box_size):          
           # print(y , x)
           # print(y , x+1)
           # print(y+1 , x)
           # print(y+1 , x+1)
            arr[0] = img[y,x]
            arr[1] = img[y,x+1]
            arr[2] = img[y+1,x]
            arr[3] = img[y+1,x+1]
            #print(y//box_size,x//box_size)
            result[y//2,x//2]=sum(arr)/(box_size*box_size)
            
    return result

#Uncomment the following lines in order to trigger one of the fixes.
start = time.time()
count=0

fix_cat()
#fix_cheetah()
#fix_city()
#fix_deer()
#fix_dog()
#fix_husky()
#fix_leopard()
#fix_ny()
#fix_rose()
#fix_tricycle()

#increase_image()

# Making the picture smaaaaaalleeer!
#decrease_image()
elapsed_time = time.time()-start
print(elapsed_time)


k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    #cv2.imwrite('images/messigray.png',img)
    cv2.destroyAllWindows()