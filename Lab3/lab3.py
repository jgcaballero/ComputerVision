import numpy as np
import cv2
from PIL import Image
import time
import pylab as plt
plt.switch_backend('TKAgg')

#✔️
def real_value_indexing(img): #✔️
    
    r0 = 1
    r1 = 1
    c0 = 20
    c1 = 20
    
    counter = 0
    iterations = 0;
    
    img2 = np.array([[1,2,3],
                  [4,5,6], 
                  [7,8,9]])

    start = time.time()

    for y in range (r0, c0+1):
        for x in range (r1, c1+1):
            counter += img[y,x]
            iterations += 1
            print(img[y,x])  
    
    print('Counter', counter)
    print('iterations', iterations)
    print('Result', counter/iterations)
    
    elapsed_time = time.time()-start
    print('Elapsed time: {0:.2f} '.format(elapsed_time)) 
    
def triangle(img):    
    triangle2 = np.full_like(img, 0)
        
    row = img.shape[0]
    col = img.shape[1]
    slope = round(img.shape[1]/img.shape[0])
    
    for y in range (1,row):
        for x in range (1,col):
            if(x == col):
                triangle2[y,x] = img[y,x]
                continue
            else:
                triangle2[y,x] = img[y,x]
                img[y,x] = 0
        col -= slope

    return img, triangle2

def transform(H,fp):
    # Transforming point fp according to H
    # Convert to homogeneous coordinates if necessary
    if fp.shape[0]==2:
          t = np.dot(H,np.vstack((fp,np.ones(fp.shape[1]))))
    else:
        t = np.dot(H,fp)
    return t[:2]
  
def forward_mapping():
    img = cv2.imread('images/banner_small.jpg',1)
    triangle1, triangle2 = triangle(img)
    
    im1 = np.array(triangle1, dtype=np.uint8)
    plt.figure(3)
    plt.imshow(triangle1)
    plt.show()
    
    x2 = [0,0,im1.shape[0]-1]
    y2 = [0,im1.shape[1]-1,0]
    fp2 = np.vstack((x2,y2))
    
#    im1 = np.fliplr(im1)
#    im1 = np.flipud(im1)
    
    im2 = np.array(triangle2, dtype=np.uint8)
    plt.figure(1)
    plt.imshow(triangle2)
    plt.show()
        
    source_im = np.array(Image.open('images/tennis.jpg'), dtype=np.uint8)
    plt.figure(2)
    plt.imshow(source_im)
    plt.show()
    
    max_row = source_im.shape[0]-1
    max_col = source_im.shape[1]-1
    
    x = [0,0,im2.shape[0]-1]
    y = [0,im2.shape[1]-1,0]
    fp = np.vstack((x,y))
    
    #print("Click destination points, top-left, top-tight, and bottom-left corners")
    tp = np.asarray(plt.ginput(n=3), dtype=np.float).T
    tp = tp[[1,0],:]
    print('fp', fp)
    print('tp', tp)
    
    #print("Click destination points, top-right, bottorm-left, and bottom-right corners")
    tp2 = np.asarray(plt.ginput(n=3), dtype=np.float).T
    tp2 = tp2[[1,0],:]
    print('fp', fp2)
    print('tp', tp2)
    
    start = time.time()

    #Using pseudoinverse
    # Generating homogeneous coordinates
    fph = np.vstack((fp,np.ones(fp.shape[1])))
    tph = np.vstack((tp,np.ones(tp.shape[1])))
    H = np.dot(tph,np.linalg.pinv(fph))
    
    fph2 = np.vstack((fp2,np.ones(fp2.shape[1])))
    tph2 = np.vstack((tp2,np.ones(tp2.shape[1])))
    H2 = np.dot(tph2,np.linalg.pinv(fph2))
    
    print('wat', (transform(H,fp)+.5).astype(np.int))
    
    Cs = (transform(H,fp)+.5).astype(np.int)
    Cs[Cs<0] = 0
    Cs[0,Cs[0]>max_row] = max_row             
    Cs[1,Cs[1]>max_col] = max_col    
    
    Cs2 = (transform(H2,fp2)+.5).astype(np.int)
    Cs2[Cs2<0] = 0
    Cs2[0,Cs2[0]>max_row] = max_row             
    Cs2[1,Cs2[1]>max_col] = max_col    
        
    #Generating pixel coordinate locations
    ind = np.arange(im2.shape[0]*im2.shape[1])
    row_vect = ind//im2.shape[1]
    col_vect = ind%im2.shape[1]
    coords = np.vstack((row_vect,col_vect))
    
    new_coords = transform(H,coords).astype(np.int)
    new_coords[new_coords<0] = 255
    new_coords[0,new_coords[0]>max_row] = max_row             
    new_coords[1,new_coords[1]>max_col] = max_col    
    target_im = source_im
    
#    print('---------------------------')
#    for y in range(im2.shape[0]):
#        for x in range(im2.shape[1]):
#            if(np.dot(im2[y,x],im2[y,x]) == 0 ):
#                np.delete(im2, im2[y,x])
#    print('---------------------------')
#
#    print('---------------------------')
#    for y in range(im1.shape[0]):
#        for x in range(im1.shape[1]):
#            if(np.dot(im1[y,x],im1[y,x]) == 0 ):
#                np.delete(im1, im1[y,x])
#    print('---------------------------')
    
    
    target_im[new_coords[0],new_coords[1],:] = im2[coords[0],coords[1],:]
    
    #Generating pixel coordinate locations
    ind2 = np.arange(im1.shape[0]*im1.shape[1])
    row_vect2 = ind2//im2.shape[1]
    col_vect2 = ind2%im2.shape[1]
    coords2 = np.vstack((row_vect2,col_vect2))
    
    new_coords2 = transform(H2,coords2).astype(np.int)
    target_im = source_im
    target_im[new_coords2[0],new_coords2[1],:] = im1[coords2[0],coords2[1],:]
    
    elapsed_time = time.time()-start
    print('Elapsed time: {0:.2f} '.format(elapsed_time))  

    cv2.imshow('image',target_im)

    
    plt.figure(3)
    plt.imshow(target_im)
    plt.show()
    
def single_point_warp():
    
    source_im = np.array(Image.open('images/opencv.jpg'), dtype=np.uint8)
    plt.figure(1)
    plt.imshow(source_im)
    plt.show()
    
    print("Click source and destination of warp point")
    p = np.asarray(plt.ginput(n=2), dtype=np.float32)
    print(p)
    print(p[0]-p[1])
    plt.plot(p[:,0], p[:,1], color="blue")
    plt.plot(p[0][0], p[0][1],marker='x', markersize=3, color="red")
    plt.plot(p[1][0], p[1][1],marker='x', markersize=3, color="red")
    #plt.show()
    start = time.time()
    
    #Generate pixels coordinates in the destination image       
    dest_im = np.zeros(source_im.shape, dtype=np.uint8)                 
    max_row = source_im.shape[0]-1
    max_col = source_im.shape[1]-1
    dest_rows = dest_im.shape[0]
    dest_cols = dest_im.shape[1]
    
    #Painting outline of source image black, so out of bounds pixels can be painted black  
    source_im[0]=0
    source_im[max_row]=0         
    source_im[:,0]=0
    source_im[:,max_col]=0 
             
    #Generate pixel coordinates in the destination image         
    ind = np.arange(dest_rows*dest_cols )
    row_vect = ind//dest_cols
    col_vect = ind%dest_cols
    coords = np.vstack((row_vect,col_vect))
    
    #Computing pixel weights, pixels close to p[1] will have higher weights    
    dist = np.sqrt(np.square(p[1][1] - row_vect) + np.square(p[1][0] - col_vect))
    weight = np.exp(-dist/100)         #Constant needs to be tweaked depending on image size
    
    #Computing pixel weights, pixels close to p[1] will have higher weights    
    source_coords = np.zeros(coords.shape, dtype=np.int)
    disp_r = (weight*(p[0][1]-p[1][1])).astype(int)
    disp_c = (weight*(p[0][0]-p[1][0])).astype(int)
    source_coords[0] = coords[0] + disp_r
    source_coords[1] = coords[1] + disp_c
                 
    #Fixing out-of-bounds coordinates               
    source_coords[source_coords<0] = 0
    source_coords[0,source_coords[0]>max_row] = max_row             
    source_coords[1,source_coords[1]>max_col] = max_col      
          
    dest_im = source_im[source_coords[0],source_coords[1],:].reshape(dest_rows,dest_cols,3)
    
    cv2.imshow('image',dest_im)
    
    elapsed_time = time.time()-start
    print('Elapsed time: {0:.2f} '.format(elapsed_time))  
    
def multi_point_warp(dest_im):
    source_im = np.array(Image.open('images/opencv.jpg'), dtype=np.uint8)
    plt.figure(1)
    plt.imshow(source_im)
    plt.show()
    
    print("Click source and destination of warp point")
    p = np.asarray(plt.ginput(n=2, mouse_stop=2), dtype=np.float32)
    print(p)
    if(p.size == 0):
        #cv2.imshow('image',dest_im)
        return dest_im, False
        
    print(p[0]-p[1])
    plt.plot(p[:,0], p[:,1], color="blue")
    plt.plot(p[0][0], p[0][1],marker='x', markersize=3, color="red")
    plt.plot(p[1][0], p[1][1],marker='x', markersize=3, color="red")
    #plt.show()
    start = time.time()
    
    #Generate pixels coordinates in the destination image       
    dest_im = np.zeros(source_im.shape, dtype=np.uint8)                 
    max_row = source_im.shape[0]-1
    max_col = source_im.shape[1]-1
    dest_rows = dest_im.shape[0]
    dest_cols = dest_im.shape[1]
    
    #Painting outline of source image black, so out of bounds pixels can be painted black  
    source_im[0]=0
    source_im[max_row]=0         
    source_im[:,0]=0
    source_im[:,max_col]=0 
             
    #Generate pixel coordinates in the destination image         
    ind = np.arange(dest_rows*dest_cols )
    row_vect = ind//dest_cols
    col_vect = ind%dest_cols
    coords = np.vstack((row_vect,col_vect))
    
    #Computing pixel weights, pixels close to p[1] will have higher weights    
    dist = np.sqrt(np.square(p[1][1] - row_vect) + np.square(p[1][0] - col_vect))
    weight = np.exp(-dist/100)         #Constant needs to be tweaked depending on image size
    
    #Computing pixel weights, pixels close to p[1] will have higher weights    
    source_coords = np.zeros(coords.shape, dtype=np.int)
    disp_r = (weight*(p[0][1]-p[1][1])).astype(int)
    disp_c = (weight*(p[0][0]-p[1][0])).astype(int)
    source_coords[0] = coords[0] + disp_r
    source_coords[1] = coords[1] + disp_c
                 
    #Fixing out-of-bounds coordinates               
    source_coords[source_coords<0] = 0
    source_coords[0,source_coords[0]>max_row] = max_row             
    source_coords[1,source_coords[1]>max_col] = max_col      
          
    dest_im = source_im[source_coords[0],source_coords[1],:].reshape(dest_rows,dest_cols,3)
    
#    cv2.imshow('image',dest_im)
    
    elapsed_time = time.time()-start
    print('Elapsed time: {0:.2f} '.format(elapsed_time))  
    
    return dest_im, True


'''
1) real value index
'''
#img = cv2.imread('images/cat.jpg',0)
#source_im = np.array(Image.open('images/opencv.jpg'), dtype=np.uint8)
#real_value_indexing(img)
    
'''
2) Forward Mapping 
'''
#forward_mapping()
    
'''
4) Single Point Warp
'''
#while True:
#    single_point_warp()


'''
5) Multi Point Warp
'''
dest_im = np.array(Image.open('images/opencv.jpg'), dtype=np.uint8)
isTrue = True
arr = []
while isTrue:
#    for x in range(arr.shape[0]):
    dest_im, isTrue = multi_point_warp(dest_im)
    arr.append(dest_im)
    if(not isTrue):
        for y in range(len(arr)):
            image = "image"+str(y)
            print('iteration #', y)
            print('iteration #', arr[y])
            cv2.imshow('image',arr[y])
            cv2.waitKey(3000)
#            plt.pause(.1)
#            plt.draw()

    
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    for y in range(arr.shape[0]):
        print('iteration #', y)
        cv2.imshow('image',arr[y])
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit0
   cv2.imshow('image',dest_im)
   cv2.destroyAllWindows()
   