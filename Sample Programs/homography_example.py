import numpy as np
from PIL import Image
import pylab as plt
plt.switch_backend('TKAgg')


def transform(H,fp):
    # Transforming point fp according to H
    # Convert to homogeneous coordinates if necessary
    if fp.shape[0]==2:
          t = np.dot(H,np.vstack((fp,np.ones(fp.shape[1]))))
    else:
        t = np.dot(H,fp)
    return t[:2]
    
im2 = np.array(Image.open('banner_small.jpg'), dtype=np.uint8)
plt.figure(1)
plt.imshow(im2)
plt.show()

source_im = np.array(Image.open('tennis.jpg'), dtype=np.uint8)
plt.figure(2)
plt.imshow(source_im)
plt.show()

x = [0,0,im2.shape[0]-1]
y = [0,im2.shape[1]-1,0]
fp = np.vstack((x,y))

print("Click destination points, top-left, top-tight, and bottom-left corners")
tp = np.asarray(plt.ginput(n=3), dtype=np.float).T
tp = tp[[1,0],:]
print(fp)
print(tp)

#Using pseudoinverse
# Generating homogeneous coordinates
fph = np.vstack((fp,np.ones(fp.shape[1])))
tph = np.vstack((tp,np.ones(tp.shape[1])))
H = np.dot(tph,np.linalg.pinv(fph))

print((transform(H,fp)+.5).astype(np.int))

#Generating pixel coordinate locations
ind = np.arange(im2.shape[0]*im2.shape[1])
row_vect = ind//im2.shape[1]
col_vect = ind%im2.shape[1]
coords = np.vstack((row_vect,col_vect))

new_coords = transform(H,coords).astype(np.int)
target_im = source_im
target_im[new_coords[0],new_coords[1],:] = im2[coords[0],coords[1],:]

plt.figure(3)
plt.imshow(target_im)
plt.show()

