import os
from urllib.request import urlretrieve
import numpy as np
import time

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA


def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

# We then define functions for loading MNIST images and labels.
# For convenience, they also download the requested files if needed.
import gzip

def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data


X_train = load_mnist_images('train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

train_ex = 1000
test_ex = 2000

X_train = X_train.reshape((X_train.shape[0], -1))[:train_ex]
X_test = X_test.reshape((X_test.shape[0], -1))[:test_ex]
y_train = y_train[:train_ex]
y_test = y_test[:test_ex]

#scaler = StandardScaler()  
#scaler.fit(X_train)
#
#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)  


pca = PCA(n_components=70, svd_solver='full')
pca.fit(X_train)
X_train = pca.transform(X_train)  
X_test = pca.transform(X_test) 

start = time.time()

MLP = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, batch_size='auto',tol=0.000000001)
MLP.fit(X_train, y_train)
MLPredictions = MLP.predict(X_test)
MLPAccuracy = np.sum(MLPredictions == y_test)/y_test.shape[0]
print('MLP accuracy : ' , MLPAccuracy)
print(classification_report(y_test,MLPredictions))

elapsed_time = time.time()-start
print('Elapsed time: {0:.2f} '.format(elapsed_time)) 

#start = time.time()
#
#knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto')
#knn.fit(X_train, y_train)
#KnnPredictions = knn.predict(X_test)
#KnnAccuracy = np.sum(KnnPredictions == y_test)/y_test.shape[0]
#print('KNN accuracy : ' , KnnAccuracy)
#
#elapsed_time = time.time()-start
#print('Elapsed time: {0:.2f} '.format(elapsed_time)) 

