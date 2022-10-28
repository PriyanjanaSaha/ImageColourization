import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#imports
import keras
from keras.preprocessing import image
from keras.engine import Layer
from keras.layers import Conv2D, Conv3D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from PIL import Image, ImageFile
from zipfile import ZipFile
from io import BytesIO,StringIO
import base64
proto = '/kaggle/input/monochrome-image-dataset/colorization_deploy_v2.prototxt.txt'
weights = '/kaggle/input/monochrome-image-dataset/colorization_release_v2_norebal.caffemodel'

# load cluster centers
pts_in_hull = np.load('/kaggle/input/monochrome-image-dataset/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
# load model
net = cv2.dnn.readNetFromCaffe(proto, weights)
net.getLayerNames()
# populate cluster centers as 1x1 convolution kernel
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
# scale layer doesn't work in OpenCV dnn module, we need to fill 2.606 to conv8_313_rh layer manually
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]
def get_images(path):
    a = []
    b = []
    c = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
        print(type(img))
        if(img is None):
            b.append(img)
        else:
            img_input = img.copy()
            img = cv2.resize(img,(600,600))

            # convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_rgb = img.copy()
            img_rgb = (img_rgb / 255.).astype(np.float32)

            # convert RGB to LAB
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
            # only L channel to be used
            img_L = img_lab[:,:,0]

            input_img = cv2.resize(img_L, (224, 224))
            input_img -= 50 # subtract 50 for mean-centering
            c.append(img_L)
            a.append(input_img)
                   
    return a,c


def predict(a,c):
    p = []
    for i in range(len(a)):
        net.setInput(cv2.dnn.blobFromImage(a[i]))
        pred = net.forward()[0,:,:,:].transpose((1, 2, 0))

        # resize to original image shape
        pred_resize = cv2.resize(pred, (600, 600))
        img_L = c[i]

        # concatenate with original image L
        pred_lab = np.concatenate([img_L[:, :, np.newaxis], pred_resize], axis=2)

        # convert LAB to RGB
        pred_rgb = cv2.cvtColor(pred_lab, cv2.COLOR_Lab2RGB)
        pred_rgb = np.clip(pred_rgb, 0, 1) * 255
        pred_rgb = pred_rgb.astype(np.uint8)
        p.append(pred_rgb)
    
    return p
    path = '/kaggle/input/monochrome-image-dataset/'
X,y = get_images(path)
plt.figure(figsize=(10,10))
for i in range(len(X)):
    plt.subplot(3,7,i+1)
    plt.imshow(X[i],cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
P = predict(X,y)
plt.figure(figsize=(10,10))
for i in range(len(P)):
    plt.subplot(3,7,i+1)
    plt.imshow(P[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
# save result image file
for i in range(len(P)):
    img = Image.fromarray(P[i], 'RGB')
    img.save('output'+str(i)+'.png')
    !zip -m images.zip output*.png
    
