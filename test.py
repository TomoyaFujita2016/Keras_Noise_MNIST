import keras_mnist as km
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import Callback, CSVLogger
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
from PIL import Image
import numpy as np
from keras.preprocessing import image
import math
from keras import backend as K
import pickle as pkl

def data_augmentation(x, y):
    datagen = image.ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=math.pi/4, # 45 degree
        zoom_range=0.4,
        fill_mode="constant",
        cval=0, # constant value for fill_mode
        )
    imgs = np.array([x.tolist()[1]]*9)
    show_img(imgs, "img1")
    for X_batch, y_batch in datagen.flow(imgs, y[:9], batch_size=9):
        show_img(X_batch, "img2")
        break    

def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.1)
    x_train = x_train1[:9000]
    y_train = y_train1[:9000]
    x_valid = x_valid[:1000]
    y_valid = y_valid[:1000]

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
    x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1).astype('float32')/255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255

    # convert one-hot vector
    y_train = keras.utils.to_categorical(y_train, 10)
    y_valid = keras.utils.to_categorical(y_valid, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    data_augmentation(x_train, y_train)

def show_noisedata():
    with open("./mnist_noise.pkl", "rb") as f:
        x_train, y_train, x_valid, y_valid = pkl.load(f)
    
    for i in range(0, 9):
        plt.subplot(330+1+i)
        plt.imshow(x_valid[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    show_y(y_valid[:9])
    plt.show()

def show_img(imgs, name):
    for i in range(0, 9):
        plt.subplot(330+1+i)
        plt.imshow(imgs[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    plt.savefig("{}.png".format(name))

def show_y(y_valid9):
    for n, y_row in enumerate(y_valid9):
        for i, y in enumerate(y_row):
            if y == 1:
                print(i, " ", end="")
                break
        if n != 0 and (n+1) % 3 == 0:
            print("")

if __name__=="__main__":
    #km.main()
    #get_mnist()
    show_noisedata()
