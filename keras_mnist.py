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

mx_vl = 0
tl = "gaussian_training_noise_validation_nonoise_relu_"

def relu_advanced(x):
    return K.relu(x, max_value=0)

def plot_result(history):
    # accuracy
    plt.figure()
    plt.plot(history.history['acc'], label='acc', marker='.')
    plt.plot(history.history['val_acc'], label='val_acc', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('accuracy')
    plt.savefig('./figs/graph_acc_{}{}.png'.format(tl, mx_vl))
    #plt.show()

    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='loss', marker='.')
    plt.plot(history.history['val_loss'], label='val_loss', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('loss')
    plt.savefig('./figs/graph_loss_{}{}.png'.format(tl, mx_vl))
    #plt.show()


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    #model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation=relu_advanced))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model

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
    print("imgs: ", x.shape)
    ad_img = [] 
    ad_label = [] 
    # configure batch size and retrieve one batch of images
    bs = 100
    img_rate = 5
    max_loop = x.shape[0]*img_rate/bs
    cnt_loop = 0
    for X_batch, y_batch in datagen.flow(x, y, batch_size=bs):
        if cnt_loop % 100 == 0:
            print("[{0}/{1}]".format(cnt_loop, max_loop))
        if cnt_loop == max_loop:
            break
        cnt_loop += 1
        ad_img.extend(X_batch.tolist())
        ad_label.extend(y_batch.tolist())
    print("added_imgs: ", len(ad_img))
    return np.array(ad_img), np.array(ad_label)

def make_gaussian_noise_data(data_x, scale=0.1):
    gaussian_data_x = data_x + np.random.normal(loc=0, scale=scale, size=data_x.shape)
    gaussian_data_x = np.clip(gaussian_data_x, 0, 1)
    return gaussian_data_x

def show_img(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def train_model(epochs=5, batch_size=128, mknoise=False):
    # load MNIST data
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

    # make noise data
    if mknoise:
        x_train, y_train = data_augmentation(x_train, y_train)
        x_valid, y_valid = data_augmentation(x_valid, y_valid)
        x_train = make_gaussian_noise_data(x_train)
        x_valid = make_gaussian_noise_data(x_valid)
        with open("./mnist_noise.pkl", "wb") as f:
            pkl.dump((x_train, y_train, x_valid, y_valid), f)
        import sys
        sys.exit()
    
    with open("./mnist_noise.pkl", "rb") as f:
        x_train, y_train, _, _ = pkl.load(f)
        #x_train, y_train, x_valid, y_valid = pkl.load(f)

    model = create_model()
    print(model.summary())
    
    # callback function
    csv_logger = CSVLogger('trainlog.csv')

    # train
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs,
                        verbose=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[csv_logger])

    # result
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {0}'.format(score[0]))
    print('Test accuracy: {0}'.format(score[1]))
    
    model.save_weights('./weights/{0}{1}.weights'.format(tl, mx_vl))
    plot_result(history)


def main():
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--epochs', dest='epochs', type=int, help='size of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='size of batch')
    parser.add_argument('--max_value', dest='max_value', type=float, help='size of relu_max_value')
    parser.add_argument('--mknoise', action='store_true', help='make noise data')
    args = parser.parse_args()
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 5
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = 128
    if args.max_value:
        mx_vl = args.max_value
    else:
        mx_vl = 0

    train_model(epochs, batch_size, mknoise=args.mknoise)

if __name__ == '__main__':
    main()


