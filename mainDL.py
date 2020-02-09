import os
import re
import time
import datetime

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from model.vgg8 import vgg8
from model.vgg16 import vgg16
from model.vgg16Batch import vgg16Batch
from model.likevgg2 import likevgg2
from model.simpleCNN import simpleCNN


# input path : ./Input/<CATEGORY>/<image.jpg>
inputPath = "./Input/"

# output path : ./Output/<MODEL>/<model_accurary_date.h5>
outputPath = "./Output/"

# target size
twidth = 256
theight = 256

# batch size
BATCHSIZE = 128

# epoch size
EPOCH = 100

# the number of category
num_category = 0
X = []
Y = []


# Add model (./model/<Model>)
def DLmodel(model=None, weight=None, shape=None, numCategory=0):

    model = []

    model.append(vgg8(model, weight, shape, numCategory))

    return model


# Get image list in direcotry
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

# Get the input data of image and category, and the number of category


def getData():

    # Get directory of category
    dirList = [dirItem for dirItem in os.listdir(
        inputPath) if os.path.isdir(os.path.join(inputPath, dirItem))]

    # Get the number of category
    global num_category
    num_category = len(dirList)

    # Get the input data of image and category
    global X, Y
    for item in dirList:

        # Get the picture in category directory
        for picture in list_pictures(os.path.join(inputPath, item)):
            img = img_to_array(
                load_img(picture, target_size=(twidth, theight)))
            X.append(img)
            Y.append(int(item))

    # convert array to np.asarray
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Convert integer value to float value(0 - 1)
    X = X.astype('float32') / 255.0

    # Convert category
    Y = to_categorical(Y, num_category)


if __name__ == "__main__":

    # Get image data
    getData()

    # devide the data to traning/test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Get model list
    modelList = DLmodel(shape=X_train, numCategory=num_category)

    # Get steps per epoch to use deep larning.
    spe = int(X_train.shape[0] / BATCHSIZE)

    # Create model and save model data
    for item in modelList:

        # start to create model
        score = item.start(X_train, Y_train, X_test, Y_test,
                           BATCHSIZE, EPOCH, steps_per_epoch=spe)

        # create file path and file name
        today = datetime.datetime.fromtimestamp(time.time())
        todayTime = today.strftime('%Y%m%d_%H%M%S')

        filename = item.__class__.__name__ + \
            "_" + str(score[1]) + "_" + todayTime
        savePath = os.path.join(outputPath, item.__class__.__name__)
        if(not os.path.isdir(savePath)):
            os.makedirs(savePath)

        # save model data
        item.save_model(os.path.join(savePath, filename+"_model.h5"),
                        os.path.join(savePath, filename+"_weight.h5"))
