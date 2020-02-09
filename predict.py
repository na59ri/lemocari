# -*- coding: utf-8 -*-

# Identify the lemon quality to use model data.

import os
import cv2
import re
import shutil
import time
import numpy as np
import tensorflow as tf
import json
import urllib.request

LABELS = ['A', 'B', 'C', 'D']

# Image size
WIDTH = 256
HEIGHT = 256
CHANNEL = 3

IMAGESIZE = WIDTH * HEIGHT * CHANNEL

# input path
INPUT = "/home/pi/LemonBot/Input/."

# output path
OUTPUT = "/home/pi/LemonBot/Output4/."

# Model data path
MODELDATA = '/home/pi/LemonBot/AI/lemon_model.h5'

# Model weight
MODELWIEGHT = '/home/pi/LemonBot/AI/lemon_weight.h5'

# Node-RED URL to display the output
url = 'http://localhost:1880/lebotResult'

# Send the header to URL
headers = {
    'Content-Type': 'application/json',
}


# Get image list in direcotry
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + ')$)', f.lower())]


# predict the lemon quality
def prediction():

    # Get model data
    loaded_model = tf.keras.models.load_model(MODELDATA)

    # Get model weight
    loaded_model.load_weights(MODELWIEGHT)

    if (loaded_model == None):
        print("error : model is none")
        break

    # prediction loop
    while (True):

        # Get image data on Input directory
        inputList = list_pictures(INPUT)

        # continue loop if not image data
        if (len(inputList) < 1):
            continue

        # There is the image data
        X = []
        counter = 0
        for image in inputList:

            # break "for" loop if more than 2 files
            if(1 <= counter):
                break

            # count up
            counter = counter + 1

            print("filename:"+image)

            # Read image
            im = cv2.imread(image)

            # Resize and Convert BGR to RGB
            im = cv2.resize(im, (WIDTH, HEIGHT))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # Append the image to array
            X.append(im)

        # Convert array to np.asarray
        X = np.asarray(X)

        # Convert integer value to float value(Range from 0.0 to 1.0)
        X = X.astype('float32') / 255.0

        # Predict
        start = time.time()
        result = loaded_model.predict(X)
        end = time.time()
        proTime = end - start

        # display the predict time
        print("predict time:"+str(proTime))

        for i in range(len(X)):
            # Get filename of input data
            basefile = os.path.basename(inputList[i])

            # display the each category of result
            for ix, acc in enumerate(result[i]):
                print("     "+LABELS[ix]+":"+str(acc))

            # Identify the lemon quality
            print("predict:::" + basefile + " is " +
                  LABELS[result[i].argmax()])

            # Convert the input fileanme to output filename
            outputpath = os.path.join(OUTPUT, str(result[i].argmax()))
            if (not os.path.isdir(outputpath)):
                os.makedirs(outputpath)

            # Get abslute path
            abspath = os.path.abspath(os.path.join(outputpath, basefile))

            # move the abspath
            shutil.move(inputList[i], abspath)

            # Get the size to file
            pattern = r'(\d+)+g.*\.[\w]+$'
            item = re.search(pattern, basefile)
            if item:

                # Set body
                data = {'filepath': abspath,
                        'grade': LABELS[result[i].argmax()], "weight": item.group(1)}

                # send Node-RED to URL
                try:
                    req = urllib.request.Request(
                        url, json.dumps(data).encode(), headers)
                    with urllib.request.urlopen(req) as res:
                        body = res.read()
                except urllib.error.HTTPError as err:
                    print(err.code)
                except urllib.error.URLError as err:
                    print(err.reason)

                time.sleep(5)


if __name__ == "__main__":
    prediction()
