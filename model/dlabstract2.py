import abc
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.models as model_from_json


class dlabstract2(metaclass=abc.ABCMeta):

    # create abstract method
    @abc.abstractmethod
    def create_model(self):
        return None

    # load model
    def load_model(self):
        # Read the learning model
        model = None

        with open(self.modelFile, 'r') as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)

        # Read the weight
        model.load_weights(self.weightFile)
        return model

    # Get model
    def getModel(self):
        if self.modelFile and self.weightFile:
            self.model = self.load_model()
        else:
            self.model = self.create_model()
        return self.model

    # Get model filename ,weight filename, shape, the number of category
    def __init__(self, model=None, weight=None, shape=None, numCategory=0):
        self.modelFile = model
        self.weightFile = weight
        self.shape = shape
        self.numCategory = numCategory

    # Start the model on TPU or CPU
    def start(self, X_train, Y_train, X_test, Y_test, batch_size, epochs, summary=True, verbose=1, steps_per_epoch=18):
        if os.environ.get("COLAB_TPU_ADDR"):
            return self.TPUstart(X_train, Y_train, X_test, Y_test, batch_size, epochs, summary, verbose, steps_per_epoch)
        else:
            return self.CPUstart(X_train, Y_train, X_test, Y_test, batch_size, epochs, summary, verbose, steps_per_epoch)

    # start the model on CPU
    @abc.abstractmethod
    def CPUstart(self, X_train, Y_train, X_test, Y_test, batch_size, epochs, summary=True, verbose=1, steps_per_epoch=18):
        return None

    # start the model on TPU
    @abc.abstractmethod
    def TPUstart(self, X_train, Y_train, X_test, Y_test, batch_size, epochs, summary=True, verbose=1, steps_per_epoch=18):
        return None

    # predict
    def predict(self, X_data, batch_size=5, verbose=1):
        model = self.getModel()
        r = model.predict(np.array([X_data]), batch_size, verbose)
        res = r[0]

        # Output the result
        for i, acc in enumerate(res):
            print(i, '=', int(acc * 100))
        return {np.argmax(res): np.max(r)}

    # Save the model data and weight
    def save_model(self, model, weight):
        tf.keras.models.save_model(self.model, model, save_format='h5')
        self.model.save_weights(weight)
