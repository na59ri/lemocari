import abc
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from model.dlabstract2 import dlabstract2


class vgg8(dlabstract2):

    # create model
    def create_model(self):

        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation='relu',
                         padding='same', input_shape=self.shape.shape[1:]))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.numCategory, activation='softmax'))

        return model

    # start model on CPU
    def CPUstart(self, X_train, Y_train, X_test, Y_test, batch_size, epochs, summary=True, verbose=1, steps_per_epoch=18):
        self.model = self.getModel()

        if (summary):
            self.model.summary()

        # Compile model
        self.model.compile(
            loss='categorical_crossentropy',  # Specify categorical_crossentropy
            optimizer='RMSProp',              # Optimization to use RMSProp
            metrics=['accuracy'])

        # Run learning
        # fit() : Learning the model to try the fixed number
        # epochs : The repeat number
        # batch_size ： the data size on 1 circle
        # validation_data
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=(X_test, Y_test), verbose=verbose, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard])

        # Validate the model
        # evaluate() : Return the loss of model and the validation value
        # verbose : Output the raw of the validation.
        score = self.model.evaluate(X_test, Y_test, verbose=1)
        print('Accuracy=', score[1], 'loss=', score[0])

        return score

    # start model on TPU
    def TPUstart(self, X_train, Y_train, X_test, Y_test, batch_size, epochs, summary=True, verbose=1, steps_per_epoch=18):

        # The rule of Colab TPU ( tpu_grpc_url - strategy.scope() )
        tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_grpc_url)
        tf.contrib.distribute.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.contrib.distribute.TPUStrategy(
            tpu_cluster_resolver, steps_per_run=100)

        with strategy.scope():

            # After strategy.scope(), create and start the model
            return self.CPUstart(X_train, Y_train, X_test, Y_test, batch_size, epochs, summary, verbose, steps_per_epoch)

    def __init__(self, model, weight, shape, numCategory):
        super().__init__(model, weight, shape, numCategory)
