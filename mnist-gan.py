import keras as K
import numpy as np
from keras.datasets import mnist
from keras import layers
from matplotlib import pyplot

class MNISTDiscriminator(object):
    ## Takes a 28x28 image, gives a binary classification (real or fake)
    def create_discriminator(self, data, input_shape=(28, 28, 1)):
        discrim_layers = [
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                          input_shape=input_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.4),

            layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.4),

            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ]

        discriminator = K.models.Sequential(discrim_layers)
        model.compile(loss='binary_crossentropy',
                      optimizer=K.optimizers.Adam(lr=0.002),
                      metrics=['accuracy'])

        return discriminator


    def __init__(self):
        self.model = self.create_discriminator()


    def create_real_samples(self, data, n):
        x = data[np.random.randint(0, data.shape[0], n)]
        y = np.ones((n, 1))

        return x, y


    def create_fake_samples(self, n):
        x = np.random.rand(28 * 28 * n).reshape((n, 28, 28, 1))
        y = np.zeros((n, 1))

        return x, y


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.expand_dims(x_train, axis=-1).astype('float32')
    x = x / 255.0
