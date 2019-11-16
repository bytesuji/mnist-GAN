import keras as K
import numpy as np
from keras.datasets import mnist
from keras import layers
from matplotlib import pyplot

class MNISTDiscriminator(object):
    ## Takes a 28x28 image, gives a binary classification (real or fake)

    def __init__(self, input_shape=(28, 28, 1)):
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

        self.model = K.models.Sequential(discrim_layers)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=K.optimizers.Adam(lr=0.002, beta_1=0.5),
                           metrics=['accuracy'])

    def create_real_samples(self, data, n):
        x = data[np.random.randint(0, data.shape[0], n)]
        y = np.ones((n, 1))

        return x, y

    def create_fake_samples(self, n):
        x = np.random.rand(28 * 28 * n).reshape((n, 28, 28, 1))
        y = np.zeros((n, 1))

        return x, y

    def train(self, data, epochs=128, batch_size=256):
        print("--- TRAIN DISCRIMINATOR ---")
        for i in range(epochs):
            x_real, y_real = self.create_real_samples(data, int(batch_size / 2))
            x_fake, y_fake = self.create_fake_samples(int(batch_size / 2))

            real_acc = self.model.train_on_batch(x_real, y_real)[1]
            fake_acc = self.model.train_on_batch(x_fake, y_fake)[1]
            avg_acc = (real_acc + fake_acc) / 2

            print("epoch {} accuracy {}%".format(i, avg_acc*100))

    def summary(self): self.model.summary()


class MNISTGenerator(object):
    def __init__(self, latent_space_dim=100, init_dim=(7, 7, 128)):
        self.latent_space_dim = latent_space_dim
        init_dim_m = init_dim[0] * init_dim[1] * init_dim[2]

        gen_layers = [
            layers.Dense(init_dim_m, input_dim=latent_space_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape(init_dim),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same')
        ]

        self.model = K.models.Sequential(gen_layers)

    def generate_latent_elements(self, n):
        return np.random.randn(self.latent_space_dim * n).reshape(n, latent_space_dim)

    def generate(self, n):
        latent = self.generate_latent_elements(n)
        x = self.model.predict(latent)
        y = np.zeros((n, 1)) ## speed up backprop by claiming gen'd images are real

        return x, y

    def summary(self): self.model.summary()


class MNISTGAN(object):
    def __init__(self):
        self.discriminator = MNISTDiscriminator()
        self.discriminator.model.trainable = False
        self.generator = MNISTGenerator()

        self.model = K.models.Sequential()
        self.model.add(self.discriminator.model)
        self.model.add(self.generator.model)

        self.model.compile(loss='binary_crossentropy',
                         optimizer=K.optimizers.Adam(lr=0.0002, beta_1=0.5))

    def train(self, epochs=100, batch_size=256):
        for i in range(epochs):
            x = self.generator.generate_latent_element(batch_size)
            y = np.ones((batch_size, 1))
            self.gan.train_on_batch(x, y)

    def summary(self): self.model.summary()

def main():
    gan = MNISTGAN()
    gan.summary()

if __name__ == '__main__':
    main()
