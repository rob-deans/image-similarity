import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

(train_images, train_labels), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28,28,1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28,28,1).astype('float32')

# Normalizing the images to the range of [0., 1.]
train_images /= 255.0
test_images /= 255.0


class AutoEncoder():
    def __init__(self, vector_size=32, conv=False):
        self.vector_size = vector_size
        self.conv = conv
        self.conv_kernel = (3,3)
        self.pool_kernel = (2,2)
        self.encoder = self.create_encoder()
        print(self.encoder.summary())
        self.decoder = self.create_decoder()
        self.ae = self.create_ae()

    def create_encoder(self):
        if self.conv:
            x = layers.Input(shape=(28,28,1), name='encoder_input')

            encoder = layers.Conv2D(64, self.conv_kernel, padding='same', activation='relu')(x)
            # encoder = layers.MaxPool2D(self.pool_kernel, padding='same')(encoder)
            encoder = layers.Conv2D(32, self.conv_kernel, padding='same', activation='relu')(encoder)
            encoder = layers.MaxPool2D(self.pool_kernel, padding='same')(encoder)
            encoder = layers.Conv2D(8, self.conv_kernel, padding='same', activation='relu')(encoder)
            encoder = layers.MaxPool2D(self.pool_kernel, padding='same')(encoder)
        else:
            x = layers.Input(shape=(784), name='encoder_input')
            encoder = layers.Dense(units=256, activation='relu')(x)
            encoder = layers.Dense(units=128, activation='relu')(encoder)
            encoder = layers.Dense(units=self.vector_size, activation='relu')(encoder)
        return models.Model(x, encoder, name='encoder_model')

    def create_decoder(self):
        if self.conv:
            decoder_input = layers.Input(shape=(7,7,8))
            decoder = layers.Conv2D(8, self.conv_kernel, padding='same', activation='relu')(decoder_input)
            decoder = layers.UpSampling2D(self.pool_kernel)(decoder)
            decoder = layers.Conv2D(32, self.conv_kernel, padding='same', activation='relu')(decoder)
            # decoder = layers.UpSampling2D(self.pool_kernel)(decoder)
            decoder = layers.Conv2D(64, self.conv_kernel, padding='same', activation='relu')(decoder)
            decoder = layers.UpSampling2D(self.pool_kernel)(decoder)
            decoder = layers.Conv2D(1, self.conv_kernel, activation='sigmoid', padding='same')(decoder)
        else:
            decoder_input = layers.Input(shape=self.vector_size)
            decoder = layers.Dense(units=128, activation='relu')(decoder_input)
            decoder = layers.Dense(units=256, activation='relu')(decoder)
            decoder = layers.Dense(units=784, activation='sigmoid')(decoder)
        return models.Model(decoder_input, decoder, name='decoder_model')

    def create_ae(self):
        if self.conv:
            ae_input = layers.Input(shape=(28,28,1), name='ae_input')
        else:
            ae_input = layers.Input(shape=(784), name='ae_input')
        encoder_out = self.encoder(ae_input)
        decoder_out = self.decoder(encoder_out)

        ae = models.Model(ae_input, decoder_out, name='ae')

        # loss = tf.keras.losses.MeanSquaredError()
        # loss = tf.keras.losses.BinaryCrossEntropy()

        ae.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return ae

ae = AutoEncoder(conv=True)

ae.ae.fit(train_images, train_images, epochs=16, shuffle=True, batch_size=64)

res = ae.encoder.predict(test_images)
t = ae.encoder.predict(train_images)

decoded = ae.decoder.predict(res)

sim = cosine_similarity(res.reshape(-1,np.prod(res[0].shape)), t[0].flatten().reshape(1,-1))
sorted_sim = np.argsort(sim,axis=None)[-5:]

plt.imshow(train_images[0].reshape(28,28))
plt.show()

for ss in sorted_sim:
    plt.imshow(decoded[ss].reshape(28,28))
    plt.show()
