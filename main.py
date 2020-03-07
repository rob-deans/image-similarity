import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

(train_images, train_labels), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28*28).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28*28).astype('float32')

# Normalizing the images to the range of [0., 1.]
train_images /= 255.0
# test_images /= 255.0

# TRAIN_BUF = 60000
# BATCH_SIZE = 100

# TEST_BUF = 10000

# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
# test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

x = layers.Input(shape=(784), name='encoder_input')
encoder = layers.Dense(units=256, activation='relu')(x)
encoder = layers.Dense(units=128, activation='relu')(encoder)
encoder = layers.Dense(units=32, activation='relu')(encoder)

encoder = models.Model(x, encoder, name='encoder_model')

decoder_input = layers.Input(shape=32)
decoder = layers.Dense(units=128, activation='relu')(decoder_input)
decoder = layers.Dense(units=256, activation='relu')(decoder)
decoder = layers.Dense(units=784)(decoder)
decoder = models.Model(decoder_input, decoder, name='decoder_model')

ae_input = layers.Input(shape=(784), name='ae_input')
encoder_out = encoder(ae_input)
decoder_out = decoder(encoder_out)

ae = models.Model(ae_input, decoder_out, name='ae')

loss = tf.keras.losses.MeanSquaredError()

ae.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
ae.fit(train_images, train_images, epochs=30, shuffle=True, batch_size=256)

print(test_images.shape)
res = encoder.predict(test_images)
t = encoder.predict(train_images[0].reshape(1,-1))

sim = cosine_similarity(res, t.reshape(1,-1))
res = decoder.predict(res)
sorted_sim = np.argsort(sim,axis=None)[-5:]
plt.imshow(train_images[0].reshape(28,28))
plt.show()
for ss in sorted_sim:
    plt.imshow(res[ss].reshape(28,28))
    plt.show()
