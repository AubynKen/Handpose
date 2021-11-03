from tensorflow import keras
from tensorflow.keras import activations
import tensorflow as tf
import Config as cfg


def convolutionalBlock(inputs, out=16, kernel=(3,3)):
    X = keras.layers.Conv2D(filters=out, kernel_size=kernel, activation=activations.relu, padding='same')(inputs)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Conv2D(filters=out, kernel_size=kernel, activation=activations.relu, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)

    return X


def unetModel(input_size=(128, 128, 3), out = 16, final_layer=21, kernel=(3,3)):
    inputs = keras.Input(input_size, batch_size=cfg.batch_size)

    X1 = convolutionalBlock(inputs, out=out, kernel=kernel)
    X = keras.layers.MaxPool2D((2, 2))(X1)

    X2 = convolutionalBlock(X, out=out * 2, kernel=kernel)
    X = keras.layers.MaxPool2D((2, 2))(X2)

    X3 = convolutionalBlock(X, out=out * 4, kernel=kernel)
    X = keras.layers.MaxPool2D((2, 2))(X3)

    X = convolutionalBlock(X, out=out * 8, kernel=kernel)
    X = keras.layers.UpSampling2D((2, 2))(X)

    X = tf.concat([X, X3], axis=-1)
    X = convolutionalBlock(X, out=out * 4, kernel=kernel)
    X = keras.layers.UpSampling2D((2, 2))(X)

    # block 6
    X = tf.concat([X, X2], axis=-1)
    X = convolutionalBlock(X, out=out * 2, kernel=kernel)
    X = keras.layers.UpSampling2D((2, 2))(X)

    # block 7
    X = tf.concat([X, X1], axis=-1)
    X = convolutionalBlock(X, out=out, kernel=kernel)

    # output
    X = convolutionalBlock(X, out=final_layer, kernel=kernel)
    out = tf.keras.activations.sigmoid(X)

    model = keras.Model(inputs, out)

    return model