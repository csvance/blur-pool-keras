import numpy as np
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
import keras


class MaxBlurPooling1D(Layer):

    def __init__(self, pool_size: int = 2, **kwargs):
        self.pool_size = pool_size
        self.avg_kernel = None
        self.blur_kernel = None
        self.pad_blur = None

        super(MaxBlurPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        ak = np.array([1 / 2, 1 / 2])
        ak = np.reshape(ak, (2, 1, 1))
        avg_init = keras.initializers.constant(ak)

        bk = np.array([2, 4, 2])
        bk = bk / np.sum(bk)
        bk = np.reshape(bk, (3, 1, 1))
        blur_init = keras.initializers.constant(bk)

        self.avg_kernel = self.add_weight(name='avg_kernel',
                                          shape=(2, 1, 1),
                                          initializer=avg_init,
                                          trainable=False)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(3, 1, 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        dimensions = []

        for d in range(0, x.shape[2]):
            xd = x[:, :, d]
            xd = K.expand_dims(xd, axis=-1)
            xd = tf.nn.pool(xd, (self.pool_size, ), strides=(1, ), padding='SAME', pooling_type='MAX', data_format='NWC')
            xd = K.conv1d(xd, self.blur_kernel, padding='same')
            xd = K.conv1d(xd, self.avg_kernel, padding='valid', strides=self.pool_size)

            dimensions.append(xd)

        return K.concatenate(dimensions, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.floor(input_shape[1] / 2)), input_shape[2]


class MaxBlurPooling2D(Layer):

    def __init__(self, pool_size: int = 2, **kwargs):
        self.pool_size = pool_size
        self.avg_kernel = None
        self.blur_kernel = None
        self.pad_blur = None

        super(MaxBlurPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):

        ak = np.array([[1 / 4, 1 / 4],
                       [1 / 4, 1 / 4]])
        ak = np.reshape(ak, (2, 2, 1, 1))
        avg_init = keras.initializers.constant(ak)

        bk = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
        bk = bk / np.sum(bk)
        bk = np.reshape(bk, (3, 3, 1, 1))
        blur_init = keras.initializers.constant(bk)

        self.avg_kernel = self.add_weight(name='avg_kernel',
                                          shape=(2, 2, 1, 1),
                                          initializer=avg_init,
                                          trainable=False)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(3, 3, 1, 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dimensions = []

        for d in range(0, x.shape[3]):
            xd = x[:, :, :, d]
            xd = K.expand_dims(xd, axis=-1)
            xd = tf.nn.pool(xd, (self.pool_size, self.pool_size), strides=(1, 1), padding='SAME', pooling_type='MAX', data_format='NHWC')
            xd = K.conv2d(xd, self.blur_kernel, padding='same')
            xd = K.conv2d(xd, self.avg_kernel, padding='valid', strides=(self.pool_size, self.pool_size))

            dimensions.append(xd)

        return K.concatenate(dimensions, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.floor(input_shape[1] / 2)), int(np.floor(input_shape[2] / 2)), input_shape[3]


class AverageBlurPooling1D(Layer):

    def __init__(self, pool_size: int = 2, **kwargs):
        self.pool_size = pool_size
        self.avg_kernel = None
        self.blur_kernel = None
        self.pad_blur = None

        super(AverageBlurPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        ak = np.array([1 / 2, 1 / 2])
        ak = np.reshape(ak, (2, 1, 1))
        avg_init = keras.initializers.constant(ak)

        bk = np.array([2, 4, 2])
        bk = bk / np.sum(bk)
        bk = np.reshape(bk, (3, 1, 1))
        blur_init = keras.initializers.constant(bk)

        self.avg_kernel = self.add_weight(name='avg_kernel',
                                          shape=(2, 1, 1),
                                          initializer=avg_init,
                                          trainable=False)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(3, 1, 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(AverageBlurPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        dimensions = []

        for d in range(0, x.shape[2]):
            xd = x[:, :, d]
            xd = K.expand_dims(xd, axis=-1)
            xd = K.conv1d(xd, self.avg_kernel, padding='same')
            xd = K.conv1d(xd, self.blur_kernel, padding='same')
            xd = K.conv1d(xd, self.avg_kernel, padding='valid', strides=self.pool_size)

            dimensions.append(xd)

        return K.concatenate(dimensions, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.floor(input_shape[1] / 2)), input_shape[2]


class AverageBlurPooling2D(Layer):

    def __init__(self, pool_size: int = 2, **kwargs):
        self.pool_size = pool_size
        self.avg_kernel = None
        self.blur_kernel = None
        self.pad_blur = None

        super(AverageBlurPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        ak = np.array([[1 / 4, 1 / 4],
                       [1 / 4, 1 / 4]])
        ak = np.reshape(ak, (2, 2, 1, 1))
        avg_init = keras.initializers.constant(ak)

        bk = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
        bk = bk / np.sum(bk)
        bk = np.reshape(bk, (3, 3, 1, 1))
        blur_init = keras.initializers.constant(bk)

        self.avg_kernel = self.add_weight(name='avg_kernel',
                                          shape=(2, 2, 1, 1),
                                          initializer=avg_init,
                                          trainable=False)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(3, 3, 1, 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(AverageBlurPooling2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dimensions = []

        for d in range(0, x.shape[3]):
            xd = x[:, :, :, d]
            xd = K.expand_dims(xd, axis=-1)
            xd = K.conv2d(xd, self.avg_kernel, padding='same', data_format='channels_last')
            xd = K.conv2d(xd, self.blur_kernel, padding='same')
            xd = K.conv2d(xd, self.avg_kernel, padding='valid', strides=(self.pool_size, self.pool_size))

            dimensions.append(xd)

        return K.concatenate(dimensions, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.floor(input_shape[1] / 2)), int(np.floor(input_shape[2] / 2)), input_shape[3]
