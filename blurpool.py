import numpy as np
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
import keras


class MaxBlurPooling1D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.avg_kernel = None
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(MaxBlurPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.kernel_size == 3:
            bk = np.array([2, 4, 2])
        elif self.kernel_size == 5:
            bk = np.array([6, 24, 36, 24, 6])
        else:
            raise ValueError

        bk = bk / np.sum(bk)
        bk = np.reshape(bk, (3, 1, 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(3, 1, 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size,), strides=(1,),
                       padding='SAME', pooling_type='MAX', data_format='NWC')

        dimensions = []
        for d in range(0, x.shape[2]):
            xd = x[:, :, d]
            xd = K.expand_dims(xd, axis=-1)
            xd = K.conv1d(xd, self.blur_kernel, padding='same')
            dimensions.append(xd)

        x = K.concatenate(dimensions, axis=-1)
        x = tf.nn.pool(x, (self.pool_size,), strides=(self.pool_size,),
                       padding='SAME', pooling_type='MAX', data_format='NWC')

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.floor(input_shape[1] / 2)), input_shape[2]


class MaxBlurPooling2D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(MaxBlurPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
        elif self.kernel_size == 5:
            bk = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
        else:
            raise ValueError

        bk = bk / np.sum(bk)
        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, 1, 1))

        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, 1, 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size, self.pool_size),
                       strides=(1, 1), padding='SAME', pooling_type='MAX', data_format='NHWC')

        dimensions = []
        for d in range(0, x.shape[3]):
            xd = x[:, :, :, d]
            xd = K.expand_dims(xd, axis=-1)
            xd = K.conv2d(xd, self.blur_kernel, padding='same')

            dimensions.append(xd)

        x = K.concatenate(dimensions, axis=-1)
        x = tf.nn.pool(x, (self.pool_size, self.pool_size),
                       strides=(self.pool_size, self.pool_size),
                       padding='VALID', pooling_type='AVG', data_format='NHWC')

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.floor(input_shape[1] / 2)), int(np.floor(input_shape[2] / 2)), input_shape[3]


class AverageBlurPooling1D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(AverageBlurPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([2, 4, 2])
        elif self.kernel_size == 5:
            bk = np.array([6, 24, 36, 24, 6])
        else:
            raise ValueError

        bk = bk / np.sum(bk)
        bk = np.reshape(bk, (3, 1, 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(AverageBlurPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size,), strides=(1,), padding='SAME', pooling_type='AVG',
                       data_format='NWC')

        dimensions = []
        for d in range(0, x.shape[2]):
            xd = x[:, :, d]
            xd = K.expand_dims(xd, axis=-1)
            xd = K.conv1d(xd, self.blur_kernel, padding='same')

            dimensions.append(xd)

        x = K.concatenate(dimensions, axis=-1)
        x = tf.nn.pool(x, (self.pool_size,), strides=(self.pool_size,), padding='VALID', pooling_type='AVG',
                       data_format='NWC')

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.floor(input_shape[1] / 2)), input_shape[2]


class AverageBlurPooling2D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(AverageBlurPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 5:
            bk = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
            bk = bk / np.sum(bk)
        else:
            raise ValueError

        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, 1, 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, 1, 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(AverageBlurPooling2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dimensions = []

        x = tf.nn.pool(x, (self.pool_size, self.pool_size), strides=(1, 1), padding='SAME', pooling_type='AVG',
                       data_format='NHWC')

        for d in range(0, x.shape[3]):
            xd = x[:, :, :, d]
            xd = K.expand_dims(xd, axis=-1)
            xd = K.conv2d(xd, self.blur_kernel, padding='same')

            dimensions.append(xd)

        x = K.concatenate(dimensions, axis=-1)
        x = tf.nn.pool(x, (self.pool_size, self.pool_size), strides=(self.pool_size, self.pool_size), padding='VALID',
                       pooling_type='AVG',
                       data_format='NHWC')

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.floor(input_shape[1] / 2)), int(np.floor(input_shape[2] / 2)), input_shape[3]
