from blurpool import *
import unittest
from keras.layers import *
from keras.models import Model


class TestBlurPool(unittest.TestCase):

    def test_avg_1d(self):

        layer_input = Input((224, 3))
        layer_pool = AverageBlurPooling1D()(layer_input)
        layer_flatten = Flatten()(layer_pool)
        layer_dense = Dense(1)(layer_flatten)

        model = Model(inputs=layer_input, outputs=layer_dense)
        model.summary()

        model.predict([np.random.random((1, 224, 3))])

    def test_max_1d(self):

        layer_input = Input((224, 3))
        layer_pool = MaxBlurPooling1D()(layer_input)
        layer_flatten = Flatten()(layer_pool)
        layer_dense = Dense(1)(layer_flatten)

        model = Model(inputs=layer_input, outputs=layer_dense)
        model.summary()

        model.predict([np.random.random((1, 224, 3))])

    def test_avg_2d(self):

        layer_input = Input((224, 224, 3))
        layer_pool = AverageBlurPooling2D()(layer_input)
        layer_flatten = Flatten()(layer_pool)
        layer_dense = Dense(1)(layer_flatten)

        model = Model(inputs=layer_input, outputs=layer_dense)
        model.summary()

        model.predict([np.random.random((1, 224, 224, 3))])

    def test_max_2d(self):

        layer_input = Input((224, 224, 3))
        layer_pool = MaxBlurPooling2D()(layer_input)
        layer_flatten = Flatten()(layer_pool)
        layer_dense = Dense(1)(layer_flatten)

        model = Model(inputs=layer_input, outputs=layer_dense)
        model.summary()

        model.predict([np.random.random((1, 224, 224, 3))])

    def test_blur_2d(self):

        layer_input = Input((224, 224, 3))
        layer_pool = BlurPool2D()(layer_input)
        layer_flatten = Flatten()(layer_pool)
        layer_dense = Dense(1)(layer_flatten)

        model = Model(inputs=layer_input, outputs=layer_dense)
        model.summary()

        model.predict([np.random.random((1, 224, 224, 3))])