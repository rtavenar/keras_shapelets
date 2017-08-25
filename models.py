from keras.models import Sequential
from keras.layers import Dense, Conv1D, Layer
from keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.regularizers import l2
import keras.backend as K
from keras.engine import InputSpec
import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class GlobalMinPooling1D(Layer):
    """Global min pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def __init__(self, **kwargs):
        super(GlobalMinPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, inputs, **kwargs):
        return K.min(inputs, axis=1)


class LocalSquaredDistanceLayer(Layer):
    def __init__(self, n_shapelets, **kwargs):
        self.n_shapelets = n_shapelets
        super(LocalSquaredDistanceLayer, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.n_shapelets, input_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        super(LocalSquaredDistanceLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        x_sq = K.expand_dims(K.sum(x ** 2, axis=2), axis=-1)
        y_sq = K.reshape(K.sum(self.kernel ** 2, axis=1), (1, 1, self.n_shapelets))
        xy = K.dot(x, K.transpose(self.kernel))
        return x_sq + y_sq - 2 * xy

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.n_shapelets


class ShapeletModel:
    def __init__(self, n_shapelets, shapelet_size, epochs=1000, optimizer="sgd", weight_regularizer=0.):
        self.n_shapelets = n_shapelets
        self.shapelet_size = shapelet_size
        self.n_classes = None
        self.optimizer = optimizer
        self.epochs = epochs
        self.weight_regularizer = weight_regularizer
        self.model = Sequential()
        self.batch_size = 500
        self.verbose_level = 2
        self.layers = []

    def fit(self, X, y):
        n_ts, sz, d = X.shape
        assert(d == 1)
        y_ = to_categorical(y)
        n_classes = y_.shape[1]
        self._set_model_layers(ts_sz=sz, d=d, n_classes=n_classes)
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=self.optimizer,
                           metrics=[categorical_accuracy, categorical_crossentropy])
        self._set_weights_false_conv()
        self.model.fit(X, y_, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose_level)
        return self

    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size, verbose=self.verbose_level)

    def _set_weights_false_conv(self):
        d = self.layers[0].get_weights()[0].shape[1]
        weights_false_conv = numpy.empty((self.shapelet_size, d, self.shapelet_size))
        for di in range(d):
            weights_false_conv[:, di, :] = numpy.eye(self.shapelet_size)
        self.layers[0].set_weights([weights_false_conv])

    def _set_model_layers(self, ts_sz, d, n_classes):
        self.layers = [
            Conv1D(filters=self.shapelet_size,
                   kernel_size=self.shapelet_size,
                   input_shape=(ts_sz, d),
                   trainable=False,
                   use_bias=False,
                   name="false_conv"),
            LocalSquaredDistanceLayer(self.n_shapelets, name="dists"),
            GlobalMinPooling1D(name="min_pooling")
        ]
        if self.weight_regularizer > 0.:
            self.layers.append(Dense(units=n_classes,
                                     activation="softmax",
                                     kernel_regularizer=l2(self.weight_regularizer),
                                     name="softmax"))
        else:
            self.layers.append(Dense(units=n_classes,
                                     activation="softmax",
                                     name="softmax"))
        for l in self.layers:
            self.model.add(l)

    def get_weights(self):
        return self.model.get_weights()


if __name__ == "__main__":
    from tslearn.datasets import CachedDatasets
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
    clf = ShapeletModel(n_shapelets=50, shapelet_size=32, optimizer=RMSprop(lr=.001), weight_regularizer=.01)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_train)
    print(numpy.sum(y_train == pred.argmin(axis=1)))
