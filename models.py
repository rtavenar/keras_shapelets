from keras.models import Sequential
from keras.layers import Dense, Conv1D, Layer
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical
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
        return (input_shape[0], input_shape[2])

    def call(self, inputs, **kwargs):
        return K.min(inputs, axis=1)


class ShapeletModel:
    def __init__(self, n_shapelets, shapelet_size, optimizer="sgd"):
        self.n_shapelets = n_shapelets
        self.shapelet_size = shapelet_size
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.model = Sequential()
        self.layers = []

    def fit(self, X, y):
        n_ts, sz, d = X.shape
        assert(d == 1)
        y_ = to_categorical(y)
        n_classes = y_.shape[1]
        self._set_model_layers(ts_sz=sz, d=d, n_classes=n_classes)
        self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=[categorical_accuracy])
        self._set_weights_false_conv()

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
                   use_bias=False),
            # TODO: add L2 distance layer here
            GlobalMinPooling1D(),
            Dense(units=n_classes)
        ]
        for l in self.layers[:1]:
            self.model.add(l)

    def get_weights(self):
        return self.model.get_weights()


if __name__ == "__main__":
    numpy.random.seed(0)
    n_classes = 2
    X = numpy.random.randn(1000, 256, 1)
    y = numpy.random.randint(n_classes, size=1000)
    clf = ShapeletModel(n_shapelets=5, shapelet_size=16)
    clf.fit(X, y)
    pred = clf.model.predict(X)
    print(X.shape, pred.shape)
    print(X[0, :20, 0])
    print(pred[0, :3])