from keras.models import Model
from keras.layers import Dense, Conv1D, Layer, Input, Concatenate
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
        `(batch_size, features)`
    """

    def __init__(self, **kwargs):
        super(GlobalMinPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, inputs, **kwargs):
        return K.min(inputs, axis=1)


class LocalSquaredDistanceLayer(Layer):
    """Pairwise (squared) distance computation between local patches and shapelets
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        3D tensor with shape:
        `(batch_size, steps, n_shapelets)`
    """
    def __init__(self, n_shapelets, **kwargs):
        self.n_shapelets = n_shapelets
        super(LocalSquaredDistanceLayer, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.n_shapelets, input_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        super(LocalSquaredDistanceLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        # (x - y)^2 = x^2 + y^2 - 2 * x * y
        x_sq = K.expand_dims(K.sum(x ** 2, axis=2), axis=-1)
        y_sq = K.reshape(K.sum(self.kernel ** 2, axis=1), (1, 1, self.n_shapelets))
        xy = K.dot(x, K.transpose(self.kernel))
        return x_sq + y_sq - 2 * xy

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.n_shapelets


def grabocka_params_to_shapelet_size_dict(ts_sz, n_classes, l, r):
    base_size = int(l * ts_sz)
    d = {}
    for sz_idx in range(r):
        shp_sz = base_size * (sz_idx + 1)
        n_shapelets = int(numpy.log10(ts_sz - shp_sz + 1) * (n_classes - 1))
        d[shp_sz] = n_shapelets
    return d

class ShapeletModel:
    def __init__(self, n_shapelets_per_size, epochs=1000, batch_size=256, verbose_level=2, optimizer="sgd",
                 weight_regularizer=0.):
        self.n_shapelets_per_size = n_shapelets_per_size
        self.n_classes = None
        self.optimizer = optimizer
        self.epochs = epochs
        self.weight_regularizer = weight_regularizer
        self.model = None
        self.batch_size = batch_size
        self.verbose_level = verbose_level
        self.layers = []
        self.categorical_y = False

    @property
    def n_shapelet_sizes(self):
        return len(self.n_shapelets_per_size)

    @property
    def shapelets(self):
        return [self.model.get_layer("shapelets_%d" % i).get_weights()[0] for i in range(self.n_shapelet_sizes)]

    def fit(self, X, y):
        n_ts, sz, d = X.shape
        assert(d == 1)
        if y.ndim == 1:
            y_ = to_categorical(y)
        else:
            y_ = y
            self.categorical_y = True
        n_classes = y_.shape[1]
        self._set_model_layers(ts_sz=sz, d=d, n_classes=n_classes)
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=self.optimizer,
                           metrics=[categorical_accuracy, categorical_crossentropy])
        self._set_weights_false_conv(d=d)
        self.model.fit(X, y_, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose_level)
        return self

    def predict(self, X):
        categorical_preds = self.model.predict(X, batch_size=self.batch_size, verbose=self.verbose_level)
        if self.categorical_y:
            return categorical_preds
        else:
            return categorical_preds.argmax(axis=1)

    def _set_weights_false_conv(self, d):
        shapelet_sizes = sorted(self.n_shapelets_per_size.keys())
        for i, sz in enumerate(sorted(shapelet_sizes)):
            weights_false_conv = numpy.empty((sz, d, sz))
            for di in range(d):
                weights_false_conv[:, di, :] = numpy.eye(sz)
            self.model.get_layer("false_conv_%d" % i).set_weights([weights_false_conv])

    def _set_model_layers(self, ts_sz, d, n_classes):
        inputs = Input(shape=(ts_sz, d), name="input")
        shapelet_sizes = sorted(self.n_shapelets_per_size.keys())
        pool_layers = []
        for i, sz in enumerate(sorted(shapelet_sizes)):
            transformer_layer = Conv1D(filters=sz,
                                       kernel_size=sz,
                                       trainable=False,
                                       use_bias=False,
                                       name="false_conv_%d" % i)(inputs)
            shapelet_layer = LocalSquaredDistanceLayer(self.n_shapelets_per_size[sz],
                                                       name="shapelets_%d" % i)(transformer_layer)
            pool_layers.append(GlobalMinPooling1D(name="min_pooling_%d" % i)(shapelet_layer))
        if len(shapelet_sizes) > 1:
            concatenated_features = Concatenate()(pool_layers)
        else:
            concatenated_features = pool_layers[0]
        if self.weight_regularizer > 0.:
            outputs = Dense(units=n_classes,
                            activation="softmax",
                            kernel_regularizer=l2(self.weight_regularizer),
                            name="softmax")(concatenated_features)
        else:
            outputs = Dense(units=n_classes,
                            activation="softmax",
                            name="softmax")(concatenated_features)
        self.model = Model(inputs=inputs, outputs=outputs)

    def get_weights(self, layer_name=None):
        if layer_name is None:
            return self.model.get_weights()
        else:
            return self.model.get_layer(layer_name).get_weights()


if __name__ == "__main__":
    from tslearn.datasets import CachedDatasets
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)

    ts_sz = X_train.shape[1]
    l, r = 0.1, 2  # Taken (for dataset Trace) from the Table at: http://fs.ismll.de/publicspace/LearningShapelets/
    n_classes = len(set(y_train))

    n_shapelets_per_size = grabocka_params_to_shapelet_size_dict(ts_sz, n_classes, l, r)
    clf = ShapeletModel(n_shapelets_per_size=n_shapelets_per_size,
                        epochs=1000,
                        optimizer=RMSprop(lr=.001),
                        weight_regularizer=.01)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_train)
    print([shp.shape for shp in clf.shapelets])
    print(numpy.sum(y_train == pred))
