import numpy as np
import myModules.activations as act
import myModules.regularizers as reg


class BaseLayer:
    def __init__(self):
        pass

    def get_output_shape(self):
        pass

    def set_output_shape(self):
        pass


class InputLayer(BaseLayer):
    def __init__(self, x_dim):
        self.output_shape = x_dim

    def __call__(self, X):
        return X

    def get_output_shape(self):
        return self.output_shape

    def set_output_shape(self, x_dim):
        self.output_shape = x_dim


class ConcatLayer(BaseLayer):
    def __init__(self, layer2_shape, forward, index_layer2=0):
        self.input2_shape = layer2_shape
        self.forward = forward
        self.index = index_layer2

    def __call__(self, S1, X):
        S2 = self.forward(X, self.index + 1)
        return np.hstack((S1, S2))

    def get_output_shape(self):
        return self.input1_shape + self.input2_shape

    def get_input1_shape(self):
        return self.input1_shape

    def get_input2_shape(self):
        return self.input2_shape

    def set_input_shape(self, layer1_shape):
        self.input1_shape = layer1_shape


class ConcatLayerInput(BaseLayer):
    def __init__(self, layer2_shape):
        self.input2_shape = layer2_shape

    def __call__(self, S1, S2):
        return np.hstack((S1, S2))

    def get_output_shape(self):
        return self.input1_shape + self.input2_shape

    def get_input1_shape(self):
        return self.input1_shape

    def get_input2_shape(self):
        return self.input2_shape

    def set_input_shape(self, layer1_shape):
        self.input1_shape = layer1_shape


class WLayer(BaseLayer):
    def __init__(
        self,
        n_neurons,
        regularizer=reg.L2(),
        activation=act.Identity(),
        weights=1e-3,
        bias=True,
    ):
        self.output_shape = n_neurons
        self.activation = activation
        self.reg = regularizer
        self.weights = weights
        self.bias = bias

    def dot(self, X):
        X_1 = np.copy(X)

        if self.bias:
            X_1 = np.hstack((np.ones((X.shape[0], 1)), X))
        return X_1.dot(self.W)

    def __call__(self, X):
        Y = self.dot(X)

        return self.activation(Y)

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.output_shape

    def get_weights(self):
        return self.W

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.init_weights()

    def init_weights(self):
        self.W = (
            np.random.uniform(
                -1, 1, size=(self.input_shape + int(self.bias), self.output_shape)
            )
            * self.weights
        )

    def update_weights(self, dW):
        self.W -= dW

