import numpy as np
import matplotlib.pyplot as plt
from myModules import (
    regularizers,
    models,
    layers,
    optimizers,
    metrics,
    activations,
    losses,
)


lambd = 0  # 1e-6
lr = 1e-1
weights = 1
epochs = 1000

x_train = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y_train = np.array([1, -1, -1, 1]).reshape(4, 1)


reg1 = regularizers.L2(lambd)
reg2 = regularizers.L1(lambd)

np.random.seed(5)

layer0 = layers.InputLayer(x_train.shape[1])
mod = models.Network(layer0)
layer1 = layers.WLayer(n_neurons=2, activation=activations.Tanh(), weights=weights, regularizer=reg1)
# layer_concat = layers.ConcatLayer(layer0.get_output_shape(), mod.forward, index_layer2=0)
layer2 = layers.WLayer(n_neurons=1, activation=activations.Tanh(), weights=weights, regularizer=reg2)
mod.add_layer(layer1)
# mod.add_layer(layer_concat)
mod.add_layer(layer2)

loss_train, acc_train, _, _ = mod.fit(x_train, y_train, epochs, loss=losses.MSE_XOR(), opt=optimizers.SGD(lr=lr), accuracy=metrics.accuracy_xor,)

np.savetxt("EJ6_1.txt", np.column_stack((loss_train, acc_train)), delimiter="\t")
