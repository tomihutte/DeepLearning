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


lambd = 1e-6  # 1e-6
lr = 1e-1
weights = 1
epochs = 100000
n = 10  # dimension de los ejemplos
n_ejemplos = 2 ** n  # numero de ejemplos
n_prima = 20  # int(n / 2)
n_test = int(n_ejemplos / 10)

np.random.seed(3)

x_train = np.array([[int(val) if val == "1" else -1 for val in np.binary_repr(num, width=n)] for num in range(n_ejemplos)])
y_train = np.prod(x_train, axis=1).reshape(len(x_train), 1)

index = np.arange(len(x_train))
np.random.shuffle(index)
x_test = x_train[index[:n_test]]
y_test = y_train[index[:n_test]]
x_train = x_train[index[n_test:]]
y_train = y_train[index[n_test:]]


reg1 = regularizers.L2(lambd)
reg2 = regularizers.L1(lambd)


layer0 = layers.InputLayer(x_train.shape[1])
mod = models.Network(layer0)
layer1 = layers.WLayer(n_neurons=n_prima, activation=activations.Tanh(), weights=weights, regularizer=reg1)
layer2 = layers.WLayer(n_neurons=1, activation=activations.Tanh(), weights=weights, regularizer=reg2)

mod.add_layer(layer1)
mod.add_layer(layer2)

loss_train, acc_train, loss_test, acc_test = mod.fit(
    x_train,
    y_train,
    epochs,
    loss=losses.MSE_XOR(),
    opt=optimizers.SGD(lr=lr),
    accuracy=metrics.accuracy_xor,
    x_test=x_test,
    y_test=y_test,
    print_epoch=500,
)

np.savetxt("EJ7_{}.txt".format(n_prima), np.column_stack((loss_train, acc_train, loss_test, acc_test)), delimiter="\t")
