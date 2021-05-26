import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from myModules import (
    regularizers,
    models,
    layers,
    optimizers,
    metrics,
    activations,
    losses,
)


lambd = 1e-3  # 1e-6
lr = 1e-3
weights = 1e-2
epochs = 200
batch_size = 50
print(lambd, lr, weights, epochs)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocesado

x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:])).astype(np.float)
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:])).astype(np.float)
y_test, y_train = y_test.ravel(), y_train.ravel()
x_mean = np.mean(x_train, axis=0)
x_std = np.std(x_train, axis=0)
x_test = (x_test - x_mean) / x_std
x_train = (x_train - x_mean) / x_std


reg1 = regularizers.L2(lambd)
reg2 = regularizers.L2(lambd)
reg3 = regularizers.L2(lambd)

np.random.seed(1)

layer0 = layers.InputLayer(x_train.shape[1])
mod = models.Network(layer0)
layer1 = layers.WLayer(n_neurons=100, activation=activations.Sigmoid(), weights=weights, regularizer=reg1)
layer2 = layers.WLayer(n_neurons=100, activation=activations.Sigmoid(), weights=weights, regularizer=reg2)
layer3 = layers.WLayer(n_neurons=10, activation=activations.Identity(), weights=weights * 10, regularizer=reg3,)
mod.add_layer(layer1)
mod.add_layer(layer2)
mod.add_layer(layer3)

loss_train, acc_train, loss_test, acc_test = mod.fit(
    x_train,
    y_train,
    epochs,
    loss=losses.MSE(),
    opt=optimizers.SGD(lr=lr, batch_size=batch_size),
    accuracy=metrics.accuracy,
    x_test=x_test,
    y_test=y_test,
)
np.savetxt("EJ8_Sigmoid-Sigmoid-MSE.txt", np.column_stack((loss_train, acc_train, loss_test, acc_test)), delimiter="\t")
