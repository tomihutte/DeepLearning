import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocesado

x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:])).astype(
    np.float
)

x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:])).astype(
    np.float
)

y_test, y_train = y_test.ravel(), y_train.ravel()
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_test = (x_test - mean) / std
x_train = (x_train - mean) / std

y_train, y_test = to_categorical(y_train), to_categorical(y_test)


# Modelos
dim_input = x_train[0].shape

############################################################
# Ejercicio 2-3
############################################################

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns

n_clases = 10  # clasificaciones de los datos
n_neuronas = 100  # neuronas de la capa intermedia
epochs = 200  # epocas de entrenamiento
batch_size = 50  # tamaño del batch
lambd = 1e-4  # 1e-4  # factor de regularización
lr = 1e-2  # learning rate
weight = 1e-3  # pesos iniciales de las matrices

model = keras.models.Sequential(name="ej3")

model.add(
    keras.layers.Dense(
        n_neuronas,
        activation="sigmoid",
        activity_regularizer=keras.regularizers.l2(lambd),
        input_shape=dim_input,
    )
)

model.add(
    keras.layers.Dense(
        n_clases,
        activation="linear",
        activity_regularizer=keras.regularizers.l2(lambd),
    )
)

optimizer = keras.optimizers.SGD(learning_rate=lr)

model.compile(
    optimizer=optimizer, loss=keras.losses.MSE, metrics=["acc"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=2,
)


# Datos y graficos

np.save("history_ej2_3.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

# plt.figure(figsize=(8, 6))
plt.plot(
    history.history["val_loss"] / np.max(history.history["val_loss"]),
    lw=lw,
    label="Costo normalizado",
)
plt.plot(history.history["val_acc"], lw=lw, label="Precisión")
plt.title(
    r"Activaciones Sigmoide+Lineal - Costo MSE - $lr={}$ - $\lambda={}$".format(
        lr, lambd
    ),
    fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("MSE normalizado - Accuracy", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.savefig("EJ2_3_loss_acc.pdf")
plt.show()


############################################################
# Ejercicio 2-4
############################################################

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns

model = keras.models.Sequential(name="ej4")

n_clases = 10  # clasificaciones de los datos
n_neuronas = 100  # neuronas de la capa intermedia
epochs = 200  # epocas de entrenamiento
batch_size = 50  # tamaño del batch
lambd = 1e-6  # 1e-4  # factor de regularización
lr = 1e-4  # learning rate
weight = 1e-3  # pesos iniciales de las matrices

model.add(
    keras.layers.Dense(
        n_neuronas,
        activation="sigmoid",
        activity_regularizer=keras.regularizers.l2(lambd),
        input_shape=dim_input,
    )
)

model.add(
    keras.layers.Dense(
        n_clases,
        activation="linear",
        activity_regularizer=keras.regularizers.l2(lambd),
    )
)

optimizer = keras.optimizers.SGD(learning_rate=lr)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=2,
)


# Datos y graficos

np.save("history_ej2_4.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

# plt.figure(figsize=(8, 6))
plt.plot(
    history.history["val_loss"] / np.max(history.history["val_loss"]),
    lw=lw,
    label="Costo normalizado",
)
plt.plot(history.history["val_acc"], lw=lw, label="Precisión")
plt.title(
    r"Activaciones Sigmoide+Lineal - Costo CCE - $lr={}$ - $\lambda={}$".format(
        lr, lambd
    ),
    fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("CCE normalizado - Accuracy", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.savefig("EJ2_4_loss_acc.pdf")
plt.show()


############################################################
# Ejercicio 2-6_1
############################################################
print("Me gusta la poronga")

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns

lr = 1e-1
epochs = 1000
tr = 0.9
print("Me gusta la poronga")

x_train = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y_train = np.array([1, 0, 0, 1]).reshape(4, 1)


# Modelo
inputs = keras.Input(shape=x_train[0].shape)
l1 = keras.layers.Dense(2, activation="tanh")(inputs)
output = keras.layers.Dense(1, activation="tanh")(l1)

model = keras.Model(inputs=inputs, outputs=output)

optimizer = keras.optimizers.SGD(learning_rate=lr)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.MSE,
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=tr)],
)

history = model.fit(x_train, y_train, epochs=epochs, verbose=2)

# Datos y graficos

np.save("history_ej2_6.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

# plt.figure(figsize=(8, 6))
plt.plot(
    history.history["loss"] / np.max(history.history["loss"]),
    lw=lw,
    label="Costo normalizado",
)
plt.plot(history.history["binary_accuracy"], lw=lw, label="Precisión")
plt.title(
    r"Modelo 1 - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("MSE normalizado - Accuracy", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ2_6_loss_acc.pdf")
plt.show()

print("Me gusta la poronga")
############################################################
# Ejercicio 2-6_2
############################################################
print("Me gusta la poronga")

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import cifar10
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns

lr = 1e-1
epochs = 1000
tr = 0.9

x_train = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y_train = np.array([1, 0, 0, 1]).reshape(4, 1)


# Modelo
inputs = keras.Input(shape=x_train[0].shape)
l1 = keras.layers.Dense(1, activation="tanh")(inputs)
l2 = keras.layers.Concatenate()([inputs, l1])
output = keras.layers.Dense(1, activation="tanh")(l2)

model = keras.Model(inputs=inputs, outputs=output)

optimizer = keras.optimizers.SGD(learning_rate=lr)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.MSE,
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=tr)],
)

history = model.fit(x_train, y_train, epochs=epochs, verbose=2)

# Datos y graficos

np.save("history_ej2_6_2.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

# plt.figure(figsize=(8, 6))
plt.plot(
    history.history["loss"] / np.max(history.history["loss"]),
    lw=lw,
    label="Costo normalizado",
)
plt.plot(history.history["binary_accuracy"], lw=lw, label="Precisión")
plt.title(
    r"Modelo 2 - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("MSE normalizado - Accuracy", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ2_6_2_loss_acc.pdf")
plt.show()
