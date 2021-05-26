import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split


(x_train_0, y_train_0), (x_test_0, y_test_0) = keras.datasets.mnist.load_data()

## Preprocesado
x_train = (x_train_0.astype(np.float32) / 255).reshape(len(x_train_0), np.prod(x_train_0[0].shape))
x_test = (x_test_0.astype(np.float32) / 255).reshape(len(x_test_0), np.prod(x_test_0[0].shape))

y_test = keras.utils.to_categorical(y_test_0)
y_train = keras.utils.to_categorical(y_train_0)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000)

#### Red de capas densas

# Parametros
lr = 1e-5
lamb = 1e-1
batch_size = 64
epochs = 100
n_1 = 20
n_2 = 10


model = keras.models.Sequential(name="EJ8_Densa")
model.add(
    keras.layers.Dense(
        n_1,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(lamb),
        name="dense_1",
        input_shape=x_train[0].shape,
    )
)
model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
model.add(
    keras.layers.Dense(
        n_2, activation="relu", kernel_regularizer=keras.regularizers.l2(lamb), name="dense_2"
    )
)
model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
model.add(
    keras.layers.Dense(
        10, activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense_3"
    )
)

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True),
    metrics=["acc"],  # keras.metrics.BinaryAccuracy(threshold=0.5)],
)

print(model.summary())

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
)

# Datos y graficos
print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

plt.figure(figsize=(6, 5))
plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
plt.title(r"Capas densas - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.ylim(0, 1)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ8_dense_acc.pdf")
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(
    history.history["loss"], lw=lw, label="Entrenamiento",
)
plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
plt.title(r"Capas densas - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ8_dense_loss.pdf")
plt.show()


###### Red convolucional
## Preprocesado
x_test = x_test.reshape(len(x_test), 28, 28, 1)
x_train = (x_train_0.astype(np.float32) / 255)[:, :, :, np.newaxis]
x_val = (x_test_0.astype(np.float32) / 255)[:, :, :, np.newaxis]
y_val = keras.utils.to_categorical(y_test_0)
y_train = keras.utils.to_categorical(y_train_0)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000)


# Parametros
lr = 1e-5
lamb = 1e-1
batch_size = 64
epochs = 100
n_1 = 32
n_2 = 16


model = keras.models.Sequential(name="EJ8_Conv")
model.add(
    keras.layers.Conv2D(
        filters=n_1,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=keras.regularizers.l2(lamb),
        name="conv_1",
        input_shape=x_train[0].shape,
    )
)
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
model.add(
    keras.layers.Conv2D(
        filters=n_2,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=keras.regularizers.l2(lamb),
        name="conv_2",
    )
)
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
model.add(keras.layers.Flatten())
model.add(
    keras.layers.Dense(
        10, activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"
    )
)

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True),
    metrics=["acc"],  # keras.metrics.BinaryAccuracy(threshold=0.5)],
)

print(model.summary())

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
)

# Datos y graficos
print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

plt.figure(figsize=(6, 5))
plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
plt.title(r"Red Convolucional - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.ylim(0, 1)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ8_conv_acc.pdf")
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(
    history.history["loss"], lw=lw, label="Entrenamiento",
)
plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
plt.title(r"Red Convolucional  - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ8_conv_loss.pdf")
plt.show()
