##### Conv Ej 8 - Fashion MNITS a MNIST transfer learning #########
###### Ejercicio 3 #######

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

# Voy a entrenar una red sobre MNIST y despues transfiero a fashion MNIST
fig_size = (7, 6)
n_train = 500  # datos para entrenar con fashion MNIST

### Cargamos los datos de MNIST
v_size = 10000
t_size = 10000

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x = np.vstack((x_train, x_test))
y = np.append(y_train, y_test)

# Preprocesado
x = x / 255.0
x = x.reshape(len(x), 28, 28, 1)
y = to_categorical(y)

x, x_test, y, y_test = train_test_split(x, y, test_size=t_size, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=v_size, stratify=y_train)


###### Red convolucional del ej 8 del tp 4 #########
###### Entrenada con MNIST ################
# Parametros
lr = 1e-4
lamb = 1e-2
batch_size = 100
epochs = 50
n_1 = 32
n_2 = 16

model = keras.models.Sequential(name="TomiNet")
model.add(keras.layers.Conv2D(filters=n_1, kernel_size=(3, 3), activation="relu", padding="same", name="conv_1", input_shape=x_train[0].shape,))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
model.add(keras.layers.Conv2D(filters=n_2, kernel_size=(3, 3), activation="relu", padding="same", name="conv_2",))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(y_train.shape[1], activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"))

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
)

print(model.summary())

# No hago data augmentation porque tengo una banda

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

# Datos y graficos
print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

plt.figure(figsize=fig_size)
plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
plt.title(r"MNIST - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.ylim(0, 1)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3__mnist_acc.pdf")
plt.show()

plt.figure(figsize=fig_size)
plt.plot(
    history.history["loss"], lw=lw, label="Entrenamiento",
)
plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
plt.title(r"MNIST  - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_mnist_loss.pdf")
plt.show()

## Guardo los datos para usarlos ahora
model.save_weights("MNIST")
model.save("TomiNet.h5")


# Cargo los datos, es necesario bajarlos desde https://www.kaggle.com/zalando-research/fashionmnist

data_train = np.loadtxt("fashion-mnist_train.csv", delimiter=",", skiprows=1)
data_test = np.loadtxt("fashion-mnist_test.csv", delimiter=",", skiprows=1)

data = np.vstack((data_train, data_test))

x = data[:, 1:].reshape(len(data), 28, 28, 1)
y = to_categorical(data[:, 0])[:, 1:]

# Preprocesado
x /= 255.0

# Division val y test
t_size = 10000
v_size = 10000

x, x_test, y, y_test = train_test_split(x, y, test_size=t_size, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=v_size, stratify=y)


# Resolvamos el problema sin usar transfer learning
# Uso el mismo modelo entrenado con MNIST
lr = 1e-3
lamb = 1e-3
epochs = 50
batch_size = int(n_train / 10)


# No se como cargar el modelo que guarde pero con los pesos random asi que lo creo de nuevo

model = keras.models.Sequential(name="TomiNet2")
model.add(keras.layers.Conv2D(filters=n_1, kernel_size=(3, 3), activation="relu", padding="same", name="conv_1", input_shape=x_train[0].shape,))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
model.add(keras.layers.Conv2D(filters=n_2, kernel_size=(3, 3), activation="relu", padding="same", name="conv_2",))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(y_train.shape[1], activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"))

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
)

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
)


# Parametros

epochs = 50

# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=45, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)

history = model.fit(
    datagen.flow(x_train[:n_train], y_train[:n_train], batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=n_train / batch_size,
    validation_data=(x_val, y_val),
    verbose=2,
)

# Datos y graficos
print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

plt.figure(figsize=fig_size)
plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
plt.title(r"FMNIST- Sin TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.ylim(0, 1)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_fmnist_noTL_acc.pdf")
plt.show()

plt.figure(figsize=fig_size)
plt.plot(
    history.history["loss"], lw=lw, label="Entrenamiento",
)
plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
plt.title(r"MNIST- Sin TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_fmnist_noTL_loss.pdf")
plt.show()


########## Vamos a clasificar MNIST con learning-transfer ##############
# Cargo el modelo, me quedo con las capas convolucionales nomas
model_emnist = keras.models.load_model("TomiNet.h5")
model_emnist = keras.Model(inputs=model_emnist.inputs, outputs=model_emnist.layers[-2].output)
model_emnist.load_weights("fashionMNIST")

model_emnist.trainable = False

## Uso el modelo con MNIST

inputs = keras.Input(shape=x_train[0].shape)
x = model_emnist(inputs)
x = keras.layers.Flatten()(x)
out = keras.layers.Dense(y_train.shape[1])(x)
model = keras.Model(inputs, out)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
)

## Cuantos datos de train queremos usar, para probar que tan bien anda la cosa

# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=45, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)

history = model.fit(
    datagen.flow(x_train[:n_train], y_train[:n_train], batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=n_train / batch_size,
    validation_data=(x_val, y_val),
    verbose=2,
)

# Ahora hago el fine tunning
lr_factor = 10
lr /= lr_factor
epochs_ft = 15

model_emnist.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
)


history2 = model.fit(
    datagen.flow(x_train[:n_train], y_train[:n_train], batch_size=batch_size),
    epochs=epochs_ft,
    steps_per_epoch=n_train / batch_size,
    validation_data=(x_val, y_val),
    verbose=2,
)

# Datos y graficos
print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25
lr = int(lr * lr_factor)

plt.figure(figsize=fig_size)
plt.plot(np.append(history.history["acc"], history2.history["acc"]), lw=lw, label="Entrenamiento")
plt.plot(np.append(history.history["val_acc"], history2.history["val_acc"]), lw=lw, label="Validación", ls="--")
plt.vlines(x=epochs, ymin=0, ymax=1, ls="--")
plt.title(r"MNIST- TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.ylim(0, 1)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_fmnist_acc.pdf")
plt.show()

plt.figure(figsize=fig_size)
plt.plot(
    np.append(history.history["loss"], history2.history["loss"]), lw=lw, label="Entrenamiento",
)
plt.plot(np.append(history.history["val_loss"], history2.history["val_loss"]), lw=lw, label="Validación", ls="--")
plt.title(r"MNIST- TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
(ymin, ymax) = plt.ylim()
plt.vlines(x=epochs, ymin=ymin, ymax=ymax, ls="--")
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_fmnist_loss.pdf")
plt.show()

