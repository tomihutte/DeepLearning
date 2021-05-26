###### MOBILE NET ########
###### Ejercicio 3 #######

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

fig_size = (6, 5)

# Voy a entrenar una red sobre EMNIST y despues transfiero a MNIST

# CIFAR10

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x = np.vstack((x_train, x_test))
y = to_categorical(np.append(y_train, y_test))

# Preprocesado
x = x / 255.0

# Division val y test
t_size = 10000
v_size = 10000

x, x_test, y, y_test = train_test_split(x, y, test_size=t_size, stratify=y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=v_size, stratify=y)

# Modelo Base MobileNet
base_model = keras.applications.MobileNet(weights="imagenet", input_shape=x_train[0].shape, include_top=False)
base_model.trainable = False  # No se entrena

# Agregamos el modelo
inputs = keras.Input(shape=x_train[0].shape)
x = base_model(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(y_train.shape[1])(x)
model = keras.Model(inputs, outputs)

print(model.summary())

## Parametros
n_train = 5000  # cuantos datos de entrenamiento uso
lr = 1e-4
batch_size = 100
epochs = 10

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
)

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
lr /= 10
epochs_ft = 10

base_modeltrainable = True
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
lr = lr * 10

plt.figure(figsize=fig_size)
plt.plot(np.append(history.history["acc"], history2.history["acc"]), lw=lw, label="Entrenamiento")
plt.plot(np.append(history.history["val_acc"], history2.history["val_acc"]), lw=lw, label="Validación", ls="--")
plt.vlines(x=epochs, ymin=0, ymax=1, ls="--")
plt.title(r"ImageNet - n_train={} - $lr={}$".format(n_train, lr), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.ylim(0, 1)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_TL_acc.pdf")
plt.show()

plt.figure(figsize=fig_size)
plt.plot(
    np.append(history.history["loss"], history2.history["loss"]), lw=lw, label="Entrenamiento",
)
ymin, ymax = plt.ylim()
plt.vlines(x=epochs, ymin=ymin, ymax=ymax, ls="--")
plt.plot(np.append(history.history["val_loss"], history2.history["val_loss"]), lw=lw, label="Validación", ls="--")
plt.title(r"ImageNet - n_train={} - $lr={}$".format(n_train, lr), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_TL_loss.pdf")
plt.show()


# Resolvamos el problema sin usar transfer learning
# Cargo el modelo, me quedo con las capas convolucionales nomas
# Modelo Base MobileNet, no cargo los pesos

base_model = keras.applications.MobileNet(input_shape=x_train[0].shape, include_top=False)

# Agregamos el modelo
inputs = keras.Input(shape=x_train[0].shape)
x = base_model(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(y_train.shape[1])(x)
model = keras.Model(inputs, outputs)

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
)

## Cuantos datos de train queremos usar, para probar que tan bien anda la cosa
n_train = 5000
epochs += epochs_ft  # asi miramos en una misma cantidad de epocas

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
plt.title(r"Random - n_train={} - $lr={}$".format(n_train, lr), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.ylim(0, 1)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_noTL_acc.pdf")
plt.show()

plt.figure(figsize=fig_size)
plt.plot(
    history.history["loss"], lw=lw, label="Entrenamiento",
)
plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
plt.title(r"Random - n_train={} - $lr={}$".format(n_train, lr), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ_noTL_loss.pdf")
plt.show()
