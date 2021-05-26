#### Ejercicio 10  VGG16#####

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000, stratify=y_train)

# Preprocesamiento
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
x_val = (x_val - mean) / std

# Modelo ALEX-NET
# Parametros
lr = 1e-4
lamb = 1e-2
batch_size = 100
epochs = 100
n_clases = y_train.shape[1]

# reg
reg = keras.regularizers.l2(lamb)

model = keras.Sequential(name="VGG16")

model.add(keras.layers.BatchNormalization(input_shape=(x_train[0].shape)))
model.add(keras.layers.Conv2D(32, (3, 3), strides=1, padding="same", kernel_regularizer=reg, activation="relu",))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", kernel_regularizer=reg, activation="relu",))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation="relu",))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, kernel_regularizer=reg, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(128, kernel_regularizer=reg, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(128, kernel_regularizer=reg, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(n_clases, kernel_regularizer=reg, activation="linear"))

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True),
    metrics=["acc"],  # keras.metrics.BinaryAccuracy(threshold=0.5)],
)

print(model.summary())

# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=45, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
# datagen.fit(x_train)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    steps_per_epoch=len(x_train) / batch_size,
    validation_data=(x_val, y_val),
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
plt.title(r"VGG - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.ylim(0, 1)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ10_VGG_cifar100_acc.pdf")
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(
    history.history["loss"], lw=lw, label="Entrenamiento",
)
plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
plt.title(r"VGG - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ10_VGG_cifar100_loss.pdf")
plt.show()
