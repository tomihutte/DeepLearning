#### Ejercicio 1 #####

## guiandome por https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

x = np.load("dogs_vs_cats_photos.npy")
y = np.load("dogs_vs_cats_labels.npy")

###### Imagenes #######

from matplotlib.image import imread

# define location of dataset
folder = "train/"
# plot first few images

n_img = 2

for i in range(n_img):
    # define subplot
    plt.subplot(n_img, 2, i * 2 + 1)
    # define filename
    filename = folder + "cat." + str(i) + ".jpg"
    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
for i in range(n_img):
    # define subplot
    plt.subplot(n_img, 2, i * 2 + 2)
    # define filename
    filename = folder + "dog." + str(i) + ".jpg"
    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
# show the figure
plt.tight_layout()
plt.savefig("full_size.pdf")
plt.show()

for i in range(n_img):
    # define subplot
    plt.subplot(n_img, 2, i * 2 + 1)
    plt.imshow(x[i])
    plt.xticks([])
    plt.yticks([])
for i in range(n_img):
    # define subplot
    plt.subplot(n_img, 2, i * 2 + 2)
    plt.imshow(x[12500 + i])
    plt.xticks([])
    plt.yticks([])
# show the figure
plt.tight_layout()
plt.savefig("compressed.pdf")
plt.show()


# Preprocesado
x = x.astype(np.float) / 255.0

# Division en test y val

t_size = 2500
v_size = 2500

x, x_test, y, y_test = train_test_split(x, y, test_size=t_size, stratify=True)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=v_size, stratify=True)


# Modelo VGG-16
# Parámetros
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
