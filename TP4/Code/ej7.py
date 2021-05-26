import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split


# Parámetros
noise_mean = 0
noise_std = 0.5
lr = 1e-3
lamb = 1e-4
batch_size = 64
epochs = 20

# Datos
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

x_train = (x_train.astype("float32") / 255)[:, :, :, np.newaxis]
x_test = (x_test.astype("float32") / 255)[:, :, :, np.newaxis]
y_train = np.copy(x_train)
y_test = np.copy(x_test)


# Agregamos ruido
np.random.seed(5)
x_train += np.random.normal(loc=noise_mean, scale=noise_std, size=x_train.shape)
x_test += np.random.normal(loc=noise_mean, scale=noise_std, size=x_test.shape)
x_train = np.clip(x_train, 0, 1)
x_test = np.clip(x_test, 0, 1)
# Modelo
# guiandome por https://blog.keras.io/building-autoencoders-in-keras.html


input_img = keras.Input(shape=x_train[0].shape)

x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
encoded = keras.layers.MaxPooling2D((2, 2), padding="same")(x)

x = keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

model = keras.Model(input_img, decoded)

model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss="binary_crossentropy")

print(model.summary())

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
)

# Graficamos
n = 3
plt.figure(figsize=(n, 2))
for i in range(1, n + 1):
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + n)
    plt.imshow(model.predict(x_test[i][np.newaxis, :]).reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("ej7.pdf")
plt.show()
