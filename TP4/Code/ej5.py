import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns
from sklearn.model_selection import train_test_split

# parametros
n_dat = 100000
n_neurons = 5
test_ratio = 0.2
val_ratio = 0.25

# creamos los datos
x = np.linspace(0, 1, n_dat)
y = 4 * x * (1 - x)

x, x_test, y, y_test = train_test_split(x, y, test_size=test_ratio)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_ratio)

# modelo
epochs = 2000
batch_size = 100
lr = 1e-5
lamd = 1e-2

inputs = keras.Input(shape=(1,))
l1 = keras.layers.Dense(n_neurons, activation="tanh")(inputs)
concat = keras.layers.Concatenate()([inputs, l1])
outputs = keras.layers.Dense(1, activation="linear")(concat)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.MSE, metrics=["mse"],
)
print(model.summary())

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2,)

# Datos y graficos
print("Precicison y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

np.save("history_ej5.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

plt.figure(figsize=(6, 5))
plt.plot(history.history["mse"], lw=lw, label="Entrenamiento")
plt.plot(history.history["val_mse"], lw=lw, label="Validación")
plt.title(r" $lr={}$ - $\lambda={}$".format(lr, lamd), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ5_mse.pdf")
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(x_test, model.predicth(x_test), lw=lw, label="Predicción")
plt.plot(x_test, y_test, lw=lw, label="Objetivo")
plt.title(
    r"Predicción sobre test - $lr={}$ - $\lambda={}$".format(lr, lamd), fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ5_preddción.pdf")
plt.show()
