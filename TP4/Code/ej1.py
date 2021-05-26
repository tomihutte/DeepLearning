import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns

seed = 42

boston = load_boston
X, Y = boston(return_X_y=True)


def data_split(x, y, val_ratio, seed=0):
    np.random.seed(seed)
    length = len(x)
    idx = np.arange(length)
    np.random.shuffle(idx)
    n_train = int(length * (1 - val_ratio))
    x_train = x[idx[:n_train]]
    y_train = y[idx[:, n_train]]
    x_val = x[idx[n_train:]]
    y_val = y[idx[n_train:]]
    # ipdb.set_trace()
    return x_train, y_train, x_val, y_val


# Preprocesamiento de los datos
val_ratio = 0.25

x_train, y_train, x_test, y_test = data_split(X, Y, val_ratio=val_ratio)


mean = x_train.mean(axis=0)
# norm = np.max(np.abs(x_train), axis=0)
norm = np.std(x_train, axis=0)

x_train = (x_train - mean) / norm
x_test = (x_test - mean) / norm


# Creamos el modelo
lr = 1e-3
lamd = 1e-4

optimizer = keras.optimizers.SGD(learning_rate=lr)
reg = keras.regularizers.l2(lamd)

input_shape = X.shape[1]
model = keras.Sequential(name="Regresion_Lineal")
model.add(
    keras.layers.Dense(1, input_shape=(input_shape,), activity_regularizer=reg)
)

model.compile(
    optimizer=optimizer, loss=keras.losses.MSE, metrics=["mse"],
)

# Entrenamiento
epochs = 200

history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=2,
)

y_pred = model.predict(x_test)

# Datos y graficos
np.save("history_ej1.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

house = np.arange(len(x_test))

plt.plot(y_test, y_test, label="Objetivo", lw=lw)
plt.scatter(y_test, y_pred, label="Predicción", s=s, c="C1")
plt.title(
    r"Ajuste sobre datos de validación-$lr = {}$-$\lambda={}$".format(lr, lamd),
    fontsize=fs,
)
plt.xlabel("Precio real [k$]", fontsize=fs)
plt.ylabel("Precio predicho [k$]", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ1_ajuste.pdf")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history["val_loss"], lw=lw)
# plt.hlines(
#     np.min(history.history["val_loss"]),
#     xmin=0,
#     xmax=epochs,
#     ls="--",
#     lw=lw,
# )
plt.title(
    r"Función de costo sobre datos de validación - $lr={}$ - $\lambda={}$".format(
        lr, lamd
    ),
    fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("MSE", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.tight_layout()
plt.savefig("EJ1_loss.pdf")
plt.show()

