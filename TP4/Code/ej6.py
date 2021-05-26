import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

# Parametros
k = 5  # k-folding
n1 = 24
n2 = 12
weights = "w.h5"
lambd = 0.1
lr = 1e-4
drop_rate = 0.5
test_size = 0.1
batch_size = 16
epochs = 100

############################################################
# Red K-Fold
############################################################

# Carga de datos
data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
x_0, y_0 = data[:, :-1], data[:, -1].reshape(data.shape[0], 1)

x, x_test, y, y_test = train_test_split(x_0, y_0, test_size=0.1)


# Arquitectura de la red

model = keras.models.Sequential(name="EJ6_folding")
model.add(
    keras.layers.Dense(
        n1,
        kernel_regularizer=keras.regularizers.l2(lambd),
        activation="relu",
        input_shape=x[0].shape,
    )
)
model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dropout(rate=drop_rate))
model.add(
    keras.layers.Dense(
        n2, kernel_regularizer=keras.regularizers.l2(lambd), activation="relu"
    )
)
model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dropout(rate=drop_rate))
model.add(
    keras.layers.Dense(
        1, activation="linear", kernel_regularizer=keras.regularizers.l2(lambd)
    )
)

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.BinaryCrossentropy(name="loss", from_logits=True),
    metrics=["acc"],  # keras.metrics.BinaryAccuracy(threshold=0.5)],
)

print(model.summary())

model.save_weights(weights)

# K- Folding con datos sin corregir

fold = sklearn.model_selection.KFold(n_splits=k)

acc_test = np.array([])
acc_train = np.array([])
loss_test = np.array([])
loss_train = np.array([])
prediction = 0

for train_index, val_index in fold.split(x):
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Cargo pesos aleatorios
    model.load_weights(weights)

    # Entrenamiento
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
    )

    acc_test = np.concatenate((acc_test, history.history["val_acc"]))
    acc_train = np.concatenate((acc_train, history.history["acc"]))

    loss_test = np.concatenate((loss_test, history.history["loss"]))
    loss_train = np.concatenate((loss_train, history.history["val_loss"]))

    prediction += model.predict(x_test)

prediction /= k

acc_test = acc_test.reshape((k, epochs))
acc_train = acc_train.reshape((k, epochs))
loss_test = loss_test.reshape((k, epochs))
loss_train = loss_train.reshape((k, epochs))

acc_val_mean = np.mean(acc_test, axis=0)
acc_val_low = np.min(acc_test, axis=0)
acc_val_high = np.max(acc_test, axis=0)

acc_train_low = np.min(acc_train, axis=0)
acc_train_mean = np.mean(acc_train, axis=0)
acc_train_high = np.max(acc_train, axis=0)

loss_val_mean = np.mean(loss_test, axis=0)
loss_val_low = np.min(loss_test, axis=0)
loss_val_high = np.max(loss_test, axis=0)

loss_train_low = np.min(loss_train, axis=0)
loss_train_mean = np.mean(loss_train, axis=0)
loss_train_high = np.max(loss_train, axis=0)

epochs = np.arange(epochs)

fs = 16
lw = 3

sns.set(style="whitegrid")

# Grafico
################# Val Acc #########################
plt.figure(figsize=(7, 6))
plt.fill_between(
    epochs, acc_val_low, acc_val_high, alpha=0.35, label="Validación variación"
)
plt.plot(acc_val_mean, ls="--", lw=lw, label="Validacion")
plt.ylim(0, 1)
plt.xlabel("Epocas", fontsize=fs)
plt.ylabel(r"Precisión ", fontsize=fs)
plt.title(
    r"Datos sin corrección - $lr={}$ - $\lambda={}$".format(lr, lambd),
    fontsize=fs,
)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("ej6_sin_val_acc.pdf")
plt.show()

################# Train Acc #########################
plt.figure(figsize=(7, 6))
plt.fill_between(
    epochs,
    acc_train_low,
    acc_train_high,
    alpha=0.35,
    label="Entrenamiento variación",
)
plt.plot(acc_train_mean, ls="--", lw=lw, label="Entrenamiento")
plt.ylim(0, 1)
plt.xlabel("Epocas", fontsize=fs)
plt.ylabel(r"Precisión ", fontsize=fs)
plt.title(
    r"Datos sin corrección - $lr={}$ - $\lambda={}$".format(lr, lambd),
    fontsize=fs,
)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("ej6_sin_train_acc.pdf")
plt.show()

################# Val Loss #########################

plt.figure(figsize=(7, 6))
plt.fill_between(
    epochs,
    loss_val_low,
    loss_val_high,
    alpha=0.35,
    label="Validación variación",
)
plt.plot(loss_val_mean, ls="--", lw=lw, label="Validacion")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=fs)
plt.title(
    r"Datos sin corrección - $lr={}$ - $\lambda={}$".format(lr, lambd),
    fontsize=fs,
)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("ej6_sin_val_loss.pdf")
plt.show()

################# Train Loss #########################

plt.figure(figsize=(7, 6))
plt.fill_between(
    epochs,
    loss_train_low,
    loss_train_high,
    alpha=0.35,
    label="Entrenamiento variación",
)
plt.plot(loss_train_mean, ls="--", lw=lw, label="Entrenamiento")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=fs)
plt.title(
    r"Datos sin corrección - $lr={}$ - $\lambda={}$".format(lr, lambd),
    fontsize=fs,
)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("ej6_sin_train_loss.pdf")
plt.show()


########## Correccion de datos ##############
########## Despues uso misma arquitectura ###
z = np.zeros(x_0.shape)
z += x_0.mean(axis=0)
z[x_0 != 0] = 0
z[:, 0] = 0
x_0 += z

x, x_test, y, y_test = train_test_split(x_0, y_0, test_size=0.1)

# K- Folding con datos sin corregir
epochs = 100

fold = sklearn.model_selection.KFold(n_splits=k)

acc_test = np.array([])
acc_train = np.array([])
loss_test = np.array([])
loss_train = np.array([])
prediction = 0

for train_index, val_index in fold.split(x):
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Cargo pesos aleatorios
    model.load_weights(weights)

    # Entrenamiento
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
    )

    acc_test = np.concatenate((acc_test, history.history["val_acc"]))
    acc_train = np.concatenate((acc_train, history.history["acc"]))

    loss_test = np.concatenate((loss_test, history.history["loss"]))
    loss_train = np.concatenate((loss_train, history.history["val_loss"]))

    prediction += model.predict(x_test)

prediction /= k

acc_test = acc_test.reshape((k, epochs))
acc_train = acc_train.reshape((k, epochs))
loss_test = loss_test.reshape((k, epochs))
loss_train = loss_train.reshape((k, epochs))

acc_val_mean = np.mean(acc_test, axis=0)
acc_val_low = np.min(acc_test, axis=0)
acc_val_high = np.max(acc_test, axis=0)

acc_train_low = np.min(acc_train, axis=0)
acc_train_mean = np.mean(acc_train, axis=0)
acc_train_high = np.max(acc_train, axis=0)

loss_val_mean = np.mean(loss_test, axis=0)
loss_val_low = np.min(loss_test, axis=0)
loss_val_high = np.max(loss_test, axis=0)

loss_train_low = np.min(loss_train, axis=0)
loss_train_mean = np.mean(loss_train, axis=0)
loss_train_high = np.max(loss_train, axis=0)

epochs = np.arange(epochs)

fs = 16
lw = 3

sns.set(style="whitegrid")

# Grafico
################# Val Acc #########################
plt.figure(figsize=(7, 6))
plt.fill_between(
    epochs, acc_val_low, acc_val_high, alpha=0.35, label="Validación variación"
)
plt.plot(acc_val_mean, ls="--", lw=lw, label="Validacion")
plt.ylim(0, 1)
plt.xlabel("Epocas", fontsize=fs)
plt.ylabel(r"Precisión ", fontsize=fs)
plt.title(
    r"Datos corregidos - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("ej6_con_val_acc.pdf")
plt.show()

################# Train Acc #########################
plt.figure(figsize=(7, 6))
plt.fill_between(
    epochs,
    acc_train_low,
    acc_train_high,
    alpha=0.35,
    label="Entrenamiento variación",
)
plt.plot(acc_train_mean, ls="--", lw=lw, label="Entrenamiento")
plt.ylim(0, 1)
plt.xlabel("Epocas", fontsize=fs)
plt.ylabel(r"Precisión ", fontsize=fs)
plt.title(
    r"Datos corregidos - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("ej6_con_train_acc.pdf")
plt.show()

################# Val Loss #########################

plt.figure(figsize=(7, 6))
plt.fill_between(
    epochs,
    loss_val_low,
    loss_val_high,
    alpha=0.35,
    label="Validación variación",
)
plt.plot(loss_val_mean, ls="--", lw=lw, label="Validacion")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=fs)
plt.title(
    r"Datos corregidos - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("ej6_con_val_loss.pdf")
plt.show()

################# Train Loss #########################

plt.figure(figsize=(7, 6))
plt.fill_between(
    epochs,
    loss_train_low,
    loss_train_high,
    alpha=0.35,
    label="Entrenamiento variación",
)
plt.plot(loss_train_mean, ls="--", lw=lw, label="Entrenamiento")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=fs)
plt.title(
    r"Datos corregidos - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("ej6_con_train_loss.pdf")
plt.show()
