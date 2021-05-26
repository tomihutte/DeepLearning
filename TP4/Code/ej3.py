import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns

####################################################
from sklearn.model_selection import train_test_split

####################################################

n_words = 10000
val_ratio = 0.2

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=n_words)


def vectorize_sequence(reviews_0, dimension=n_words):
    reviews = np.copy(reviews_0)
    ret = np.zeros(shape=(len(reviews), dimension))
    i = 0
    for i, review in enumerate(reviews):
        # mas información poner cuantas veces esta la palabra a que si solo esta o no
        words, counts = np.unique(review, return_counts=True)
        ret[i, words - 1] = counts
    return ret


# kk kk kk kk = train_test_split(x_train, test_size=0.2,stratify=y_train)
# kk kk kk kk = train_test_split(x_train, test_size=10000,stratify=y_train)

x_train = vectorize_sequence(x_train)
x_test = vectorize_sequence(x_test)

n_train = int(len(x_train) * (1 - val_ratio))

idx = np.arange(len(x_train))
np.random.shuffle(idx)
x_val = x_train[idx[n_train:]]
y_val = y_train[idx[n_train:]]
x_train = x_train[idx[:n_train]]
y_train = y_train[idx[:n_train]]


############################################################
# Ejercicio regularización L2
############################################################

lr = 1e-4
epoch = 50
lambd = 0.01
batch_size = 256
tr = 0.5

inputs = keras.Input(shape=x_train[0].shape)
l1 = keras.layers.Dense(100, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd))(inputs)
l2 = keras.layers.Dense(10, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd))(l1)
concat = keras.layers.Concatenate()([inputs, l2])
l3 = keras.layers.Dense(1, activation="linear", kernel_regularizer=keras.regularizers.l2(lambd))(concat)

model = keras.Model(inputs=inputs, outputs=l3)
model.summary()

# optimizer = keras.optimizers.Adam(learning_rate=lr)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(threshold=tr)],
)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=(x_val, y_val), verbose=2)

# Datos y graficos
print("Precicison y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

np.save("history_ej3_L2.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

# plt.figure(figsize=(8, 6))
plt.plot(
    history.history["binary_accuracy"], lw=lw, label="Entrenamiento",
)
plt.plot(history.history["val_binary_accuracy"], lw=lw, label="Validación")
plt.title(
    r"Regularización L2 - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_L2_acc.pdf")
plt.show()
print("Me gusta la poronga")

plt.plot(
    history.history["loss"] , lw=lw, label="Entrenamiento",
)
plt.plot(
    history.history["val_loss"] , lw=lw, label="Validación",
)
plt.title(
    r"Regularización L2 - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.yscale('log')
plt.ylim(0.2, 10)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_L2_loss.pdf")
plt.show()

############################################################
# Ejercicio batch normalization
############################################################
print("Me gusta la poronga")

lr = 1e-4
epoch = 50
lambd = 1
batch_size = 256
tr = 0.5

inputs = keras.Input(shape=x_train[0].shape)
l1 = keras.layers.Dense(100, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd))(inputs)
bn1 = keras.layers.BatchNormalization()(l1)
l2 = keras.layers.Dense(10, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd))(bn1)
bn2 = keras.layers.BatchNormalization()(l2)
concat = keras.layers.Concatenate()([inputs, bn2])
l3 = keras.layers.Dense(1, activation="linear", kernel_regularizer=keras.regularizers.l2(lambd))(concat)

model = keras.models.Model(inputs=inputs, outputs=l3)
model.summary()

# optimizer = keras.optimizers.Adam(learning_rate=lr)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(threshold=tr)],
)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=(x_val, y_val), verbose=2,)

# Datos y graficos

print("Precicison y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

np.save("history_ej3_BN.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25
print("Me gusta la poronga")

# plt.figure(figsize=(8, 6))
plt.plot(
    history.history["binary_accuracy"], lw=lw, label="Entrenamiento",
)
plt.plot(history.history["val_binary_accuracy"], lw=lw, label="Validación")
plt.title(
    r"Batch Normalization + L2 - $lr={}$ - $ \lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_BN_acc.pdf")
plt.show()

plt.plot(
    history.history["loss"], lw=lw, label="Entrenamiento",
)
plt.plot(
    history.history["val_loss"], lw=lw, label="Validación",
)
plt.title(
    r"Batch Normalization + L2 - $lr={}$ - $ \lambda={}$".format(lr, lambd), fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.yscale('log')
plt.ylim(0.2, 10)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_BN_loss.pdf")
plt.show()


############################################################
# Ejercicio dropout
############################################################
print("Me gusta la poronga")

lr = 1e-4
epoch = 50
lambd = 0.01
batch_size = 256
tr = 0.5
dropout_rate = 0.5

inputs = keras.Input(shape=x_train[0].shape)
l1 = keras.layers.Dense(100, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd))(inputs)
d1 = keras.layers.Dropout(rate=dropout_rate)(l1)
l2 = keras.layers.Dense(10, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd))(d1)
d2 = keras.layers.Dropout(rate=dropout_rate)(l2)
concat = keras.layers.Concatenate()([inputs, d2])
l3 = keras.layers.Dense(1, activation="linear", kernel_regularizer=keras.regularizers.l2(lambd))(concat)

model = keras.models.Model(inputs=inputs, outputs=l3)
model.summary()

# optimizer = keras.optimizers.Adam(learning_rate=lr)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(threshold=tr)],
)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=(x_val, y_val), verbose=2,)

# Datos y graficos
print("Precicison y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

np.save("history_ej3_DO.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

# plt.figure(figsize=(8, 6))
plt.plot(
    history.history["binary_accuracy"], lw=lw, label="Entrenamiento",
)
plt.plot(history.history["val_binary_accuracy"], lw=lw, label="Validación")
plt.title(
    r"Drop Out + L2 - $lr={}$ - drop rate = ${}$ - $\lambda={}$".format(lr, dropout_rate, lambd), fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_DO_acc.pdf")
plt.show()

plt.plot(
    history.history["loss"] , lw=lw, label="Entrenamiento",
)
plt.plot(
    history.history["val_loss"] , lw=lw, label="Validación",
)
plt.title(
    r"Drop Out + L2 - $lr={}$ - drop rate = ${}$ - $\lambda={}$".format(lr, dropout_rate, lambd), fontsize=fs,
)
plt.xlabel("Epoca", fontsize=fs)
plt.yscale('log')
plt.ylim(0.2, 10)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ3_DO_loss.pdf")
plt.show()

