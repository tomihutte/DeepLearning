import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import ipdb
import seaborn as sns
from sklearn.model_selection import train_test_split


n_words = 10000
review_lenght = 500
emb_dim = 32
epochs = 50
batch_size = 50
test_size = 5000
val_size = 5000


(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=n_words
)

# Preprocesamiento pad + division en val y test

x = np.hstack((x_train, x_test))
y = np.hstack((y_train, y_test))

x = keras.preprocessing.sequence.pad_sequences(x, maxlen=review_lenght)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, stratify=y
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=val_size, stratify=y_train
)


############################################################
# Red no convolucional
############################################################

lr = 1e-5
lambd = 1e-2
dropout_rate = 0.5


inputs = keras.Input(shape=x_train[0].shape)
embedding = keras.layers.Embedding(
    n_words, emb_dim, input_length=review_lenght, name="embedding_1"
)(inputs)
flat = keras.layers.Flatten(name="flatt_1")(embedding)
l1 = keras.layers.Dense(
    100, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd)
)(flat)
d1 = keras.layers.Dropout(rate=dropout_rate)(l1)
l2 = keras.layers.Dense(
    10, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd)
)(d1)
d2 = keras.layers.Dropout(rate=dropout_rate)(l2)
concat = keras.layers.Concatenate()([flat, d2])
l3 = keras.layers.Dense(
    1, activation="linear", kernel_regularizer=keras.regularizers.l2(lambd)
)(concat)
model1 = keras.Model(inputs=inputs, outputs=l3)
model1.compile(
    optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.BinaryCrossentropy(name="loss", from_logits=True),
    metrics=["acc"],
)
print(model1.summary())

history = model1.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2,
)

# Datos y graficos
print(
    "Precicison y loss sobre test = {}".format(model1.evaluate(x_test, y_test))
)

np.save("history_ej4_emb.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

plt.figure(figsize=(6, 5))
plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
plt.plot(history.history["val_acc"], lw=lw, label="Validación")
plt.title(r"Embedding - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ4_emb_acc.pdf")
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(
    history.history["loss"], lw=lw, label="Entrenamiento",
)
plt.plot(
    history.history["val_loss"], lw=lw, label="Validación",
)
plt.title(r"Embedding - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ4_emb_loss.pdf")
plt.show()


############################################################
# Red convolucional
############################################################

emb_dim = 16
epochs = 50
batch_size = 50
lr = 1e-5
lambd = 1
dropout_rate = 0.5

model = keras.Sequential()
model.add(keras.Input(shape=x_train[0].shape))
model.add(keras.layers.Embedding(n_words, emb_dim, input_length=review_lenght))
model.add(keras.layers.BatchNormalization())
model.add(
    keras.layers.Conv1D(
        filters=16,
        kernel_size=3,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(lambd),
        activation="relu",
    )
)
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.BatchNormalization())
model.add(
    keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(lambd),
        activation="relu",
    )
)
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(rate=dropout_rate))
model.add(keras.layers.Dense(1, activation="linear"))

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["acc"],
)

print(model.summary())

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=2,
)

# Datos y graficos
print(
    "Precicison y loss sobre test = {}".format(model1.evaluate(x_test, y_test))
)

np.save("history_ej4_conv.npy", history.history)

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25

plt.figure(figsize=(6, 5))
plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
plt.plot(history.history["val_acc"], lw=lw, label="Validación")
plt.title(
    r"CONV - Embdedding - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Precisión", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ4_conv_acc.pdf")
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(
    history.history["loss"], lw=lw, label="Entrenamiento",
)
plt.plot(
    history.history["val_loss"], lw=lw, label="Validación",
)
plt.title(
    r"CONV - Embedding - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs
)
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig("EJ4_conv_loss.pdf")
plt.show()
