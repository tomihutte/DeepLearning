def ej1():
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
    model.add(keras.layers.Dense(1, input_shape=(input_shape,), activity_regularizer=reg))

    model.compile(
        optimizer=optimizer, loss=keras.losses.MSE, metrics=["mse"],
    )

    # Entrenamiento
    epochs = 200

    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=2,)

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
        r"Ajuste sobre datos de validación-$lr = {}$-$\lambda={}$".format(lr, lamd), fontsize=fs,
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
        r"Función de costo sobre datos de validación - $lr={}$ - $\lambda={}$".format(lr, lamd), fontsize=fs,
    )
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("MSE", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.tight_layout()
    plt.savefig("EJ1_loss.pdf")
    plt.show()


def ej2():
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    from keras.datasets import cifar10
    import tensorflow as tf
    from tensorflow import keras
    import ipdb
    import seaborn as sns

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocesado

    x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:])).astype(np.float)

    x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:])).astype(np.float)

    y_test, y_train = y_test.ravel(), y_train.ravel()
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_test = (x_test - mean) / std
    x_train = (x_train - mean) / std

    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    # Modelos
    dim_input = x_train[0].shape

    ############################################################
    # Ejercicio 2-3
    ############################################################

    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    from keras.datasets import cifar10
    import tensorflow as tf
    from tensorflow import keras
    import ipdb
    import seaborn as sns

    n_clases = 10  # clasificaciones de los datos
    n_neuronas = 100  # neuronas de la capa intermedia
    epochs = 200  # epocas de entrenamiento
    batch_size = 50  # tamaño del batch
    lambd = 1e-4  # 1e-4  # factor de regularización
    lr = 1e-2  # learning rate
    weight = 1e-3  # pesos iniciales de las matrices

    model = keras.models.Sequential(name="ej3")

    model.add(keras.layers.Dense(n_neuronas, activation="sigmoid", activity_regularizer=keras.regularizers.l2(lambd), input_shape=dim_input,))

    model.add(keras.layers.Dense(n_clases, activation="linear", activity_regularizer=keras.regularizers.l2(lambd),))

    optimizer = keras.optimizers.SGD(learning_rate=lr)

    model.compile(
        optimizer=optimizer, loss=keras.losses.MSE, metrics=["acc"],
    )

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=2,)

    # Datos y graficos

    np.save("history_ej2_3.npy", history.history)

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    # plt.figure(figsize=(8, 6))
    plt.plot(
        history.history["val_loss"] / np.max(history.history["val_loss"]), lw=lw, label="Costo normalizado",
    )
    plt.plot(history.history["val_acc"], lw=lw, label="Precisión")
    plt.title(
        r"Activaciones Sigmoide+Lineal - Costo MSE - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
    )
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("MSE normalizado - Accuracy", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.savefig("EJ2_3_loss_acc.pdf")
    plt.show()

    ############################################################
    # Ejercicio 2-4
    ############################################################

    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    from keras.datasets import cifar10
    import tensorflow as tf
    from tensorflow import keras
    import ipdb
    import seaborn as sns

    model = keras.models.Sequential(name="ej4")

    n_clases = 10  # clasificaciones de los datos
    n_neuronas = 100  # neuronas de la capa intermedia
    epochs = 200  # epocas de entrenamiento
    batch_size = 50  # tamaño del batch
    lambd = 1e-6  # 1e-4  # factor de regularización
    lr = 1e-4  # learning rate
    weight = 1e-3  # pesos iniciales de las matrices

    model.add(keras.layers.Dense(n_neuronas, activation="sigmoid", activity_regularizer=keras.regularizers.l2(lambd), input_shape=dim_input,))

    model.add(keras.layers.Dense(n_clases, activation="linear", activity_regularizer=keras.regularizers.l2(lambd),))

    optimizer = keras.optimizers.SGD(learning_rate=lr)

    model.compile(
        optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["acc"],
    )

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=2,)

    # Datos y graficos

    np.save("history_ej2_4.npy", history.history)

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    # plt.figure(figsize=(8, 6))
    plt.plot(
        history.history["val_loss"] / np.max(history.history["val_loss"]), lw=lw, label="Costo normalizado",
    )
    plt.plot(history.history["val_acc"], lw=lw, label="Precisión")
    plt.title(
        r"Activaciones Sigmoide+Lineal - Costo CCE - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
    )
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("CCE normalizado - Accuracy", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.savefig("EJ2_4_loss_acc.pdf")
    plt.show()

    ############################################################
    # Ejercicio 2-6_1
    ############################################################
    print("Me gusta la poronga")

    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    from keras.datasets import cifar10
    import tensorflow as tf
    from tensorflow import keras
    import ipdb
    import seaborn as sns

    lr = 1e-1
    epochs = 1000
    tr = 0.9
    print("Me gusta la poronga")

    x_train = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    y_train = np.array([1, 0, 0, 1]).reshape(4, 1)

    # Modelo
    inputs = keras.Input(shape=x_train[0].shape)
    l1 = keras.layers.Dense(2, activation="tanh")(inputs)
    output = keras.layers.Dense(1, activation="tanh")(l1)

    model = keras.Model(inputs=inputs, outputs=output)

    optimizer = keras.optimizers.SGD(learning_rate=lr)

    model.compile(
        optimizer=optimizer, loss=keras.losses.MSE, metrics=[tf.keras.metrics.BinaryAccuracy(threshold=tr)],
    )

    history = model.fit(x_train, y_train, epochs=epochs, verbose=2)

    # Datos y graficos

    np.save("history_ej2_6.npy", history.history)

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    # plt.figure(figsize=(8, 6))
    plt.plot(
        history.history["loss"] / np.max(history.history["loss"]), lw=lw, label="Costo normalizado",
    )
    plt.plot(history.history["binary_accuracy"], lw=lw, label="Precisión")
    plt.title(
        r"Modelo 1 - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
    )
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("MSE normalizado - Accuracy", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ2_6_loss_acc.pdf")
    plt.show()

    print("Me gusta la poronga")
    ############################################################
    # Ejercicio 2-6_2
    ############################################################
    print("Me gusta la poronga")

    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    from keras.datasets import cifar10
    import tensorflow as tf
    from tensorflow import keras
    import ipdb
    import seaborn as sns

    lr = 1e-1
    epochs = 1000
    tr = 0.9

    x_train = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    y_train = np.array([1, 0, 0, 1]).reshape(4, 1)

    # Modelo
    inputs = keras.Input(shape=x_train[0].shape)
    l1 = keras.layers.Dense(1, activation="tanh")(inputs)
    l2 = keras.layers.Concatenate()([inputs, l1])
    output = keras.layers.Dense(1, activation="tanh")(l2)

    model = keras.Model(inputs=inputs, outputs=output)

    optimizer = keras.optimizers.SGD(learning_rate=lr)

    model.compile(
        optimizer=optimizer, loss=keras.losses.MSE, metrics=[tf.keras.metrics.BinaryAccuracy(threshold=tr)],
    )

    history = model.fit(x_train, y_train, epochs=epochs, verbose=2)

    # Datos y graficos

    np.save("history_ej2_6_2.npy", history.history)

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    # plt.figure(figsize=(8, 6))
    plt.plot(
        history.history["loss"] / np.max(history.history["loss"]), lw=lw, label="Costo normalizado",
    )
    plt.plot(history.history["binary_accuracy"], lw=lw, label="Precisión")
    plt.title(
        r"Modelo 2 - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
    )
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("MSE normalizado - Accuracy", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ2_6_2_loss_acc.pdf")
    plt.show()


def ej3():
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
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(
        history.history["val_loss"], lw=lw, label="Validación",
    )
    plt.title(
        r"Regularización L2 - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
    )
    plt.xlabel("Epoca", fontsize=fs)
    plt.yscale("log")
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
    plt.yscale("log")
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
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(
        history.history["val_loss"], lw=lw, label="Validación",
    )
    plt.title(
        r"Drop Out + L2 - $lr={}$ - drop rate = ${}$ - $\lambda={}$".format(lr, dropout_rate, lambd), fontsize=fs,
    )
    plt.xlabel("Epoca", fontsize=fs)
    plt.yscale("log")
    plt.ylim(0.2, 10)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ3_DO_loss.pdf")
    plt.show()


def ej4():
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

    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=n_words)

    # Preprocesamiento pad + division en val y test

    x = np.hstack((x_train, x_test))
    y = np.hstack((y_train, y_test))

    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=review_lenght)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, stratify=y_train)

    ############################################################
    # Red no convolucional
    ############################################################

    lr = 1e-5
    lambd = 1e-2
    dropout_rate = 0.5

    inputs = keras.Input(shape=x_train[0].shape)
    embedding = keras.layers.Embedding(n_words, emb_dim, input_length=review_lenght, name="embedding_1")(inputs)
    flat = keras.layers.Flatten(name="flatt_1")(embedding)
    l1 = keras.layers.Dense(100, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd))(flat)
    d1 = keras.layers.Dropout(rate=dropout_rate)(l1)
    l2 = keras.layers.Dense(10, activation="relu", kernel_regularizer=keras.regularizers.l2(lambd))(d1)
    d2 = keras.layers.Dropout(rate=dropout_rate)(l2)
    concat = keras.layers.Concatenate()([flat, d2])
    l3 = keras.layers.Dense(1, activation="linear", kernel_regularizer=keras.regularizers.l2(lambd))(concat)
    model1 = keras.Model(inputs=inputs, outputs=l3)
    model1.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.BinaryCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )
    print(model1.summary())

    history = model1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2,)

    # Datos y graficos
    print("Precicison y loss sobre test = {}".format(model1.evaluate(x_test, y_test)))

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
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", kernel_regularizer=keras.regularizers.l2(lambd), activation="relu",))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", kernel_regularizer=keras.regularizers.l2(lambd), activation="relu",))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(1, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=["acc"],
    )

    print(model.summary())

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=2,)

    # Datos y graficos
    print("Precicison y loss sobre test = {}".format(model1.evaluate(x_test, y_test)))

    np.save("history_ej4_conv.npy", history.history)

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    plt.figure(figsize=(6, 5))
    plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
    plt.plot(history.history["val_acc"], lw=lw, label="Validación")
    plt.title(r"CONV - Embdedding - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs)
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
    plt.title(r"CONV - Embedding - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ4_conv_loss.pdf")
    plt.show()


def ej5():
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


def ej6():
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
    model.add(keras.layers.Dense(n1, kernel_regularizer=keras.regularizers.l2(lambd), activation="relu", input_shape=x[0].shape,))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(rate=drop_rate))
    model.add(keras.layers.Dense(n2, kernel_regularizer=keras.regularizers.l2(lambd), activation="relu"))
    model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(rate=drop_rate))
    model.add(keras.layers.Dense(1, activation="linear", kernel_regularizer=keras.regularizers.l2(lambd)))

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
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

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
    plt.fill_between(epochs, acc_val_low, acc_val_high, alpha=0.35, label="Validación variación")
    plt.plot(acc_val_mean, ls="--", lw=lw, label="Validacion")
    plt.ylim(0, 1)
    plt.xlabel("Epocas", fontsize=fs)
    plt.ylabel(r"Precisión ", fontsize=fs)
    plt.title(
        r"Datos sin corrección - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
    )
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("ej6_sin_val_acc.pdf")
    plt.show()

    ################# Train Acc #########################
    plt.figure(figsize=(7, 6))
    plt.fill_between(
        epochs, acc_train_low, acc_train_high, alpha=0.35, label="Entrenamiento variación",
    )
    plt.plot(acc_train_mean, ls="--", lw=lw, label="Entrenamiento")
    plt.ylim(0, 1)
    plt.xlabel("Epocas", fontsize=fs)
    plt.ylabel(r"Precisión ", fontsize=fs)
    plt.title(
        r"Datos sin corrección - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
    )
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("ej6_sin_train_acc.pdf")
    plt.show()

    ################# Val Loss #########################

    plt.figure(figsize=(7, 6))
    plt.fill_between(
        epochs, loss_val_low, loss_val_high, alpha=0.35, label="Validación variación",
    )
    plt.plot(loss_val_mean, ls="--", lw=lw, label="Validacion")
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=fs)
    plt.title(
        r"Datos sin corrección - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
    )
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("ej6_sin_val_loss.pdf")
    plt.show()

    ################# Train Loss #########################

    plt.figure(figsize=(7, 6))
    plt.fill_between(
        epochs, loss_train_low, loss_train_high, alpha=0.35, label="Entrenamiento variación",
    )
    plt.plot(loss_train_mean, ls="--", lw=lw, label="Entrenamiento")
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=fs)
    plt.title(
        r"Datos sin corrección - $lr={}$ - $\lambda={}$".format(lr, lambd), fontsize=fs,
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
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

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
    plt.fill_between(epochs, acc_val_low, acc_val_high, alpha=0.35, label="Validación variación")
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
        epochs, acc_train_low, acc_train_high, alpha=0.35, label="Entrenamiento variación",
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
        epochs, loss_val_low, loss_val_high, alpha=0.35, label="Validación variación",
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
        epochs, loss_train_low, loss_train_high, alpha=0.35, label="Entrenamiento variación",
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


def ej7():
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

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=2,)

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


def ej8():
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    import tensorflow as tf
    from tensorflow import keras
    import seaborn as sns
    import sklearn
    from sklearn.model_selection import train_test_split

    (x_train_0, y_train_0), (x_test_0, y_test_0) = keras.datasets.mnist.load_data()

    ## Preprocesado
    x_train = (x_train_0.astype(np.float32) / 255).reshape(len(x_train_0), np.prod(x_train_0[0].shape))
    x_test = (x_test_0.astype(np.float32) / 255).reshape(len(x_test_0), np.prod(x_test_0[0].shape))

    y_test = keras.utils.to_categorical(y_test_0)
    y_train = keras.utils.to_categorical(y_train_0)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000)

    #### Red de capas densas

    # Parametros
    lr = 1e-5
    lamb = 1e-1
    batch_size = 64
    epochs = 100
    n_1 = 20
    n_2 = 10

    model = keras.models.Sequential(name="EJ8_Densa")
    model.add(
        keras.layers.Dense(n_1, activation="relu", kernel_regularizer=keras.regularizers.l2(lamb), name="dense_1", input_shape=x_train[0].shape,)
    )
    model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
    model.add(keras.layers.Dense(n_2, activation="relu", kernel_regularizer=keras.regularizers.l2(lamb), name="dense_2"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
    model.add(keras.layers.Dense(10, activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense_3"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True),
        metrics=["acc"],  # keras.metrics.BinaryAccuracy(threshold=0.5)],
    )

    print(model.summary())

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

    # Datos y graficos
    print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    plt.figure(figsize=(6, 5))
    plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
    plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
    plt.title(r"Capas densas - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ8_dense_acc.pdf")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
    plt.title(r"Capas densas - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ8_dense_loss.pdf")
    plt.show()

    ###### Red convolucional
    ## Preprocesado
    x_test = x_test.reshape(len(x_test), 28, 28, 1)
    x_train = (x_train_0.astype(np.float32) / 255)[:, :, :, np.newaxis]
    x_val = (x_test_0.astype(np.float32) / 255)[:, :, :, np.newaxis]
    y_val = keras.utils.to_categorical(y_test_0)
    y_train = keras.utils.to_categorical(y_train_0)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000)

    # Parametros
    lr = 1e-5
    lamb = 1e-1
    batch_size = 64
    epochs = 100
    n_1 = 32
    n_2 = 16

    model = keras.models.Sequential(name="EJ8_Conv")
    model.add(
        keras.layers.Conv2D(
            filters=n_1,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=keras.regularizers.l2(lamb),
            name="conv_1",
            input_shape=x_train[0].shape,
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
    model.add(
        keras.layers.Conv2D(
            filters=n_2, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=keras.regularizers.l2(lamb), name="conv_2",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True),
        metrics=["acc"],  # keras.metrics.BinaryAccuracy(threshold=0.5)],
    )

    print(model.summary())

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

    # Datos y graficos
    print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    plt.figure(figsize=(6, 5))
    plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
    plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
    plt.title(r"Red Convolucional - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ8_conv_acc.pdf")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
    plt.title(r"Red Convolucional  - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ8_conv_loss.pdf")
    plt.show()


def ej9():
    #### EJErcicio 9 ####

    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    import tensorflow as tf
    from tensorflow import keras
    import seaborn as sns
    import sklearn
    from sklearn.model_selection import train_test_split

    (x_train, y_train_0), (x_test, y_test_0) = keras.datasets.mnist.load_data()

    # PERMUTACION
    permutation = np.random.permutation(28 * 28)
    x_train_perm = x_train.reshape(x_train.shape[0], -1)
    x_train_perm = x_train_perm[:, permutation]
    x_train_perm = x_train_perm.reshape(x_train.shape)
    x_test_perm = x_test.reshape(x_test.shape[0], -1)
    x_test_perm = x_test_perm[:, permutation]
    x_test_perm = x_test_perm.reshape(x_test.shape)

    x_train_0 = x_train_perm  ## uso _0 para no sobreescribir
    x_test_0 = x_test_perm  ## porque la convolucional lleva otro preprocesado

    #### Red de capas densas
    ## Preprocesado
    x_train = (x_train_0.astype(np.float32) / 255).reshape(len(x_train_0), np.prod(x_train_0[0].shape))
    x_test = (x_test_0.astype(np.float32) / 255).reshape(len(x_test_0), np.prod(x_test_0[0].shape))

    y_test = keras.utils.to_categorical(y_test_0)
    y_train = keras.utils.to_categorical(y_train_0)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000)

    # Parametros
    lr = 1e-5
    lamb = 1e-1
    batch_size = 64
    epochs = 100
    n_1 = 20
    n_2 = 10

    model = keras.models.Sequential(name="EJ9_Densa")
    model.add(
        keras.layers.Dense(n_1, activation="relu", kernel_regularizer=keras.regularizers.l2(lamb), name="dense_1", input_shape=x_train[0].shape,)
    )
    model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
    model.add(keras.layers.Dense(n_2, activation="relu", kernel_regularizer=keras.regularizers.l2(lamb), name="dense_2"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
    model.add(keras.layers.Dense(10, activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense_3"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True),
        metrics=["acc"],  # keras.metrics.BinaryAccuracy(threshold=0.5)],
    )

    print(model.summary())

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

    # Datos y graficos
    print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    plt.figure(figsize=(6, 5))
    plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
    plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
    plt.title(r"Capas densas - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ9_dense_acc.pdf")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
    plt.title(r"Capas densas - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ9_dense_loss.pdf")
    plt.show()

    ###### Red convolucional
    ## Preprocesado
    x_test = x_test.reshape(len(x_test), 28, 28, 1)
    x_train = (x_train_0.astype(np.float32) / 255)[:, :, :, np.newaxis]
    x_val = (x_test_0.astype(np.float32) / 255)[:, :, :, np.newaxis]
    y_val = keras.utils.to_categorical(y_test_0)
    y_train = keras.utils.to_categorical(y_train_0)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000)

    # Parametros
    lr = 1e-5
    lamb = 1e-1
    batch_size = 64
    epochs = 100
    n_1 = 32
    n_2 = 16

    model = keras.models.Sequential(name="EJ9_Conv")
    model.add(
        keras.layers.Conv2D(
            filters=n_1,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=keras.regularizers.l2(lamb),
            name="conv_1",
            input_shape=x_train[0].shape,
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
    model.add(
        keras.layers.Conv2D(
            filters=n_2, kernel_size=(3, 3), activation="relu", padding="same", kernel_regularizer=keras.regularizers.l2(lamb), name="conv_2",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True),
        metrics=["acc"],  # keras.metrics.BinaryAccuracy(threshold=0.5)],
    )

    print(model.summary())

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

    # Datos y graficos
    print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    plt.figure(figsize=(6, 5))
    plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
    plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
    plt.title(r"Red Convolucional - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ9_conv_acc.pdf")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
    plt.title(r"Red Convolucional  - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ9_conv_loss.pdf")
    plt.show()


def ej10AlexNet():
    #### Ejercicio 10 Alex Net#####

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

    model = keras.Sequential(name="AlexNet")
    model.add(
        keras.layers.Conv2D(
            filters=96,
            kernel_size=(4, 4),  ## salida de 15x15x8
            strides=2,
            padding="valid",
            kernel_regularizer=reg,
            activation="relu",
            input_shape=(x_train[0].shape),
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1))  ## salida de 13x13x8

    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.Conv2D(56, kernel_size=(3, 3), strides=2, padding="valid", kernel_regularizer=reg, activation="relu",)
    )  ## salida de 6x6x16

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1))  ## salida de 5x5x8

    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.Conv2D(384, kernel_size=(3, 3), strides=1, padding="same", kernel_regularizer=reg, activation="relu",)
    )  ## salida de 5x5x16
    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.Conv2D(384, kernel_size=(3, 3), strides=1, padding="same", kernel_regularizer=reg, activation="relu",)
    )  ## salida de 5x5x16
    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding="same", kernel_regularizer=reg, activation="relu",)
    )  ## salida de 5x5x16

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1))  ## salida de 4x4x16

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())  ## tamaño de 256

    model.add(keras.layers.Dense(512, kernel_regularizer=reg, activation="relu"))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, kernel_regularizer=reg, activation="relu"))
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
    plt.title(r"AlexNet - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ10_AlexNet_cifar10_acc.pdf")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
    plt.title(r"AlexNet - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ10_AlexNet_cifar10_loss.pdf")
    plt.show()


def ej10VGG16():
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
