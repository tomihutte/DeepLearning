def ej3_letters():
    ##### Conv Ej 8 - EMNIST a MNIST transfer learning #########
    ###### Ejercicio 3 #######

    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    import tensorflow as tf
    from tensorflow import keras
    import seaborn as sns
    import sklearn
    from sklearn.model_selection import train_test_split

    # Voy a entrenar una red sobre EMNIST letters y despues transfiero a MNIST
    fig_size = (7, 6)
    n_train = 100  # datos para entrenar con MNIST

    # Cargo los datos, es necesario bajarlos desde https://www.kaggle.com/crawford/emnist?select=emnist-letters-train.csv

    data_train = np.loadtxt("emnist-letters-train.csv", delimiter=",")
    data_test = np.loadtxt("emnist-letters-test.csv", delimiter=",")

    data = np.vstack((data_train, data_test))

    x = data[:, 1:].reshape(len(data), 28, 28, 1)
    y = to_categorical(data[:, 0])[:, 1:]

    # Preprocesado
    x /= 255.0

    # Division val y test
    t_size = 10000
    v_size = 10000

    x, x_test, y, y_test = train_test_split(x, y, test_size=t_size, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=v_size, stratify=y)

    ###### Red convolucional del ej 8 del tp 4 #########
    ###### Entrenada con  EMNIST letters ################
    # Parametros
    lr = 1e-4
    lamb = 1e-2
    batch_size = 100
    epochs = 50
    n_1 = 32
    n_2 = 16

    model = keras.models.Sequential(name="TomiNet")
    model.add(keras.layers.Conv2D(filters=n_1, kernel_size=(3, 3), activation="relu", padding="same", name="conv_1", input_shape=x_train[0].shape,))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
    model.add(keras.layers.Conv2D(filters=n_2, kernel_size=(3, 3), activation="relu", padding="same", name="conv_2",))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(y_train.shape[1], activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )

    print(model.summary())

    # No hago data augmentation porque tengo una banda

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

    # Datos y graficos
    print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    plt.figure(figsize=fig_size)
    plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
    plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
    plt.title(r"EMNIST letters - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    # plt.savefig("EJ3_mnist_letters_acc.pdf")
    plt.show()

    plt.figure(figsize=fig_size)
    plt.plot(
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
    plt.title(r"EMNIST letters  - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    # plt.savefig("EJ3_mnist_letters_loss.pdf")
    plt.show()

    ## Guardo los datos para usarlos ahora
    model.save_weights("EMNIST")
    model.save("TomiNet.h5")

    ### Cargamos los datos de MNIST
    v_size = 10000
    t_size = 10000

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x = np.vstack((x_train, x_test))
    y = np.append(y_train, y_test)

    # Preprocesado
    x = x / 255.0
    x = x.reshape(len(x), 28, 28, 1)
    y = to_categorical(y)

    x, x_test, y, y_test = train_test_split(x, y, test_size=t_size, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=v_size, stratify=y_train)

    # Resolvamos el problema sin usar transfer learning
    # Uso el mismo modelo entrenado con EMNIST letters
    lr = 1e-3
    lamb = 1e-3
    epochs = 50
    batch_size = int(n_train / 10)

    # No se como cargar el modelo que guarde pero con los pesos random asi que lo creo de nuevo

    model = keras.models.Sequential(name="TomiNet2")
    model.add(keras.layers.Conv2D(filters=n_1, kernel_size=(3, 3), activation="relu", padding="same", name="conv_1", input_shape=x_train[0].shape,))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
    model.add(keras.layers.Conv2D(filters=n_2, kernel_size=(3, 3), activation="relu", padding="same", name="conv_2",))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(y_train.shape[1], activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )

    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )

    # Parametros

    epochs = 50

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
    plt.title(r"MNIST- Sin TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    # plt.savefig("EJ3_emnist_mnist_noTL_acc.pdf")
    plt.show()

    plt.figure(figsize=fig_size)
    plt.plot(
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
    plt.title(r"MNIST- Sin TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    # plt.savefig("EJ3_emnist_mnist_noTL_loss.pdf")
    plt.show()

    ########## Vamos a clasificar MNIST con learning-transfer ##############
    # Cargo el modelo, me quedo con las capas convolucionales nomas
    model_emnist = keras.models.load_model("TomiNet.h5")
    model_emnist = keras.Model(inputs=model_emnist.inputs, outputs=model_emnist.layers[-2].output)
    model_emnist.load_weights("EMNIST")

    model_emnist.trainable = False

    ## Uso el modelo con MNIST

    inputs = keras.Input(shape=x_train[0].shape)
    x = model_emnist(inputs)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(y_train.shape[1])(x)
    model = keras.Model(inputs, out)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )

    ## Cuantos datos de train queremos usar, para probar que tan bien anda la cosa

    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=45, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)

    epochs = 35

    history = model.fit(
        datagen.flow(x_train[:n_train], y_train[:n_train], batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=n_train / batch_size,
        validation_data=(x_val, y_val),
        verbose=2,
    )

    # Ahora hago el fine tunning
    lr_factor = 10
    epochs_ft = 15

    model_emnist.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr / lr_factor), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
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

    plt.figure(figsize=fig_size)
    plt.plot(np.append(history.history["acc"], history2.history["acc"]), lw=lw, label="Entrenamiento")
    plt.plot(np.append(history.history["val_acc"], history2.history["val_acc"]), lw=lw, label="Validación", ls="--")
    plt.vlines(x=epochs, ymin=0, ymax=1, ls="--")
    plt.title(r"MNIST- Letters TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    # plt.savefig("EJ3_emnist_mnist_acc.pdf")
    plt.show()

    plt.figure(figsize=fig_size)
    plt.plot(
        np.append(history.history["loss"], history2.history["loss"]), lw=lw, label="Entrenamiento",
    )
    plt.plot(np.append(history.history["val_loss"], history2.history["val_loss"]), lw=lw, label="Validación", ls="--")
    plt.title(r"MNIST- Letters TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
    (ymin, ymax) = plt.ylim()
    plt.vlines(x=epochs, ymin=ymin, ymax=ymax, ls="--")
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    # plt.savefig("EJ3_emnist_mnist_loss.pdf")
    plt.show()


def ej3_fashion():
    ##### Conv Ej 8 - Fashion MNITS a MNIST transfer learning #########
    ###### Ejercicio 3 #######

    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    import tensorflow as tf
    from tensorflow import keras
    import seaborn as sns
    import sklearn
    from sklearn.model_selection import train_test_split

    # Voy a entrenar una red sobre fashion MNIST y despues transfiero a MNIST
    fig_size = (7, 6)
    n_train = 100  # datos para entrenar con MNIST

    # Cargo los datos, es necesario bajarlos desde https://www.kaggle.com/zalando-research/fashionmnist

    data_train = np.loadtxt("fashion-mnist_train.csv", delimiter=",", skiprows=1)
    data_test = np.loadtxt("fashion-mnist_test.csv", delimiter=",", skiprows=1)

    data = np.vstack((data_train, data_test))

    x = data[:, 1:].reshape(len(data), 28, 28, 1)
    y = to_categorical(data[:, 0])[:, 1:]

    # Preprocesado
    x /= 255.0

    # Division val y test
    t_size = 10000
    v_size = 10000

    x, x_test, y, y_test = train_test_split(x, y, test_size=t_size, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=v_size, stratify=y)

    ###### Red convolucional del ej 8 del tp 4 #########
    ###### Entrenada con fashion MNIST ################
    # Parametros
    lr = 1e-4
    lamb = 1e-2
    batch_size = 100
    epochs = 50
    n_1 = 32
    n_2 = 16

    model = keras.models.Sequential(name="TomiNet")
    model.add(keras.layers.Conv2D(filters=n_1, kernel_size=(3, 3), activation="relu", padding="same", name="conv_1", input_shape=x_train[0].shape,))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
    model.add(keras.layers.Conv2D(filters=n_2, kernel_size=(3, 3), activation="relu", padding="same", name="conv_2",))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(y_train.shape[1], activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )

    print(model.summary())

    # No hago data augmentation porque tengo una banda

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

    # Datos y graficos
    print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))

    sns.set_style("whitegrid")
    fs = 16
    lw = 3
    s = 25

    plt.figure(figsize=fig_size)
    plt.plot(history.history["acc"], lw=lw, label="Entrenamiento")
    plt.plot(history.history["val_acc"], lw=lw, label="Validación", ls="--")
    plt.title(r"Fashion MNIST - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ3_fashion_acc.pdf")
    plt.show()

    plt.figure(figsize=fig_size)
    plt.plot(
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
    plt.title(r"Fashion MNIST  - $lr={}$ - $\lambda={}$".format(lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ3_fashion_loss.pdf")
    plt.show()

    ## Guardo los datos para usarlos ahora
    model.save_weights("fashionMNIST")
    model.save("TomiNet.h5")

    ### Cargamos los datos de MNIST
    v_size = 10000
    t_size = 10000

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x = np.vstack((x_train, x_test))
    y = np.append(y_train, y_test)

    # Preprocesado
    x = x / 255.0
    x = x.reshape(len(x), 28, 28, 1)
    y = to_categorical(y)

    x, x_test, y, y_test = train_test_split(x, y, test_size=t_size, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=v_size, stratify=y_train)

    # Resolvamos el problema sin usar transfer learning
    # Uso el mismo modelo entrenado con fashion MNIST
    lr = 1e-3
    lamb = 1e-3
    epochs = 50
    batch_size = int(n_train / 10)

    # No se como cargar el modelo que guarde pero con los pesos random asi que lo creo de nuevo

    model = keras.models.Sequential(name="TomiNet2")
    model.add(keras.layers.Conv2D(filters=n_1, kernel_size=(3, 3), activation="relu", padding="same", name="conv_1", input_shape=x_train[0].shape,))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
    model.add(keras.layers.Conv2D(filters=n_2, kernel_size=(3, 3), activation="relu", padding="same", name="conv_2",))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(y_train.shape[1], activation="tanh", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )

    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )

    # Parametros

    epochs = 50

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
    plt.title(r"MNIST- Sin TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ3_fashion_mnist_noTL_acc.pdf")
    plt.show()

    plt.figure(figsize=fig_size)
    plt.plot(
        history.history["loss"], lw=lw, label="Entrenamiento",
    )
    plt.plot(history.history["val_loss"], lw=lw, label="Validación", ls="--")
    plt.title(r"MNIST- Sin TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ3_fashion_mnist_noTL_loss.pdf")
    plt.show()

    ########## Vamos a clasificar MNIST con learning-transfer ##############
    # Cargo el modelo, me quedo con las capas convolucionales nomas
    model_emnist = keras.models.load_model("TomiNet.h5")
    model_emnist = keras.Model(inputs=model_emnist.inputs, outputs=model_emnist.layers[-2].output)
    model_emnist.load_weights("fashionMNIST")

    model_emnist.trainable = False

    ## Uso el modelo con MNIST

    inputs = keras.Input(shape=x_train[0].shape)
    x = model_emnist(inputs)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(y_train.shape[1])(x)
    model = keras.Model(inputs, out)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )

    ## Cuantos datos de train queremos usar, para probar que tan bien anda la cosa

    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=45, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)

    epochs = 35

    history = model.fit(
        datagen.flow(x_train[:n_train], y_train[:n_train], batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=n_train / batch_size,
        validation_data=(x_val, y_val),
        verbose=2,
    )

    # Ahora hago el fine tunning
    lr_factor = 10
    epochs_ft = 15

    model_emnist.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr / lr_factor), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
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

    plt.figure(figsize=fig_size)
    plt.plot(np.append(history.history["acc"], history2.history["acc"]), lw=lw, label="Entrenamiento")
    plt.plot(np.append(history.history["val_acc"], history2.history["val_acc"]), lw=lw, label="Validación", ls="--")
    plt.vlines(x=epochs, ymin=0, ymax=1, ls="--")
    plt.title(r"MNIST- Fashion TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs - 1)
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Precisión", fontsize=fs)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ3_fashion_mnist_acc.pdf")
    plt.show()

    plt.figure(figsize=fig_size)
    plt.plot(
        np.append(history.history["loss"], history2.history["loss"]), lw=lw, label="Entrenamiento",
    )
    plt.plot(np.append(history.history["val_loss"], history2.history["val_loss"]), lw=lw, label="Validación", ls="--")
    plt.title(r"MNIST- Fashion TL - n_train={} - $lr={}$ - $\lambda={}$".format(n_train, lr, lamb), fontsize=fs - 1)
    (ymin, ymax) = plt.ylim()
    plt.vlines(x=epochs, ymin=ymin, ymax=ymax, ls="--")
    plt.xlabel("Epoca", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig("EJ3_fashion_mnist_loss.pdf")
    plt.show()


def ej4():
    ############ Entrenando red en MNIST para ej4 ##############
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    import tensorflow as tf
    from tensorflow import keras
    import seaborn as sns
    import sklearn
    from sklearn.model_selection import train_test_split

    #### Primero entrenamos la red ######

    ### Cargamos los datos de MNIST
    v_size = 10000
    t_size = 10000

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x = np.vstack((x_train, x_test))
    y = np.append(y_train, y_test)

    # Preprocesado
    x = x / 255.0
    x = x.reshape(len(x), 28, 28, 1)
    y = to_categorical(y)

    x, x_test, y, y_test = train_test_split(x, y, test_size=t_size, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=v_size, stratify=y_train)

    # Parametros
    lr = 1e-4
    lamb = 1e-1
    batch_size = 64
    epochs = 50
    n_1 = 32
    n_2 = 16

    model = keras.models.Sequential(name="TomiNet")
    model.add(keras.layers.Conv2D(filters=n_1, kernel_size=(3, 3), activation="relu", padding="same", name="conv_1", input_shape=x_train[0].shape,))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_1"))
    model.add(keras.layers.Conv2D(filters=n_2, kernel_size=(3, 3), activation="relu", padding="same", name="conv_2",))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
    model.add(keras.layers.BatchNormalization(name="batch_norm_2"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(y_train.shape[1], activation="linear", kernel_regularizer=keras.regularizers.l2(lamb), name="dense"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.CategoricalCrossentropy(name="loss", from_logits=True), metrics=["acc"],
    )

    print(model.summary())

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2,)

    # Datos y graficos
    print("Precisión y loss sobre test = {}".format(model.evaluate(x_test, y_test)))
    ## Guardo los datos para usarlos ahora
    model.save_weights("MNIST")
    model.save("TomiNet_MNIST.h5")
    ###### Visualización de la red entrenada sobre MNIST Letters #####
    ######## Filtros ########
    ## Guiandome por https://keras.io/examples/vision/visualizing_what_convnets_learn/

    # Cargamos el modelo
    model_emnist = keras.models.load_model("TomiNet_MNIST.h5")
    model_emnist.load_weights("MNIST")

    # The dimensions of our input image (MNIST 32x32x1)
    img_width = 28
    img_height = 28
    img_depth = 1

    ## Buscamos la activación de la capa elegida
    layer_name = "conv_2"
    layer = model_emnist.get_layer(name=layer_name)
    feature_extractor = keras.Model(inputs=model_emnist.inputs, outputs=layer.output)

    ## Funciones para calcular el loss y el gradiente
    def compute_loss(input_image, filter_index):
        activation = feature_extractor(input_image)
        # We avoid border artifacts by only involving non-border pixels in the loss.
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)  # esto calcula la media de todos los pixeles de la activación

    @tf.function  # no entiendo bien que hace este arroba
    def gradient_ascent_step(img, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)  # creo que esto hace que siga a la img para ver que le pasa y asi poder calcular el gradiente
            loss = compute_loss(img, filter_index)  # aca por ejemplo ve que hace compute_loss con la img
        # Compute gradients.
        grads = tape.gradient(loss, img)  # ahora puede calcular el gradiente
        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads  # como es +=, es ascendente
        return loss, img

    # esto crea una imagen neutra ponele
    def initialize_image():
        # We start from a gray image with some random noise
        # devuelve un tensor con propiedades de tensor de tf
        img = tf.random.uniform((1, img_width, img_height, img_depth))
        # Uso imagenes que estan entre 0 y 1 para entrenar,
        # las achico para que sea mas neutra
        return (img) * 0.25

    # para ver la imagen que maximiza la activación del filtro
    def visualize_filter(filter_index):
        # We run gradient ascent for 30 steps
        iterations = 100
        learning_rate = 0.1  # muy alto pero mejor no lo toco
        img = initialize_image()
        for iteration in range(iterations):
            loss, img = gradient_ascent_step(img, filter_index, learning_rate)

        # Decode the resulting input image
        img = deprocess_image(img[0].numpy())
        return loss, img

    # Reescala la imagen
    def deprocess_image(img):
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()  # proceso del tensor
        img /= img.std() + 1e-5  # por si es 0 en algun caso
        img *= 0.15

        # Center crop # Lo saco porque mis imagenes son diferentes no se si andará bien
        # img = img[25:-25, 25:-25, :]

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Me parece mas piola no hacer el clip si no restarle el min y dividir por (max-min) pero veremos como queda
        # Convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")
        return img

    # Compute image inputs that maximize per-filter activations
    # for the first n filters of our target layer
    n = 16
    all_imgs = []
    for filter_index in range(n):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index)
        all_imgs.append(img)

    n = int(np.sqrt(n))

    plt.figure(figsize=(4, 4))
    for i, img in enumerate(all_imgs):
        plt.subplot(4, 4, i + 1)
        plt.imshow(img[:, :, 0], cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.01)  # , hspace=0.1, wspace=-0.5)
    # plt.savefig("{}_visualization.pdf".format(layer_name))
    plt.show()
