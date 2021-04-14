import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import seaborn as sns
import time

##### 1  #####
data = pd.read_csv('airline-passengers.csv')
data = data['Passengers'].values.astype('float32')

#### 2 ####

# Normalizamos
x_max = np.max(data)
x_min = np.min(data)
y = (data - x_min) / (x_max - x_min)
x = (data - x_min) / (x_max - x_min) + np.random.normal(scale=0.02,
                                                        size=y.shape)

# Reformateo
lag = 1


def data_create(data_noise, data, lag=1):
    dataX, dataY = [], []
    for i in range(len(data) - lag):
        a = data_noise[i:(i + lag)]
        dataX.append(a)
        dataY.append(data[i + lag])
    return np.array(dataX), np.array(dataY)


x, y = data_create(x, y, lag=lag)

#### 3 ####

##x += np.random.normal(scale=0.02, size=x.shape)

#### 5-4 ####

## x.shape = (samples,time_steps,feautures=1)
x = x[:, :, np.newaxis]
# x = x[:, np.newaxis, :]

test_size = 0.5
n_test = int(test_size * len(x))

x_test = x[-n_test:]
x_train = x[:-n_test]
y_test = y[-n_test:]
y_train = y[:-n_test]

#### 6 ####
### Creamos el modelo ####
lr = 1e-3
n_out = 4

model = keras.models.Sequential()
model.add(
    keras.layers.LSTM(n_out,
                      return_sequences=False,
                      name='LSTM_1',
                      input_shape=x_train[0].shape))
model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')

print(model.summary())

### 7 ###
epochs = 50
batch_size = 1

history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)

# Datos y graficos

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25
fig_size = (6, 5)

plt.figure(figsize=fig_size)
plt.plot(
    history.history["loss"],
    lw=lw,
    label="Entrenamiento",
)
plt.plot(history.history["val_loss"], lw=lw, label="Test")
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.title(r'Capa LSTM - $l={}$'.format(lag), fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig('Loss.pdf')
plt.show()

### 8 ####
y_train_pred = model.predict(x_train) * (x_max - x_min) + x_min
month_train = np.arange(lag, len(y_train) + lag)
y_test_pred = model.predict(x_test) * (x_max - x_min) + x_min
month_test = np.arange(len(y_train) + lag, len(data))
y = y * (x_max - x_min) + x_min
months_y = np.arange(lag, len(data))
# Datos y graficos

sns.set_style("whitegrid")
fs = 16
lw = 2
s = 25
fig_size = (7, 5.5)

plt.figure(figsize=fig_size)
plt.plot(months_y, y, lw=lw, label="Datos", c='C0')
plt.plot(month_train,
         y_train_pred,
         lw=lw,
         label="Predicción train",
         ls='--',
         c='C1')
plt.plot(month_test,
         y_test_pred,
         lw=lw,
         label="Predicción test",
         ls='--',
         c='red')
plt.xlabel("Mes", fontsize=fs)
plt.ylabel("Numero de pasajeros", fontsize=fs)
plt.title(r'Capa LSTM - $l={}$'.format(lag), fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig('Pred.pdf')
plt.show()

print('\n \n \n')

# #### 9 ####

## Vamos a barrer el parametro lag (l) y ver como cambia con eso ##
lag_max = 25
prom = 20
lags = np.arange(1, lag_max)
losses_test = np.zeros(shape=(lag_max - 1, prom))
losses_train = np.zeros(shape=(lag_max - 1, prom))

print('Evaluando en diferentes lags')

for lag in lags:
    start = time.time()
    for n in range(prom):
        # datos normalizados
        y = (data - x_min) / (x_max - x_min)
        x = (data - x_min) / (x_max - x_min) + np.random.normal(scale=0.02,
                                                                size=y.shape)
        x, y = data_create(x, y, lag=lag)
        # ruido
        x += np.random.normal(scale=0.02, size=x.shape)
        # reshape
        x = x[:, :, np.newaxis]
        # x = x[:, np.newaxis, :]

        # test/train
        test_size = 0.5
        n_test = int(test_size * len(x))

        x_test = x[-n_test:]
        x_train = x[:-n_test]
        y_test = y[-n_test:]
        y_train = y[:-n_test]

        # modelo
        n_out = 4
        lr = 1e-3

        model = keras.models.Sequential()
        model.add(
            keras.layers.LSTM(n_out,
                              return_sequences=False,
                              name='LSTM_1',
                              input_shape=x_train[0].shape))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')

        # entrenamiento
        epochs = 100
        batch_size = 1

        history = model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0)

        losses_test[lag - 1, n] = model.evaluate(x_test, y_test, verbose=0)
        losses_train[lag - 1, n] = history.history['loss'][-1]

    print('{}/{} - {} s - loss_test = {} - loss_train = {}'.format(
        lag, lag_max - 1,
        time.time() - start, losses_test[lag - 1], losses_train[lag - 1]))

# Datos y graficos

sns.set_style("whitegrid")
fs = 16
lw = 2
s = 25
fig_size = (6.5, 5.5)

plt.figure(figsize=fig_size)
# plt.plot(lags, losses_test.mean(axis=1), lw=lw, marker='o', label='Test')
plt.plot(lags, losses_train.mean(axis=1), lw=lw, marker='o', label='Train')
plt.xlabel(r"Parámetro $l$", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.title('Loss al finalizar entrenamiento', fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig('enfunciondeL_train.pdf')
plt.show()

print('\n \n \n')

# ### 10 ###
## Repito varias partes de lo anterior y solo cambio el modele ###
print('Modelo con capas densas')

# Normalizamos
x_max = np.max(data)
x_min = np.min(data)
y = (data - x_min) / (x_max - x_min)
x = (data - x_min) / (x_max - x_min) + np.random.normal(scale=0.02,
                                                        size=y.shape)

# Reformateo
lag = 1


def data_create(data_noise, data, lag=1):
    dataX, dataY = [], []
    for i in range(len(data) - lag):
        a = data_noise[i:(i + lag)]
        dataX.append(a)
        dataY.append(data[i + lag])
    return np.array(dataX), np.array(dataY)


x, y = data_create(x, y, lag=lag)

## x.shape = (samples,time_steps,feautures=1)
x = x[:, :, np.newaxis]
# x = x[:, np.newaxis, :]

test_size = 0.5
n_test = int(test_size * len(x))

x_test = x[-n_test:]
x_train = x[:-n_test]
y_test = y[-n_test:]
y_train = y[:-n_test]

### Creamos el modelo ####
lr = 1e-3
n_out = 4

model = keras.models.Sequential()
model.add(
    keras.layers.Dense(n_out, name='dense_1', input_shape=x_train[0].shape))
model.add(keras.layers.Dense(1, name='output'))
model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')

print(model.summary())

epochs = 50
batch_size = 1

history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)

# Datos y graficos

sns.set_style("whitegrid")
fs = 16
lw = 3
s = 25
fig_size = (6, 5)

plt.figure(figsize=fig_size)
plt.plot(
    history.history["loss"],
    lw=lw,
    label="Entrenamiento",
)
plt.plot(history.history["val_loss"], lw=lw, label="Test")
plt.xlabel("Epoca", fontsize=fs)
plt.ylabel("Loss", fontsize=fs)
plt.title(r'Capa Densa - $l={}$'.format(lag), fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig('Loss_dense.pdf')
plt.show()

### 8 ####
y_train_pred = (model.predict(x_train) * (x_max - x_min) + x_min).ravel()
month_train = np.arange(lag, len(y_train) + lag)
y_test_pred = (model.predict(x_test) * (x_max - x_min) + x_min).ravel()
month_test = np.arange(len(y_train) + lag, len(data))
y = y * (x_max - x_min) + x_min
months_y = np.arange(lag, len(data))

# Datos y graficos

sns.set_style("whitegrid")
fs = 16
lw = 2
s = 25
fig_size = (7, 5.5)

plt.figure(figsize=fig_size)
plt.plot(months_y, y, lw=lw, label="Datos", c='C0')
plt.plot(month_train,
         y_train_pred,
         lw=lw,
         label="Predicción train",
         ls='--',
         c='C1')
plt.plot(month_test,
         y_test_pred,
         lw=lw,
         label="Predicción test",
         ls='--',
         c='red')
plt.xlabel("Mes", fontsize=fs)
plt.ylabel("Numero de pasajeros", fontsize=fs)
plt.title(r'Capa densa - $l={}$'.format(lag), fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs)
plt.tight_layout()
plt.savefig('Pred_dense.pdf')
plt.show()