###### Visualización de la red entrenada sobre MNIST Letters #####

## Guiandome por https://keras.io/examples/vision/visualizing_what_convnets_learn/

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split


# Cargamos el modelo
model_emnist = keras.models.load_model("TomiNet.h5")
model_emnist.load_weights("EMNIST")


# The dimensions of our input image (MNIST 32x32x1)
img_width = 28
img_height = 28
img_depth = 1

## Buscamos la activación de la capa elegida
layer_name = "conv_1"
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
    iterations = 30
    learning_rate = 10.0  # muy alto pero mejor no lo toco
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

plt.figure(figsize=(4, 8))
for i, img in enumerate(all_imgs):
    plt.subplot(n * 2, n / 2, i + 1)
    plt.imshow(img[:, :, 0])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.1, wspace=-0.5)
plt.savefig("{}_visualization.pdf".format(layer_name))
plt.show()
