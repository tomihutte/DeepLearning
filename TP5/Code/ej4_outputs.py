########### Ejercicio 4 -  OUTPUTS ###############################
###### Visualización de la red entrenada sobre MNIST #####

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
model_mnist = keras.models.load_model("TomiNet_MNIST.h5")
model_mnist.load_weights("MNIST")

print(model_mnist.summary())

## Buscamos la activación de la capa elegida
layer_name = "dense"
layer = model_mnist.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model_mnist.inputs, outputs=layer.output)

# The dimensions of our input image (MNIST 32x32x1)
img_width = 28
img_height = 28
img_depth = 1

## Funciones para calcular el loss y el gradiente
def compute_loss(input_image, output_index):
    return -model_mnist.loss(feature_extractor(input_image), to_categorical(np.array([output_index]), num_classes=10))


# no entiendo bien que hace este arroba
def gradient_ascent_step(img, output_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)  # creo que esto hace que siga a la img para ver que le pasa y asi poder calcular el gradiente
        loss = compute_loss(img, output_index)  # aca por ejemplo ve que hace compute_loss con la img
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
def visualize_output(output_index):
    # We run gradient ascent for 30 steps
    iterations = 30
    learning_rate = 1.0  # muy alto pero mejor no lo toco
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, output_index, learning_rate)

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


# Compute image inputs that maximize per-output activations
# for the first n outputs of our target layer
n = 10
all_imgs = []
for output_index in range(n):
    print("Processing output %d" % (output_index,))
    loss, img = visualize_output(output_index)
    all_imgs.append(img)


plt.figure(figsize=(4, 8))
for i, img in enumerate(all_imgs):
    plt.imshow(img[:, :, 0], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()
