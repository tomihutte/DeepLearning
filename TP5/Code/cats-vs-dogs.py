# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

## Parametros
img_size = (150, 150)

# define location of dataset
folder = "train/"
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
    # determine class
    output = 0.0
    if file.startswith("cat"):
        output = 1.0
    # load image
    photo = load_img(folder + file, target_size=img_size)
    # convert to numpy array
    photo = img_to_array(photo)
    # store
    photos.append(photo)
    labels.append(output)
# convert to a numpy arrays
photos = np.asarray(photos).astype(np.uint8)
labels = np.asarray(labels).astype(np.uint8)
print(photos.shape, labels.shape)
# save the reshaped photos
np.save("dogs_vs_cats_photos_{}.npy".format(img_size), photos)
np.save("dogs_vs_cats_labels_{}.npy".format(img_size), labels)

