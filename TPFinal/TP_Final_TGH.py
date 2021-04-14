########## Preprocesado de los datos ##########

# me guio por https://medium.com/swlh/image-captioning-in-python-with-keras-870f976e0f18
#             https://www.kaggle.com/shadabhussain/automated-image-captioning-flickr8
# https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
# dataset usado: https://www.kaggle.com/adityajn105/flickr8k

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
import json

############ Preprocesado de datos ###############

### preprocesado de imagenes

### feature extraction ####


def extract_features_VGG16(directory, save):
    # cargamos el modelo
    model = keras.applications.vgg16.VGG16()
    # nos quedamos con la
    model = keras.models.Model(inputs=model.inputs,
                               outputs=model.layers[-2].output)
    # resumen del modelo
    print(model.summary())
    # vamos a sacar las features y guardarlas en un diccionario
    features = {}
    for i, name in enumerate(os.listdir(directory)):
        # path del archivo
        filename = directory + '/' + name
        # cargamos la imagen con el tamaño adecuado para el modelo
        image = keras.preprocessing.image.load_img(filename,
                                                   target_size=(224, 224))
        # convertimos la imagen a un array
        image = keras.preprocessing.image.img_to_array(image)
        # reshape para el modelo, es una imagen a color de 224x224x3, lo convertimos a 1x224x224x3
        image = image[np.newaxis, :]
        # preprocesado del modelo
        image = keras.applications.vgg16.preprocess_input(image)
        # obtenemos las fetures de la imagen
        feature = model.predict(image, verbose=0)
        # lo guardamos en el diccionario
        features[name] = feature
        print(i)
    pickle.dump(features, open(save + '.pkl', 'wb'))
    return features


def extract_features_NASNet(directory, save):
    # cargamos el modelo
    model = keras.applications.nasnet.NASNetLarge()
    # nos quedamos con la
    model = keras.models.Model(inputs=model.inputs,
                               outputs=model.layers[-2].output)
    # resumen del modelo
    print(model.summary())
    # vamos a sacar las features y guardarlas en un diccionario
    features = {}
    for i, name in enumerate(os.listdir(directory)):
        # path del archivo
        filename = directory + '/' + name
        # cargamos la imagen con el tamaño adecuado para el modelo
        image = keras.preprocessing.image.load_img(filename,
                                                   target_size=(331, 331))
        # convertimos la imagen a un array
        image = keras.preprocessing.image.img_to_array(image)
        # reshape para el modelo, es una imagen a color de 224x224x3, lo convertimos a 1x224x224x3
        image = image[np.newaxis, :]
        # preprocesado del modelo
        image = keras.applications.nasnet.preprocess_input(image)
        # obtenemos las fetures de la imagen
        feature = model.predict(image, verbose=0)
        # lo guardamos en el diccionario
        features[name] = feature
        print(i)
    pickle.dump(features, open(save + '.pkl', 'wb'))
    return features


### vamos a cargar todos los archivos primero, que los tengo en drive ##


def load_descriptions(descriptions_path):
    # las cargo separadas por renglon
    descriptions = open(descriptions_path, 'r').read().split('\n')
    # vamos a guardar las descriptions en un diccionario
    mapping = {}
    for caption in descriptions[1:-1]:
        # divido la linea en nombre de imagen, descripción
        key, desc = caption.split(',', 1)
        if key in mapping:
            mapping[key].append(desc)
        else:
            mapping[key] = [desc]
    return mapping


def clean_descriptions(descriptions):
    # vamos a "limpiar" las descriptions, haciendo que sean minuscula y esas cosas
    import string
    for desc_list in descriptions.values():
        # creamos una lista de traducción que asigne None a todos los signos de puntiacion
        table = str.maketrans('', '', string.punctuation)
        # desc_list es una lista de descriptions de la imagen (5 por imagen)
        for idx, desc in enumerate(desc_list):
            # hacemos una lista con cada unos de los caracteres que conforman a una descripción
            desc = desc.split()
            # sacamos la puntuación
            desc = [w.translate(table) for w in desc]
            # sacamos los numeros
            desc = [word for word in desc if word.isalpha()]
            # sacamos las palabras de una letra
            desc = [word for word in desc if len(word) > 1]
            # minusculas
            desc = [word.lower() for word in desc]
            # guardamos
            desc_list[idx] = ' '.join(desc)


def to_vocabulary(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


def save_descriptions(descriptions, filename, test=1000, val=1000):
    random_keys = np.array(list(descriptions.keys()))
    np.random.shuffle(random_keys)
    val_keys = random_keys[:val]
    test_keys = random_keys[val:test + val]
    train_keys = random_keys[val + test:]
    keys_list = (val_keys, test_keys, train_keys)
    names = ['_val', '_test', '_train']
    for i, keys in enumerate(keys_list):
        lines = []
        for key in keys:
            for desc in descriptions[key]:
                lines.append(key + ' ' + desc)
        data = "\n".join(lines)
        file = open(filename + names[i] + '.txt', 'w')
        file.write(data)
        file.close()


def create_set(descriptions, filename='set.txt', test=1000, val=1000):
    # la idea es tener una lista con los nombres de las imagenes de cada set
    random_keys = np.array(list(descriptions.keys()))
    np.random.shuffle(random_keys)
    val_keys = random_keys[:val]
    test_keys = random_keys[val:test + val]
    train_keys = random_keys[val + test:]
    np.savetxt("val_" + filename, val_keys, delimiter='\t', fmt='%s')
    np.savetxt("test_" + filename, test_keys, delimiter='\t', fmt='%s')
    np.savetxt("train_" + filename, train_keys, delimiter='\t', fmt='%s')


# ## feature extraction / sacar el comentario para extraer y guardar las features
# extract_features_NASNet('Images','features_NASNETLarge')
# extract_features_VGG16('Images','features_VGG16')

# vamos primero con las descriptions
descriptions_path = 'captions.txt'
descriptions = load_descriptions(descriptions_path)
img_name = list(descriptions.keys())

# vamos a "limpiar" las descriptions
clean_descriptions(descriptions)

# creamos el vocabulario
vocabulary = to_vocabulary(descriptions)
print('Vocabulary size: {}'.format(len(vocabulary)))

# guardamos el las captions "limpias"
desc_file = open('clean_descriptions.pkl', "wb")
pickle.dump(descriptions, desc_file)
desc_file.close()

# creamos train-test-val set
create_set(descriptions, val=0)

##### FUNCIONES DE CARGA #####
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
import json
from pickle import load
from nltk.translate.bleu_score import corpus_bleu
import seaborn as sns

sns.set(style='whitegrid')


def load_clean_descriptions(filename, dataset):
    # todas la descripciones
    all_clean_desc = pickle.load(open(filename, 'rb'))
    # me quedo con las que esten en el dataset y les agrego el
    # prefijo startseq y el sufijo endseq
    dataset_clean_desc = {
        k: ['startseq ' + desc + ' endseq' for desc in all_clean_desc[k]]
        for k in dataset
    }
    return dataset_clean_desc


def load_features(filename, dataset):
    # todas las features
    all_features = pickle.load(open(filename, 'rb'))
    # me quedo con las features de las imagenes del dataset
    features = {k: all_features[k] for k in dataset}

    return features


def create_tokenizer(descriptions, num_words=None):
    # creamos un array de todas las descripciones
    # el tokenizer le asigna un numero a cada una de las palabras
    lines = np.array(list(descriptions.values())).ravel()
    print(num_words)
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the length of the description with the most words
def get_max_length(descriptions):
    lines = np.array(list(descriptions.values())).ravel()
    return max(len(d.split()) for d in lines)


# vamos a crear un generador de datos porque no alcanza la memoria
def data_generator(descriptions,
                   features,
                   tokenizer,
                   vocab_size,
                   max_length,
                   batch_size=1):
    # en cada yield entregamos algo y guardamos el estado de la función
    while 1:
        k = 0
        keys = np.array(list(descriptions.keys()))
        desc = np.array(list(descriptions.values()))
        idx = np.arange(len(keys))
        np.random.shuffle(idx)
        keys = keys[idx]
        desc = desc[idx]
        for i, key in enumerate(keys):
            k += 1
            # retrieve the photo feature
            feature = features[key][0]
            in_img, in_seq, out_word = create_sequences(
                tokenizer, desc[i], feature, vocab_size, max_length)
            if k == 1:
                in_imgs, in_seqs, out_words = in_img, in_seq, out_word
            else:
                in_imgs = np.vstack((in_imgs, in_img))
                in_seqs = np.vstack((in_seqs, in_seq))
                out_words = np.vstack((out_words, out_word))
            if k == batch_size or i == len(descriptions) - 1:
                k = 0
                yield ([in_imgs, in_seqs], out_words)


def create_sequences(tokenizer, desc_list, feature, vocab_size, max_length):
    X1, X2, y = [], [], []
    # convertimos las descripciones a secuencias de numeros
    seq_list = tokenizer.texts_to_sequences(desc_list)
    for seq in seq_list:
        # vamos a crear la secuencia
        for i in range(1, len(seq)):
            # la entrada y el objetivo
            in_seq, out_seq = seq[:i], seq[i]
            # lo pasamos a categorical, que es como trabaja la red
            out_seq = keras.utils.to_categorical([out_seq],
                                                 num_classes=vocab_size)[0]
            # lo guardamos
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    # la rellenamos con ceros para que tengan todas la misma longitud
    X2 = keras.preprocessing.sequence.pad_sequences(X2,
                                                    maxlen=max_length,
                                                    padding='post')
    return np.array(X1), np.array(X2), np.array(y)


def define_model_merge(vocab_size,
                       feature_shape,
                       max_length,
                       out_shape_merge,
                       save,
                       RNNLayer=keras.layers.LSTM):
    # vamos a crear un modelo tipo merge, las imagenes codificadas se suman al
    # texto codificado

    # el codificador de las imagenes
    inputs1 = keras.layers.Input(shape=feature_shape)
    fe1 = keras.layers.Dropout(0.5)(inputs1)
    fe2 = keras.layers.Dense(out_shape_merge, activation='relu')(fe1)

    # el codificador de las secuencias
    inputs2 = keras.layers.Input(shape=(max_length, ))
    se1 = keras.layers.Embedding(input_dim=vocab_size,
                                 output_dim=out_shape_merge,
                                 mask_zero=True)(inputs2)
    se2 = keras.layers.Dropout(0.5)(se1)
    se3 = RNNLayer(out_shape_merge)(se2)

    # decodificador
    decoder1 = keras.layers.Add()([fe2, se3])
    decoder2 = keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

    # modelo
    model = keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='acc')

    # resumen
    print(model.summary())
    keras.utils.plot_model(model, to_file=save, show_shapes=True)

    return model


def define_model_inject(
    vocab_size,
    feature_shape,
    max_length,
    out_shape,
    save,
    RNNLayer=keras.layers.LSTM,
):
    # vamos a crear un modelo tipo merge, las imagenes codificadas se suman al
    # texto codificado

    # el codificador de las imagenes
    inputs1 = keras.layers.Input(shape=feature_shape)
    fe1 = keras.layers.Dropout(0.5)(inputs1)
    fe2 = keras.layers.Dense(out_shape, activation='relu')(fe1)
    fe3 = keras.layers.Reshape((out_shape, 1))(fe2)

    # el codificador de las secuencias
    inputs2 = keras.layers.Input(shape=(max_length, ))
    se1 = keras.layers.Embedding(input_dim=vocab_size,
                                 output_dim=out_shape,
                                 mask_zero=True)(inputs2)
    se1 = keras.layers.Permute((2, 1))(se1)

    print(fe3.shape)
    print(se1.shape)

    # decodificador
    decoder1 = keras.layers.Concatenate()([fe3, se1])
    decoder1 = keras.layers.Permute((2, 1))(decoder1)
    decoder2 = keras.layers.LSTM(out_shape)(decoder1)
    decoder3 = keras.layers.Dropout(0.5)(decoder2)
    decoder4 = keras.layers.Dense(256, activation='relu')(decoder3)
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder4)

    # modelo
    model = keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='acc')

    # resumen
    print(model.summary())
    keras.utils.plot_model(model, to_file=save, show_shapes=True)

    return model


def generate_desc(model, tokenizer, feature, max_length, vocab_size):
    # crea la descripción de una imagen
    # como semilla siempre lo mismo
    in_seq = tokenizer.texts_to_sequences(['startseq'])
    for i in range(max_length):
        # la paddeamos para que cumpla con el formato de la red
        seq = keras.preprocessing.sequence.pad_sequences(in_seq,
                                                         maxlen=max_length,
                                                         padding='post')
        # predecimos la palabra siguiente
        out_seq = np.argmax(model.predict([feature, seq], verbose=0))
        if out_seq > vocab_size:
            break
        in_seq[0].append(out_seq)
        if out_seq == tokenizer.texts_to_sequences(['endseq'])[0][0]:
            break
    text = tokenizer.sequences_to_texts(in_seq)
    return text


def evaluate_model(model, descriptions, features, tokenizer, max_length,
                   vocab_size):
    actual, predicted = [], []
    for key, desc_list in descriptions.items():
        # generamos la descripcion
        predicted.append(
            generate_desc(model, tokenizer, features[key], max_length,
                          vocab_size)[0].split())
        # generamos las correctas
        actual.append([d.split() for d in desc_list])
    print('BLEU-1: %f' %
          corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' %
          corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' %
          corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' %
          corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def picture_with_caption(model, tokenizer, max_length, vocab_size, features,
                         key, save):
    # buscamos las features de la imagen
    if isinstance(features, dict):
        features = features[key]
    caption = generate_desc(model, tokenizer, features, max_length, vocab_size)
    caption = ' '.join([
        word for word in caption[0].split(' ')
        if word != 'startseq' and word != 'endseq'
    ])
    ### mostramos una imagen con su caption como ejemplo
    fs = 16
    img = mpimg.imread('Images/' + key)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.figtext(0.5,
                0.01,
                caption,
                wrap=True,
                horizontalalignment='center',
                fontsize=fs)
    plt.savefig(save)
    plt.show()


def plot_hist(hist, title, fs, lw, save):
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,
                       len(hist['loss']) + 1),
             hist['loss'] / np.max(hist['loss']),
             lw=lw,
             label='Loss Normalizada')
    plt.plot(np.arange(1,
                       len(hist['acc']) + 1),
             hist['acc'],
             lw=lw,
             label='Precisión')
    plt.title(title, fontsize=fs)
    plt.xlabel('Época', fontsize=fs)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(np.arange(1, len(hist['loss']) + 1, 5))
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()


##### MODEL - TRAIN ######

## cargamos los datos
## definimos tel modelo que vamos a usar
feature_extractor = 'NASNETLarge'  # 'VGG16' o 'NASNETLarge
RNN_Layer_name = 'LSTM'  # 'LSTM' o 'SimpleRNN
num_words = 500  # algun numero
architecture = 'merge'  # 'merge' o 'inject'

features_path = 'features' + '_' + feature_extractor + '.pkl'
if RNN_Layer_name == 'LSTM':
    RNN_Layer = keras.layers.LSTM
elif RNN_Layer_name == 'SimpleRNN':
    RNN_Layer = keras.layers.SimpleRNN

train_set = np.loadtxt('train_set.txt', dtype='str')
train_descriptions = load_clean_descriptions('clean_descriptions.pkl',
                                             train_set)
train_features = load_features(features_path, train_set)

val_set = np.loadtxt('val_set.txt', dtype='str')
val_descriptions = load_clean_descriptions('clean_descriptions.pkl', val_set)
val_features = load_features(features_path, val_set)

test_set = np.loadtxt('test_set.txt', dtype='str')
test_descriptions = load_clean_descriptions('clean_descriptions.pkl', test_set)
test_features = load_features(features_path, test_set)

# preprocesamos los datos y eso #
print('Tamaño de train set: {}'.format(len(train_set)))
tokenizer = create_tokenizer(train_descriptions, num_words=num_words)

if num_words is not None:
    vocab_size = num_words
else:
    vocab_size = len(tokenizer.word_index) + 1
print('Numero de palabras: {}'.format(len(tokenizer.word_index)))
max_length = get_max_length(train_descriptions)
print('Caption mas larga: {}'.format(max_length))

# guardamos el tokenizer
#pickle.dump(tokenizer, open('tokenizer{}.pkl'.format(vocab_size),'wb'))

# creamos el modelo
features_shape = list(train_features.values())[0][0].shape
epochs = 40
batch_size = 50
steps = int(len(train_descriptions) / batch_size + 1)
out_shape = 256

if architecture == 'merge':
    model = define_model_merge(
        vocab_size, features_shape, max_length, out_shape,
        'model_{}-{}-{}-vocab={}-epochs={}.pdf'.format(feature_extractor,
                                                       RNN_Layer_name,
                                                       architecture,
                                                       vocab_size, epochs),
        RNN_Layer)
elif architecture == 'inject':
    model = define_model_inject(
        vocab_size, features_shape, max_length, out_shape,
        'model_{}-{}-{}-vocab={}-epochs={}.pdf'.format(feature_extractor,
                                                       RNN_Layer_name,
                                                       architecture,
                                                       vocab_size, epochs),
        RNN_Layer)

# # creemos las secuencias de train, val y test
train_generator = data_generator(train_descriptions, train_features, tokenizer,
                                 vocab_size, max_length, batch_size)
val_generator = data_generator(test_descriptions, test_features, tokenizer,
                               vocab_size, max_length, batch_size)

# # entrenamos y guardamos modelo
his = model.fit(x=train_generator,
                epochs=epochs,
                verbose=2,
                steps_per_epoch=steps)
# model.save('model_{}-{}-{}-vocab={}-epochs={}.h5'.format(feature_extractor,RNN_Layer_name,architecture,vocab_size,epochs))

# # guardamos el history
hist_file = open(
    'hist_{}-{}-{}-vocab={}-epochs={}.pkl'.format(feature_extractor,
                                                  RNN_Layer_name, architecture,
                                                  vocab_size, epochs), "wb")
pickle.dump(his.history, hist_file)
# hist_file.close()

evaluate_model(model, test_descriptions, test_features, tokenizer, max_length,
               vocab_size)

keys = list(test_features.keys())
np.random.shuffle(keys)
key = keys[0]

picture_with_caption(
    model,
    tokenizer,
    max_length,
    vocab_size,
    test_features[key],
    key,
    save='caption-img:{}_{}-{}-{}-vocab={}-epochs={}.pdf'.format(
        key[:-4], feature_extractor, RNN_Layer_name, architecture, vocab_size,
        epochs))

plot_hist(
    his.history, 'Inject', 16, 3,
    'plot_{}-{}-{}-vocab={}-epochs={}.pdf'.format(feature_extractor,
                                                  RNN_Layer_name, architecture,
                                                  vocab_size, epochs))
