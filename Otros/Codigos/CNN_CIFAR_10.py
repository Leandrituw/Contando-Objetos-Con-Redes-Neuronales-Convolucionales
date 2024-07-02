# Importamos las librerías que vamos a usar en este cuaderno.
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Para silenciar posibles warnings de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

num_train, img_channels, img_rows, img_cols =  train_images.shape
num_train, img_channels, img_rows, img_cols

num_test, _, _, _ =  test_images.shape
num_test, img_channels, img_rows, img_cols

class_names = ['airplane','automobile','bird','cat','deer', 'dog','frog','horse','ship','truck']

fig = plt.figure(figsize=(8,3))
for i in range(len(class_names)):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(train_labels == i)[0]
    features_idx = train_images[idx]
    img_num = np.random.randint(features_idx.shape[0])
    im = features_idx[img_num]
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

# Normalizamos el train y el test data-set entre 0 y 1
train_features = train_images.astype('float32') / 255.
test_features = test_images.astype('float32') / 255.

# Convertimos las etiquetas a variables One-Hot Encoded

num_classes = len(np.unique(train_labels))

train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Definición del modelo
# Iniciamos el modelo de manera secuencial
model = tf.keras.models.Sequential()
# Continuamos añadiendo al modelo las capas sin preocuparnos de la dimensionalidad de los inputs
# salvo en la primera capa
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                 input_shape=train_features.shape[1:]))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))
# Hacemos un flattening de la última capa de Pooling
model.add(Flatten())

model.add(Dense(units=512, activation = 'relu'))

model.add(Dropout(rate=0.25))

model.add(Dense(units=num_classes, activation='softmax'))

# Model summary y ploteo del grafo
model.summary()

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_dir = os.getcwd() + '/checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Checkpoint Best Convolutional Neural Network Model Only

filepath= checkpoint_dir + "/weights.best.hdf5"

# Intentaremos guardar un checkpoint cada vez que acabe una Epoch (period=1) en base a la predicción de validación 'val_acc'.
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
callbacks_list = [checkpoint]

start = time.time()

# Haciendo uso de la función fit() ajustaremos nuestros modelo a los datos provistos.
model_info = model.fit(train_features, train_labels,
                       batch_size=128, epochs=10,
                       validation_data = (test_features, test_labels),
                       callbacks=callbacks_list,
                       verbose=1)
end = time.time()

def plot_model_history(history):
    fig, ax = plt.subplots(1, 2,figsize=(20,7))

    # Resumen de la evolución de la precisión
    ax[0].plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'])
    ax[0].plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'])
    ax[0].set_ylabel('Precision')
    ax[0].set_xlabel('Epoch')
    ax[0].set_xticks(np.arange(1, len(history.history['acc']) + 1), len(history.history['acc']) / 10)
    ax[0].legend(['train', 'validation'], loc='best')

    # Resumen de la evolución de la función de pérdida
    ax[1].plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
    ax[1].plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'])
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_xticks(np.arange(1, len(history.history['loss']) + 1), len(history.history['loss']) / 10)
    ax[1].legend(['train', 'validation'], loc='best')

    plt.show()

def accuracy(test_image, test_label, model):

    # Haciendo uso de model.predict
    result = model.predict(test_image)

    # La predicción de la clase se obtiene haciendo uso de np.argmax()
    predicted_class = np.argmax(result, axis=1)

    # Contamos con la clase verdadera del test dataset,
    # para compararla con la clase predicha
    true_class = np.argmax(test_label, axis=1)

    # Calculamos la precisión sobre el test dataset de nuestro clasificador
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

# Bondad del modelo entrenado teniendo en cuenta el test data-set.
scores = model.evaluate(test_features, test_labels, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

plot_model_history(model_info)
print("El entrenamiento del modelo duró %0.2f segundos"%(end - start))

# compute test accuracy
print("La Accuracy sobre el test data-set es de: %0.2f %%" %accuracy(test_features, test_labels, model))
