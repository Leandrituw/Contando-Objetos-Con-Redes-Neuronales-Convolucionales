import numpy as np
from keras.utils import to_categorical

# Cargar las imágenes y las etiquetas desde los archivos .npy
X_train = np.load('C:/Users/Leandro/Desktop/Archivos_Binarios_RGB/images.npy', allow_pickle=True)
y_train = np.load('C:/Users/Leandro/Desktop/Archivos_Binarios_RGB/labels.npy', allow_pickle=True)

print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)

# Verificar los valores únicos en y_train
etiquetas_unicas = np.unique(y_train)
print("Etiquetas únicas en y_train:", etiquetas_unicas)

# Verificar el valor máximo en y_train
etiqueta_maxima = np.max(y_train)
print("Etiqueta máxima en y_train:", etiqueta_maxima)

# Configurar num_classes correctamente
num_classes = etiqueta_maxima + 1
print("Número de clases (num_classes):", num_classes)

# Convertir etiquetas a formato categórico
y_train = to_categorical(y_train, num_classes)
print("Forma de y_train después de to_categorical:", y_train.shape)
