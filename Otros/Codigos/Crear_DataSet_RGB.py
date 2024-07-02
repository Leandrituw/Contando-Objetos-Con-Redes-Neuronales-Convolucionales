import os
import numpy as np
import h5py
from PIL import Image

# Ruta al directorio del dataset
dataset_dir = 'C:/Users/Leandro/Desktop/DataSet_1a20_800_225x400_RGB'

# Tamaño de las imágenes
img_width, img_height = 225, 400

# Obtener las clases del dataset (subcarpetas)
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit()]
num_classes = len(classes)

# Inicializar listas para almacenar las imágenes y etiquetas
images = []
labels = []

# Procesar cada clase
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    # La etiqueta es el nombre de la subcarpeta convertido a entero
    class_idx = int(class_name) - 1  # Restar 1 para que las clases sean de 0 a 19
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).resize((img_width, img_height))
        img_array = np.array(img)
        
        # Asegurarse de que la imagen tiene tres canales (RGB)
        if img_array.shape == (img_height, img_width, 3):
            images.append(img_array)
            labels.append(class_idx)

# Convertir listas a arrays de NumPy
images = np.array(images)
labels = np.array(labels)

# Guardar las imágenes y etiquetas en un archivo HDF5
with h5py.File('DataSet.h5', 'w') as h5f:
    h5f.create_dataset('images', data=images)
    h5f.create_dataset('labels', data=labels)

print("Terminó el proceso")

