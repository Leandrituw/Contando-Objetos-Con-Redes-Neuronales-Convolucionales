import h5py
import matplotlib.pyplot as plt
import numpy as np

# Función para visualizar una imagen por cada clase en un dataset HDF5 en formato de grid
def visualizar_imagenes_por_clase(archivo_h5, nombre_dataset_imagenes, nombre_dataset_etiquetas, class_names, filas=2, img_size=5, espacio=0.3):
    with h5py.File(archivo_h5, 'r') as archivo:
        if nombre_dataset_imagenes in archivo and nombre_dataset_etiquetas in archivo:
            dataset_imagenes = archivo[nombre_dataset_imagenes]
            dataset_etiquetas = archivo[nombre_dataset_etiquetas]
            print(f"Visualizando Una Imagen Por Cada Clase En El Dataset '{nombre_dataset_imagenes}'...")

            num_clases = len(class_names)
            columnas = (num_clases + filas - 1) // filas
            fig = plt.figure(figsize=(columnas * img_size, filas * img_size))

            # Iterar sobre las clases y encontrar un ejemplo de cada clase
            for i, class_name in enumerate(class_names):
                idx = np.where(dataset_etiquetas[:] == (i + 1))[0]
                ax = fig.add_subplot(filas, columnas, 1 + i, xticks=[], yticks=[])
                if len(idx) > 0:
                    imagen_idx = idx[0]  # Tomar el primer índice de la clase
                    im = dataset_imagenes[imagen_idx]
                    ax.imshow(im)
                    ax.set_title(class_name, fontsize=12)
                else:
                    ax.axis('off')
                    ax.set_title(class_name, fontsize=12)

            plt.subplots_adjust(wspace=espacio, hspace=espacio)
            plt.show()
        else:
            print(f"El dataset '{nombre_dataset_imagenes}' o '{nombre_dataset_etiquetas}' no se encontró en el archivo.")

# Ruta del archivo HDF5
archivo_h5 = "C:/Users/Leandro/Desktop/DataSet.h5"

# Nombre del dataset que contiene las imágenes y etiquetas
nombre_dataset_imagenes = "images"
nombre_dataset_etiquetas = "labels"

# Lista de nombres de clases (reemplaza esto con tu lista de nombres de clases)
class_names = ["Clase 1", "Clase 2", "Clase 3", "Clase 4", "Clase 5", 
               "Clase 6", "Clase 7", "Clase 8", "Clase 9", "Clase 10",
               "Clase 11", "Clase 12", "Clase 13", "Clase 14", "Clase 15",
               "Clase 16", "Clase 17", "Clase 18", "Clase 19", "Clase 20"]

# Visualizar una imagen por cada clase en el dataset especificado en formato de grid
visualizar_imagenes_por_clase(archivo_h5, nombre_dataset_imagenes, nombre_dataset_etiquetas, class_names, filas=4, img_size=5, espacio=0.5)

