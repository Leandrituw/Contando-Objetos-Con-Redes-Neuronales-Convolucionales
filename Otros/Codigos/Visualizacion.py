import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# Cargar el modelo preentrenado
model = tf.keras.models.load_model('D:/Users/Leandro/Downloads/Redes Neuronales/DataSets/Maices_Solos/Modelo_97.50%_1000f_RGB_Maices_Solos.h5')

# Imprimir los nombres de las capas para identificarlas
for layer in model.layers:
    print(layer.name)

# Numero de capa que deseas ver, 0 para 1ra capa conv - 3 para 2da capa conv
layer_index = 0

# Obtener la capa
layer = model.layers[layer_index]

# Obtener los pesos de la capa seleccionada (filtros y sesgos)
weights = layer.get_weights()

# Los filtros son el primer elemento de la lista de pesos
filters = weights[0]

# Imprimir la forma de los filtros
print("Forma de los filtros:", filters.shape)

# Número de filtros
num_filters = filters.shape[-1]

# Crear un nuevo modelo que incluya la capa seleccionada
new_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

# Cargar la imagen y preprocesarla
img_path = 'D:/Users/Leandro/Downloads/Redes Neuronales/DataSets/Maices_Solos/DataSet_1a20_1000_Maices_Solos_225x400_RGB/2/2_1.jpg'
img = load_img(img_path, target_size=(400, 225))  # Mantén el tamaño original

# Convertir la imagen a un arreglo de NumPy
img_array = img_to_array(img)

# Añadir dimensión de batch y normalizar
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Obtener las activaciones de la capa seleccionada (mapas de características)
activations = new_model.predict(img_array)

# Número de mapas de características (profundidad)
n = activations.shape[-1]

# Visualizar los mapas de características
for i in range(n):
    plt.figure(figsize=(5, 5))
    plt.imshow(activations[0, :, :, i], cmap='viridis', interpolation='nearest')
    plt.title(f"Mapa de características {i+1}")
    plt.axis('off')
    plt.show()

# Visualizar y graficar los valores numéricos de los filtros
for i in range(num_filters):
    # Seleccionar el filtro i-ésimo
    filter_img = filters[:, :, :, i]
    
    # Redondear los valores a 2 decimales
    filter_img_rounded = np.round(filter_img, 2)
    
    print(f"\nValores del Filtro {i+1} (redondeado a 2 decimales):")
    print(filter_img_rounded)  # Imprimir los valores numéricos redondeados del filtro
    
    # Verificar cuántos canales tiene el filtro
    num_channels = filter_img.shape[-1]
    
    for channel in range(num_channels):
        # Asignar un colormap dependiendo del canal
        cmap = 'Reds' if channel == 0 else 'Greens' if channel == 1 else 'Blues' if channel == 2 else 'gray'
        color = 'red' if channel == 0 else 'green' if channel == 1 else 'blue'
        
        # Graficar el canal con el colormap correspondiente
        plt.figure(figsize=(5, 5))
        plt.imshow(filter_img_rounded[:, :, channel], cmap=cmap, interpolation='nearest')
        plt.title(f"Filtro {i+1} - Canal {channel+1}")
        
        # Añadir los valores numéricos a la gráfica con el color correspondiente
        for y in range(filter_img_rounded.shape[0]):
            for x in range(filter_img_rounded.shape[1]):
                value = filter_img_rounded[y, x, channel]
                plt.text(x, y, str(value), color=color, fontsize=8, ha='center', va='center')
        
        plt.axis('off')
        plt.show()
