import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import csv

# Cargar el modelo preentrenado
model = tf.keras.models.load_model('C:/Users/Leandro/Desktop/Modelo_97.50%_1000f_RGB_Maices_Solos.h5')

# Imprimir los nombres de las capas para identificarlas
for layer in model.layers:
    print(layer.name)

# Localizar la capa flatten
flatten_layer_index = None
for idx, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.Flatten):
        flatten_layer_index = idx
        break

# Si no hay capa flatten, indicar que no se encontró
if flatten_layer_index is None:
    print("No se encontró ninguna capa flatten en el modelo.")
else:
    # Crear un modelo que termine en la capa flatten
    flatten_layer = model.layers[flatten_layer_index]
    new_model = tf.keras.Model(inputs=model.inputs, outputs=flatten_layer.output)

    # Cargar la imagen y preprocesarla
    img_path = 'C:/Users/Leandro/Desktop/5_1.jpg'
    img = load_img(img_path, target_size=(400, 225))  # Mantén el tamaño original

    # Convertir la imagen a un arreglo de NumPy
    img_array = img_to_array(img)

    # Añadir dimensión de batch y normalizar
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Obtener las activaciones de la capa flatten (es decir, el vector aplanado)
    flattened_output = new_model.predict(img_array)

    # Guardar los valores del vector aplanado en un archivo CSV
    output_file = 'flatten_output.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Valor"])  # Encabezado opcional
        for value in flattened_output[0]:
            writer.writerow([value])

    print(f"Los valores del flatten se guardaron en {output_file}")
