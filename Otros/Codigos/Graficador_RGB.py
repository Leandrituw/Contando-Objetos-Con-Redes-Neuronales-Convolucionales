import cv2
import matplotlib.pyplot as plt

def graficar_curvas_rgb(ruta_imagen):
    # Cargar la imagen desde la ruta especificada
    imagen = cv2.imread(ruta_imagen)

    # Convertir la imagen de BGR a RGB
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Separar los canales R, G y B de la imagen
    canal_rojo, canal_verde, canal_azul = cv2.split(imagen_rgb)

    # Crear la figura y los ejes para las curvas
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar las curvas RGB
    ax.plot(canal_rojo, color='red', label='Rojo')
    ax.plot(canal_verde, color='green', label='Verde')
    ax.plot(canal_azul, color='blue', label='Azul')

    # Configurar el título y las etiquetas de los ejes
    ax.set_title('Curvas RGB de la imagen')
    ax.set_xlabel('Posición del píxel')
    ax.set_ylabel('Valor del píxel')

    # Mostrar la leyenda
    ax.legend()

    # Mostrar la gráfica
    plt.show()

# Ruta de la imagen que deseas analizar (cambia esto por la ruta de tu propia imagen)
ruta_imagen = "URL"

# Llamar a la función para graficar las curvas RGB
graficar_curvas_rgb(ruta_imagen)
