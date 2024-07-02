from PIL import Image, ExifTags
import os
import matplotlib.pyplot as plt

def reducir_resolucion(carpeta, factor):
    # Lista de archivos en la carpeta
    archivos = os.listdir(carpeta)

    # Iterar sobre cada archivo en la carpeta
    for archivo in archivos:
        # Ruta completa del archivo
        ruta_completa = os.path.join(carpeta, archivo)

        # Verificar si es un archivo de imagen
        if os.path.isfile(ruta_completa):
            try:
                # Abrir la imagen
                imagen = Image.open(ruta_completa)

                # Intentar obtener los datos EXIF de la imagen
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation] == 'Orientation':
                            break
                    exif = dict(imagen._getexif().items())
                    if exif[orientation] == 3:
                        imagen = imagen.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        imagen = imagen.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        imagen = imagen.rotate(90, expand=True)
                except (AttributeError, KeyError, IndexError):
                    # No hay datos EXIF, no se hace nada
                    pass

                # Obtener las dimensiones originales de la imagen
                ancho, alto = imagen.size

                # Calcular las nuevas dimensiones reduciendo a los factores deseados
                nuevo_ancho = int(ancho / factor)
                nuevo_alto = int(alto / factor)

                # Redimensionar la imagen con otro método de interpolación (por ejemplo, BILINEAR)
                imagen_reducida = imagen.resize((nuevo_ancho, nuevo_alto), resample=Image.BILINEAR)

                # Guardar la imagen reducida con un nombre específico
                nombre_reducido = f"{archivo.split('.')[0]}_reducida.jpg"
                ruta_reducida = os.path.join(carpeta, nombre_reducido)
                imagen_reducida.save(ruta_reducida)
                print(f"Se Ha Reducido La Resolución De {archivo} Correctamente.")

            except Exception as e:
                print(f"No Se Pudo Reducir La Resolución De {archivo}: {str(e)}")

def mostrar_comparacion(carpeta, nombre_imagen):
    # Ruta completa de la imagen original y reducida
    ruta_original = os.path.join(carpeta, nombre_imagen)
    ruta_reducida = os.path.join(carpeta, f"{nombre_imagen.split('.')[0]}_reducida.jpg")

    # Verificar si los archivos de imagen reducida y original existen
    if os.path.exists(ruta_original) and os.path.exists(ruta_reducida):
        try:
            # Abrir las imágenes
            imagen_original = Image.open(ruta_original)
            imagen_reducida = Image.open(ruta_reducida)

            # Mostrar la comparación utilizando Matplotlib
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(imagen_original)
            axs[0].set_title('Imagen Original')
            axs[0].axis('off')
            axs[1].imshow(imagen_reducida)
            axs[1].set_title('Imagen Reducida')
            axs[1].axis('off')
            plt.show()
        except Exception as e:
            print(f"No se pudo mostrar la comparación: {str(e)}")
    else:
        print("No se encontraron los archivos de imagen original o reducida.")

# Carpeta que contiene las imágenes
carpeta_imagenes = r"C:/Users/lmichel/Desktop/Fotos_Prediccion"

# Llamar a la función para reducir la resolución de las imágenes en la carpeta
factores = [15]  # CAMBIAR EL NUMERO AL FACTOR DESEADO
for factor in factores:
    reducir_resolucion(carpeta_imagenes, factor)

# Solicitar al usuario el nombre de la imagen que desea visualizar
nombre_imagen = input("Ingrese el nombre de la imagen que desea visualizar (incluyendo la extensión): ")
# Mostrar la comparación entre la imagen original y la reducida
mostrar_comparacion(carpeta_imagenes, nombre_imagen)

