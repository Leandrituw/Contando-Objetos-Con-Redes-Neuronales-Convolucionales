import os
from PIL import Image

def flip_images_in_folder(folder_path):
    # Contador para las imágenes procesadas
    images_processed = 0

    # Lista todos los archivos en la carpeta
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Verifica si el archivo es una imagen basada en la extensión
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            with Image.open(file_path) as img:
                # Voltea verticalmente
                img_flip_v = img.transpose(Image.FLIP_TOP_BOTTOM)
                new_filename_v = f"{os.path.splitext(filename)[0]}_flip_v{os.path.splitext(filename)[1]}"
                img_flip_v.save(os.path.join(folder_path, new_filename_v))

                # Voltea horizontalmente
                img_flip_h = img.transpose(Image.FLIP_LEFT_RIGHT)
                new_filename_h = f"{os.path.splitext(filename)[0]}_flip_h{os.path.splitext(filename)[1]}"
                img_flip_h.save(os.path.join(folder_path, new_filename_h))

                # Incrementa el contador
                images_processed += 1

    # Imprime un mensaje al final del proceso
    print(f"Proceso terminado. Se procesaron {images_processed} imágenes.")

# Ruta a la carpeta con las imágenes
folder_path = "D:/Users/Leandro/Downloads/DataSet_1a20_1000_225x400_RGB/20"
flip_images_in_folder(folder_path)
