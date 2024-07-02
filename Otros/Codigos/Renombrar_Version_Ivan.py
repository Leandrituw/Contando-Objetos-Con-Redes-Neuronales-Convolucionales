import os
from pathlib import Path

def rename_photos(directory):
    # Verificar si el directorio existe
    if not os.path.isdir(directory):
        print(f"El directorio {directory} no existe.")
        return

    # Obtener todos los archivos en el directorio
    files = list(Path(directory).glob('*'))

    # Filtrar y ordenar los archivos por tamaño ascendente
    files = sorted([file for file in files if file.is_file()], key=lambda x: x.stat().st_size, reverse=False)

    # Inicializar N y M
    N = 33
    M = 20


    # Iterar sobre cada archivo en el directorio
    for file_path in files:
        # Verificar si el archivo es una foto (se puede ajustar para otros tipos de archivos)
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Construir el nuevo nombre de archivo
            new_filename = f"{M}_{N}{file_path.suffix}"

            # Construir la nueva ruta del archivo
            new_path = file_path.with_name(new_filename)

            # Renombrar el archivo
            file_path.rename(new_path)

            print(f"Renombrado: {new_path}")

            # Incrementar N y verificar si se reinicia a 1 y se incrementa M
            N += 1
            if N > 40:
                N = 1
                M += 1

# Directorio donde se encuentran las fotos a renombrar
directory_path = r"C:\Users\lmichel\Desktop\Test img (160)\20"

# Llamar a la función para renombrar las fotos en el directorio
rename_photos(directory_path)
