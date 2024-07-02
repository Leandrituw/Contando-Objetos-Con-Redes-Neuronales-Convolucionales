from PIL import Image

def resize_image(input_path, output_path, scale_percent):
    # Abrir la imagen
    img = Image.open(input_path)
    
    # Corregir la orientación si es necesario
    if hasattr(img, '_getexif'):
        orientation = 0x0112
        exif = img._getexif()
        if exif is not None:
            orientation = exif[orientation]
            rotations = {
                3: Image.ROTATE_180,
                6: Image.ROTATE_270,
                8: Image.ROTATE_90
            }
            if orientation in rotations:
                img = img.transpose(rotations[orientation])
    
    # Obtener las dimensiones originales de la imagen
    width, height = img.size
    
    # Calcular las nuevas dimensiones basadas en el porcentaje de escala
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    
    # Redimensionar la imagen manteniendo las proporciones
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Guardar la imagen redimensionada en el nuevo archivo
    img.save(output_path)

# Ruta de la imagen de entrada
input_path = "C:/Users/ivana/Desktop/11_19.jpg"

# Ruta de la imagen de salida (donde se guardará la imagen redimensionada)
output_path = "C:/Users/ivana/Desktop/reducida2.jpg"

# Porcentaje de escala (90% en este caso)
scale_percent = 7

# Llamar a la función para redimensionar la imagen
resize_image(input_path, output_path, scale_percent)

print("Imagen redimensionada con éxito.")
