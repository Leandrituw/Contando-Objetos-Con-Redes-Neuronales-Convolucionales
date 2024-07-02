import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.metrics import confusion_matrix
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Cargando los datos...")

img_height = 400
img_width = 225

# Cargar los datos
images = np.load('D:/Users/Leandro/Downloads/Archivos_Binarios_RGB/images.npy')
labels = np.load('D:/Users/Leandro/Downloads/Archivos_Binarios_RGB/labels.npy')

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizar las imágenes
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convertir etiquetas a One-Hot Encoding
num_classes = 20  # Asegúrate de que el número de clases es correcto
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Definir el modelo de la red neuronal
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)))  # RGB: (img_height, img_width, 3)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(num_classes, activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Crear un generador de imágenes para el data augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,   # Volteo horizontal aleatorio
    vertical_flip=True      # Volteo vertical aleatorio
)

# Generar datos aumentados
augmented_data = []
augmented_labels = []
target_augmented_size = len(X_train) * 2  # Queremos 2000 imágenes aumentadas además de las 1000 originales

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False):
    augmented_data.extend(X_batch)
    augmented_labels.extend(y_batch)
    if len(augmented_data) >= target_augmented_size:
        break  # Salir después de generar el número requerido de imágenes

# Convertir a matriz numpy y truncar a tamaño exacto deseado
augmented_data = np.array(augmented_data[:target_augmented_size])
augmented_labels = np.array(augmented_labels[:target_augmented_size])

# Concatenar los datos aumentados con los originales
X_train_augmented = np.concatenate((X_train, augmented_data), axis=0)
y_train_augmented = np.concatenate((y_train, augmented_labels), axis=0)

# Entrenar el modelo con los datos de entrenamiento aumentados
start_time = time.time()
history = model.fit(X_train_augmented, y_train_augmented, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)
end_time = time.time()

# Evaluar el modelo con los datos de prueba
_, accuracy = model.evaluate(X_test, y_test)
print(f"\n-------------------------------")
print(f'Accuracy: {accuracy*100:.2f}%')
print(f"-------------------------------\n")

# Guardar el modelo entrenado
model.save('D:/Users/Leandro/Downloads/Archivos_Binarios_RGB/modelo.h5')

# Calcular el tiempo de predicción
prediction_start_time = time.time()
predicciones = model.predict(X_test)
prediction_end_time = time.time()
prediction_time = prediction_end_time - prediction_start_time

print(f"\n-------------------------------")
print(f'Tiempo de predicción: {prediction_time:.4f} segundos')
print(f"-------------------------------")

# Visualizar la curva de aprendizaje
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Curva de Aprendizaje')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Convertir predicciones y etiquetas de prueba de One-Hot a etiquetas originales
y_pred = np.argmax(predicciones, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcular la matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz De Confusión')
plt.show()

print(f"\nTerminó la ejecución de la red")
