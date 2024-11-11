import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
import time

print("Cargando los datos...")

img_height = 400
img_width = 225

# Cargar los datos
images = np.load('D:/Users/Leandro/Downloads/Redes Neuronales/DataSets/Maices_Solos/images.npy')
labels = np.load('D:/Users/Leandro/Downloads/Redes Neuronales/DataSets/Maices_Solos/labels.npy')

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizar las imágenes
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convertir etiquetas a One-Hot Encoding
num_classes = 20  # Asegúrate de que el número de clases es correcto
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Definir el modelo de la red neuronal usando `Input`
model = Sequential()
model.add(Input(shape=(img_height, img_width, 3)))  # Definir la entrada del modelo
model.add(Conv2D(2, (3, 3), padding='same', activation='relu'))  # Primera capa sin input_shape
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(0.25))
model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(num_classes, activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Entrenar el modelo con los datos de entrenamiento
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)
end_time = time.time()

# Evaluar el modelo con los datos de prueba
_, accuracy = model.evaluate(X_test, y_test)
print(f"\n-------------------------------")
print(f'Accuracy: {accuracy*100:.2f}%')
print(f"-------------------------------\n")

# Guardar el modelo entrenado
model.save('D:/Users/Leandro/Downloads/Redes Neuronales/DataSets/Maices_Solos/modelo.h5')

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

# Calcular el MSE
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

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
