# Importar las bibliotecas necesarias
# Primero, numpy para trabajar con matrices
# Segundo, las partes específicas de Keras que necesitamos para construir el modelo de red neuronal
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import History
import time

# Cargar el conjunto de datos desde un archivo CSV
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')

# Separar el conjunto de datos en variables de entrada (X) y de salida (y)
X = dataset[:,0:8]  # Las primeras 8 columnas son las variables de entrada
y = dataset[:,8]    # La última columna es la variable de salida

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el tiempo de inicio
start_time = time.time()

# Definir el modelo de la red neuronal utilizando Keras
model = Sequential()  # Inicializar un modelo secuencial (una pila lineal de capas)
model.add(Input(shape=(8,))) # Capa de entrada con la forma de los datos de entrada

model.add(Dense(12, activation='relu'))  # Añadir una capa densa con 12 neuronas y función de activación ReLU 
model.add(Dense(8, activation='relu'))  # Añadir otra capa densa con 8 neuronas y función de activación ReLU
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Añadir una capa densa con 1 neurona y función de activación sigmoide (para problemas de clasificación binaria)

# Compilar el modelo especificando la función de pérdida, el optimizador y las métricas que se utilizarán para evaluar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Crear un objeto History para almacenar el historial de entrenamiento
history = History()

# Entrenar el modelo con los datos de entrenamiento y evaluarlo en los datos de prueba
model.fit(X_train, y_train, epochs=200, batch_size=10, validation_data=(X_test, y_test), callbacks=[history])

# Evaluar el modelo utilizando los datos de prueba
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))  # Imprimir la precisión del modelo en porcentaje

# Imprimir el tiempo de compilación
print("Tiempo De Compilación:", time.time() - start_time)

# Graficar ambas curvas en un mismo gráfico
plt.figure(figsize=(10, 5))

# Curva de Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

# Curva de Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Training')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()

model.save('modelo_entrenado.keras')
