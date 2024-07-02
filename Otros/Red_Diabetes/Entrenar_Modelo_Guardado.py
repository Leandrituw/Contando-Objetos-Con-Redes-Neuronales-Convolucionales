# Importar las bibliotecas necesarias
# Primero, numpy para trabajar con matrices
# Segundo, las partes específicas de Keras que necesitamos para construir el modelo de red neuronal
from numpy import loadtxt
from tensorflow.keras.models import load_model
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

# Cargar el modelo guardado
model = load_model('modelo_entrenado.keras')

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo con los datos de entrenamiento
history = model.fit(X_train, y_train, epochs=200, batch_size=10, validation_data=(X_test, y_test))

# Evaluar el modelo utilizando los datos de prueba
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))  # Imprimir la precisión del modelo en porcentaje

# Imprimir el tiempo de compilación
print("Tiempo De Compilación:", time.time() - start_time)
