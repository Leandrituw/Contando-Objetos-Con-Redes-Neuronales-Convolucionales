![](img/GFA-logo.png)

# Contando Objetos Con Redes Neuronales Convolucionales 

### 1. Objetivo

Crear un dataset con imagenes propias para entrenar una red neuronal. A partir de este, diseñar y optimizar una arquitectura de una red convolucional para clasificar la cantidad de maices (1 a 20).

### 2. Introducción

Las Redes Neuronales Convolucionales (CNN) son un tipo de red neuronal diseñada para procesar y analizar datos con una estructura de cuadrícula, como imágenes. Han revolucionado el reconocimiento de imágenes y tienen aplicaciones en video, procesamiento del lenguaje natural y bioinformática.

Componentes Principales
1. Convolución (Filtros y Kernel)
➤Filtros (Kernels): Son matrices pequeñas que recorren la imagen para detectar características específicas como bordes, texturas y patrones. Cada filtro se aplica a una región de la imagen de entrada y produce una característica en la salida.
➤Operación de Convolución: Multiplica los valores del kernel por los valores de la imagen en la región cubierta por el kernel y suma estos valores para obtener un solo valor en la salida. Esto se repite para cada posición del kernel en la imagen.
![Convolucion](/img/convolucion.gif)

2. Mapas de Características (Feature Maps)
➤Generación de Características: Al aplicar múltiples filtros a una imagen de entrada, se generan varios mapas de características que capturan diferentes aspectos de la imagen.
➤Dimensiones:
Alto y Largo: Dependen del tamaño del filtro, el paso (stride) y el relleno (padding). No siempre son iguales a las dimensiones de la imagen de entrada.
Profundidad: Igual al número de filtros aplicados en la capa de convolución.
<div align="center">
	<img src="img/ESQUEMA.png">
	<em> Figura 1 - Esquema Feature Map </em>
</div>

3. Pooling
➤Max Pooling y Average Pooling: Reducen la dimensionalidad de los mapas de características. Max pooling toma el valor máximo en una región específica, mientras que average pooling toma el promedio. Esto reduce el número de parámetros y la carga computacional, además de hacer la red más robusta a pequeñas variaciones en la posición de las características.
![Pooling](/img/maxpool.gif)

4. Capas Completamente Conectadas (Fully Connected Layers)
➤Clasificación: Después de varias capas convolucionales y de pooling, los mapas de características se aplanan y pasan a través de una o más capas completamente conectadas. Estas capas actúan como una red neuronal tradicional y se utilizan para la clasificación final.
![Fully Conected](/img/fullyconect.gif)

5. Algoritmos de Entrenamiento
➤Backpropagation: Ajusta los pesos de la red calculando el gradiente del error con respecto a cada peso mediante descenso de gradiente.
➤Optimización: Algoritmos como SGD (Stochastic Gradient Descent), Adam y RMSprop actualizan los pesos de manera eficiente durante el entrenamiento.
➤Función de Pérdida: Las CNNs usan funciones de pérdida como la entropía cruzada para cuantificar el error entre las predicciones de la red y las etiquetas reales.

6. Ventajas de las CNN
➤Extracción Automática de Características: Los filtros aprenden automáticamente a detectar características relevantes durante el entrenamiento.
➤Invariancia a la Translación: Las operaciones de pooling hacen que las CNNs sean robustas a la posición de las características dentro de la imagen.
➤Reducción de Parámetros: Las CNNs reducen significativamente el número de parámetros gracias al uso de filtros compartidos y pooling.

7. Aplicaciones
➤Reconocimiento de Imágenes: Clasificación, detección de objetos, segmentación semántica.
➤Procesamiento de Video: Detección y seguimiento de objetos en secuencias de video.
➤Procesamiento de Lenguaje Natural: Análisis de sentimientos, clasificación de texto.
➤Bioinformática: Análisis de secuencias de ADN, predicción de estructuras proteicas.
	
### 3. Armado Del Dataset 

<div align="center">
	<img src="FOTO DEL DISPOSITIVO PARA SACAR FOTOS">
</div>
CRITERIOS CON LOS QUE SE SACARON LAS FOTOS
CARACTERISTICAS DE LAS FOTOS
FOTOS DE EJEMPLO
LAS 3 TRANDAS DE IMAGENES GENERADAS ACLARANDO CANTIDAD DE FOTOS 
EXPLICACION DE LAS FOTOS HASTA .NPY Y PORQUE (RESIZE, ORDENADO, COLORES, RENOMBRAR, MENCIONAR EJEMPLOS TOMADOS
LINK A LAS 3 TANDAS DE FOTOS 
CODIGO 

### 5. Diseño De La Red

ACLARAR LIBRERIAS
SE PARTIO DEL DATASET 80 - 20
ACLARAR QUE SE COMENZO POR SOLO MAICES Y FUERON VARIANDO LOS CRITERIOS (FILTROS, CAPAS, POOLING, EPOCHS) Y SE CONTROLABA LA PRECISION DE LOS DATOS DE TESTEO
ESQUEMA DE ARQUITECTURA FINAL
SUMMARY
CODIGO

<div align="center">
	<img src="img/cifar-10.png">
</div>

### 6. Prediccion y Resultados 

CURVA DE PRECISION - MAICES SOLOS 
IMAGEN PREDECIDA 
MATRIZ DE CONFUSION 
EXPLICACION DE LAS OTRAS TANDAS DE ENTRENAMIENTO 
IMAGENES DE PREDICCION

<div align="center">
	<img src="img/FOTO.png">
</div>

### 7. Recursos 

Dentro Del Repositorio Se Encuentra: 
* 📄Informe: Redes Neuronales Convolucionales - UTN-FRD📄
* 📂Armado_Del_Dataset📂
* 🡪DataSet_1a20_1000_Maices_Solos_225x400_RGB.zip🡨
* 🡪DataSet_1a20_2000_Maices_Lentejas_225x400_RGB.zip🡨
* 🡪DataSet_1a20_2000_Maices_Lentejas_Arroz_225x400_RGB.zip🡨
* 🡪Reducir_Resolucion_Crear_DataSet_RGB.py🡨
* 📂Entrenamiento_De_Arquitecturas📂 
* 🡪Entrenar_Red_Neuronal_RGB.py🡨
* 📂Prediccion_De_Imagenes📂 
* 🡪Prediccion_RGB.py🡨

* ⚠️SE RECOMIENDA LEER LOS COMENTARIOS DE LOS CODIGOS⚠️

### 8. Fuentes

_[Playlist](https://www.youtube.com/playlist?list=PL-Ogd76BhmcC_E2RjgIIJZd1DQdYHcVf0)_

_[Blog Red Neuronal Para Detectar Diabetes](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)_

_[Introduccion Redes Convolucionales](https://bootcampai.medium.com/redes-neuronales-convolucionales-5e0ce960caf8)_

_[GitHub Sobre CNN De La UTN-GFA](https://github.com/UTN-GFA/UTN-GFA.github.io)_

_[GitHub Sobre CIFAR-10](https://gist.github.com/eblancoh/d379d92a3680360857581d8937ef114b)_

_[Como Entrenar Una Red Con CIFAR-10](https://datasmarts.net/es/como-entrenar-una-red-neuronal-en-cifar-10-con-keras/)_

_[Blog De Funcionamiento De CIFAR-10/100](https://www.cs.toronto.edu/%7Ekriz/cifar.html)_

_[Como Crear Un Dataset Similar a CIFAR-10](https://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10)_
