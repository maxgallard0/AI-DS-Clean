# Importar las librerías necesarias
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# Función para inicializar los centroides aleatoriamente
def kMeansInitCentroids(X, K):
    # Seleccionamos aleatoriamente K puntos del conjunto de datos
    indices = random.sample(range(X.shape[0]), K)
    centroids = X[indices, :]
    return centroids

# Función para encontrar el centroide más cercano a cada ejemplo
def findClosestCentroids(X, centroids):
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        idx[i] = np.argmin(distances)
    return idx

# Función para computar los nuevos centroides
def computeCentroids(X, idx, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0)
    return centroids

# Función que ejecuta el algoritmo K-means con opción de graficar centroides
def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    centroids = initial_centroids
    previous_centroids = []
    
    if plot_progress:
        plt.figure(figsize=(10, 5))
    
    for i in range(max_iters):
        # Encontrar los centroides más cercanos
        idx = findClosestCentroids(X, centroids)
        
        # Graficar el progreso de los centroides si se solicita
        if plot_progress:
            # Graficar los puntos de datos (X) con los colores del índice idx
            plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='rainbow', marker='o', edgecolor='k', alpha=0.5)
            
            previous_centroids.append(np.copy(centroids))
            for j, centroid in enumerate(centroids):
                if i > 0:
                    # Dibujar una línea entre la posición anterior y la nueva del centroide
                    plt.plot([previous_centroids[-2][j, 0], centroid[0]],
                             [previous_centroids[-2][j, 1], centroid[1]], 'k-', lw=2)
                # Marcar la posición actual del centroide
                plt.scatter(centroid[0], centroid[1], marker='x', s=100, c='black', lw=3)
        
        # Actualizar los centroides
        centroids = computeCentroids(X, idx, centroids.shape[0])
    
    if plot_progress:
        plt.title(f'Centroid Movements after {max_iters} iterations')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.show()
    
    return centroids, idx

# Cargar la imagen y convertirla en una matriz
def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image) / 255  # Normalizamos los valores entre 0 y 1
    return image

# Función para aplicar K-means a la imagen y comprimirla
def compress_image(image_path, K, max_iters):
    # Cargar la imagen
    image = load_image(image_path)
    
    # Reshape de la imagen a una matriz (16384, 3)
    X = image.reshape(-1, 3)
    
    # Inicializar los centroides
    initial_centroids = kMeansInitCentroids(X, K)
    
    # Ejecutar K-means con visualización de centroides
    centroids, idx = runkMeans(X, initial_centroids, max_iters, plot_progress=True)
    
    # Reemplazar cada pixel con su centroide más cercano
    X_compressed = centroids[idx]
    X_compressed = X_compressed.reshape(image.shape)
    
    # Mostrar la imagen original y la comprimida
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Imagen original')

    plt.subplot(1, 2, 2)
    plt.imshow(X_compressed)
    plt.title('Imagen comprimida')

    plt.show()

# Parámetros para la compresión
K = 16         # Número de clusters (colores)
max_iters = 10  # Número de iteraciones del algoritmo

# Ruta de la imagen
image_path = '/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/M1: EPCD/P4/bird_small.png'

# Aplicar K-means y comprimir la imagen
compress_image(image_path, K, max_iters)