import numpy as np
import matplotlib.pyplot as plt

# Leer los datos del archivo de texto
data = np.loadtxt('/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/M1: EPCD/ACT_CLASE/K Means/ex7data2.txt')

# Definir el número de clusters
k = 3

# Inicializar los centroides de forma aleatoria
centroids = data[np.random.choice(range(len(data)), k, replace=False)]

# Función para asignar cada punto al centroide más cercano
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = np.linalg.norm(centroids - point, axis=1)
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters

# Función para actualizar los centroides
def update_centroids(data, clusters, k):
    centroids = []
    for i in range(k):
        cluster_points = data[np.where(np.array(clusters) == i)]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return centroids

# Ejecutar el algoritmo K-means
clusters = assign_clusters(data, centroids)
prev_clusters = None

while clusters != prev_clusters:
    prev_clusters = clusters
    centroids = update_centroids(data, clusters, k)
    clusters = assign_clusters(data, centroids)

# Graficar los resultados
for i in range(k):
    cluster_points = data[np.where(np.array(clusters) == i)]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1])

plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], marker='x', color='red')
plt.show()