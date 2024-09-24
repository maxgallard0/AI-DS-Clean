import numpy as np
from scipy.spatial import distance

# Definir los puntos y los centroides
points = np.array([[2, 3, 5], [1, 3, 2], [6, 2, 4], [-1, 1, 3]])
centroid_1 = np.array([0, 1, 1])
centroid_2 = np.array([4, 1, 2])

# Calcular las distancias entre cada punto y los centroides
clusters = []
for point in points:
    dist_c1 = distance.euclidean(point, centroid_1)
    dist_c2 = distance.euclidean(point, centroid_2)
    
    # Asignar el punto al clúster más cercano
    if dist_c1 < dist_c2:
        clusters.append(1)
    else:
        clusters.append(2)

# Imprimir los resultados
print("Asignación de clusters:", clusters)