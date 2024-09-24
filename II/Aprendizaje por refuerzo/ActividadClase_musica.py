import numpy as np

# Función softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Lista de géneros musicales
genres = [
    "Rock",       # 1
    "Pop",        # 2
    "Jazz",       # 3
    "Classical",  # 4
    "Hip-Hop",    # 5
    "Country",    # 6
    "Electronic", # 7
    "Reggae",     # 8
    "Blues",      # 9
    "Metal"       # 10
]

# Ejemplo de muestras con valores del 1 al 10
np.random.seed(42)  # Aseguramos la reproducibilidad
samples = np.random.randint(1, 11, size=100)

# Inicialización de parámetros
params = np.zeros((10,), dtype=np.float64)
r_prom = 0

# Proceso de entrenamiento simulado
for i, sample in enumerate(samples):
    r = -1
    if sample == 1:
        r = 1
    
    # Actualización de la recompensa promedio
    r_prom += (1 / (i + 1)) * (r - r_prom)
    
    # Cálculo de la representación one-hot y el gradiente
    onehot = np.eye(params.shape[0])
    grad = onehot[sample - 1]
    
    # Actualizar parámetros
    params += 0.01 * grad * r_prom

    # Predicción softmax actual
    prediction = softmax(params)
    
    # Mostrar iteración y género musical
    print(f"Iteración {i}: Género: {genres[sample - 1]}, Predicción: {prediction}, Error: {grad - prediction}")

# Resultados finales
final_params = params
final_distribution = softmax(params)

print(f"Resultado Final:\nParámetros finales = {final_params}\nDistribución Final = {final_distribution}")