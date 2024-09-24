import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Evita overflow
    return exp_x / exp_x.sum(axis=0)

# Suponiendo que counts representa la cantidad de géneros y tiene 10 entradas
counts = np.zeros(10, dtype=np.float64)  # Inicializamos para 10 géneros
params = np.zeros_like(counts, dtype=np.float64)
print(params, softmax(params).dtype)

# Supongamos que 'samples' es una lista de muestras donde cada entrada corresponde a uno de los 10 géneros (valores de 1 a 10)
samples = np.random.randint(1, 11, size=100)  # Ejemplo: 100 muestras aleatorias de géneros entre 1 y 10

r_prom = 0
learning_rate = 0.01  # Tasa de aprendizaje

for i, sample in enumerate(samples):
    r = -1
    if sample == 1:  # Aquí se evalúa si el sample pertenece al género 1
        r = 1
    r_prom += (1 / (i + 1)) * (r - r_prom)
    
    onehot = np.eye(params.shape[0])  # Crea una matriz identidad de 10x10 para los géneros
    grad = onehot[sample - 1]  # Ajusta para el rango de 10 géneros (índices de 0 a 9)
    
    prediction = softmax(params)
    error = grad - prediction  # Calcula el error entre la predicción y el valor real
    
    params += learning_rate * error  # Actualiza los parámetros usando el gradiente descendiente
    
    print(f"Iteración {i}: Sample: {sample}, Predicción: {prediction}, Error: {error}")

print(f"Parámetros finales: {params}")
print(f"Distribución final softmax: {softmax(params)}")