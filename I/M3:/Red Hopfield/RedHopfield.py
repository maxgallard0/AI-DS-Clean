import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist # type: ignore

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar y binarizar las imágenes: Convertir los píxeles a valores binarios: 1 y -1
train_images = np.where(train_images > 127, 1, -1)
test_images = np.where(test_images > 127, 1, -1)

# Aplanar las imágenes (de 28x28 a un vector de 784)
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# Función para añadir ruido
def add_noise(pattern, noise_level=0.3):
    noisy_pattern = np.copy(pattern)
    n_flip = int(noise_level * len(pattern))  # número de bits a voltear
    flip_indices = np.random.choice(len(pattern), n_flip, replace=False)
    noisy_pattern[flip_indices] *= -1  # voltear bits (cambia 1 a -1 y viceversa)
    return noisy_pattern

# Entrenamiento de la red Hopfield
def train_hopfield(patterns):
    n = patterns[0].size  # tamaño de cada patrón (784 para MNIST)
    W = np.zeros((n, n))  # matriz de pesos
    for p in patterns:
        W += np.outer(p, p)  # suma de productos externos
    np.fill_diagonal(W, 0)  # diagonal en cero
    return W / len(patterns)  # normalizar

# Recuperación en la red Hopfield
def hopfield_predict(W, pattern, steps=5):
    for _ in range(steps):
        pattern = np.sign(W @ pattern)  # activación con la regla de Hopfield
    return pattern

# Función para evaluar la red con diferentes niveles de ruido y calcular la tasa de recuperación
def evaluate_noise_recovery(W, patterns, noise_levels, steps=5):
    recovery_rates = []
    for noise_level in noise_levels:
        correct_recovery = 0
        for pattern in patterns:
            noisy_pattern = add_noise(pattern, noise_level)
            recovered_pattern = hopfield_predict(W, noisy_pattern, steps)
            if np.array_equal(recovered_pattern, pattern):
                correct_recovery += 1
        recovery_rate = correct_recovery / len(patterns)
        recovery_rates.append(recovery_rate)
    return recovery_rates

# Seleccionar 10 imágenes (una de cada dígito) para entrenar
unique_digits = [train_images[np.where(train_labels == i)[0][0]] for i in range(10)]
W = train_hopfield(unique_digits)

# Evaluar la red con diferentes niveles de ruido
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
recovery_rates = evaluate_noise_recovery(W, unique_digits, noise_levels, steps=10)

# Graficar los resultados
plt.figure(figsize=(8, 6))
plt.plot(noise_levels, recovery_rates, marker='o', color='b')
plt.title('Rendimiento de la red Hopfield en función del nivel de ruido')
plt.xlabel('Nivel de Ruido')
plt.ylabel('Tasa de Recuperación')
plt.grid(True)
plt.show()