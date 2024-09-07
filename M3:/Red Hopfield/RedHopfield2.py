import numpy as np
from tensorflow.keras.datasets import mnist # type: ignore
import matplotlib.pyplot as plt

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar y binarizar las imágenes
# Aquí estamos convirtiendo las imágenes en una forma binaria: 1 para píxeles brillantes y -1 para píxeles oscuros
train_images = np.where(train_images > 127, 1, -1)  # Convertir a binario: 1 y -1
test_images = np.where(test_images > 127, 1, -1)

# Aplanar las imágenes (de 28x28 a un vector de 784)
# Esto convierte las imágenes bidimensionales de 28x28 en vectores unidimensionales de tamaño 784
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# Función para añadir ruido a las imágenes
# Esta función toma un patrón y voltea una proporción de sus bits para simular ruido
def add_noise(pattern, noise_level=0.3):  # Cambiar el nivel de ruido a 30%
    noisy_pattern = np.copy(pattern)
    n_flip = int(noise_level * len(pattern))  # número de bits a voltear
    flip_indices = np.random.choice(len(pattern), n_flip, replace=False)  # seleccionar qué bits voltear
    noisy_pattern[flip_indices] *= -1  # voltear bits (cambia 1 a -1 y viceversa)
    return noisy_pattern

# Función de entrenamiento de la red Hopfield
# Esta función crea la matriz de pesos W sumando los productos externos de cada patrón
def train_hopfield(patterns):
    n = patterns[0].size  # tamaño de cada patrón (784 para MNIST)
    W = np.zeros((n, n))  # matriz de pesos
    for p in patterns:
        W += np.outer(p, p)  # suma de productos externos
    np.fill_diagonal(W, 0)  # poner la diagonal en cero (no se permiten auto-conexiones)
    return W / len(patterns)  # normalizar

# Función para recuperar el patrón original en la red Hopfield
# Aplica la regla de activación de Hopfield (W @ pattern) para cada paso
def hopfield_predict(W, pattern, steps=20):
    for _ in range(steps):
        pattern = np.sign(W @ pattern)  # activación con la regla de Hopfield (signo de W * patrón)
    return pattern

# Seleccionar ejemplos de entrenamiento (solo 2 ejemplos por dígito, dígitos 0 y 1)
unique_digits = []
for i in range(2):  # Solo los dígitos 0 y 1
    unique_digits.extend(train_images[np.where(train_labels == i)[0][:2]])  # Usar 2 ejemplos de cada dígito
W = train_hopfield(unique_digits)  # Entrenar la red Hopfield con los patrones seleccionados

# Visualización de los dígitos 0 y 1 antes y después de añadir ruido y la recuperación
plt.figure(figsize=(10, 3))  # Tamaño de la figura
for i in range(2):
    # Seleccionar imagen de prueba (primera imagen del dígito 0 o 1)
    test_image = test_images[np.where(test_labels == i)[0][0]]
    
    # Añadir ruido a la imagen de prueba
    noisy_image = add_noise(test_image, noise_level=0.3)  # Añadir ruido al 30%
    
    # Recuperar el patrón original usando la red Hopfield con más pasos de actualización
    recovered_image = hopfield_predict(W, noisy_image, steps=20)
    
    # Plotear las tres imágenes: original, ruidosa y recuperada
    plt.subplot(3, 2, i + 1)
    plt.imshow(test_image.reshape(28, 28), cmap='gray')  # Mostrar imagen original
    plt.title(f"Original {i}")
    plt.axis('off')

    plt.subplot(3, 2, i + 3)
    plt.imshow(noisy_image.reshape(28, 28), cmap='gray')  # Mostrar imagen con ruido
    plt.title(f"Ruidosa {i}")
    plt.axis('off')

    plt.subplot(3, 2, i + 5)
    plt.imshow(recovered_image.reshape(28, 28), cmap='gray')  # Mostrar imagen recuperada por la red Hopfield
    plt.title(f"Recuperada {i}")
    plt.axis('off')

plt.tight_layout()  # Ajustar el diseño de los subplots
plt.show()  # Mostrar la figura