import numpy as np

# Función sigmoidal
def sigmoidal(z):
    return 1 / (1 + np.exp(-z))

# Derivada de la función sigmoidal
def sigmoidal_derivada(z):
    return z * (1 - z)

# Datos XOR
XOR = np.array([
    [1, 0, 0],  # Incluye el 1 para el sesgo
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Salidas deseadas
Y_deseada = np.array([0, 1, 1, 0])

# Pesos iniciales
w13 = 0.5
w14 = 0.2
w23 = -0.3
w24 = 0.5
w34 = 0.1
w45 = 0.3
b3 = 0.6
b4 = -0.4
b5 = 0.8

# Agrupamos los pesos y los sesgos en matrices para facilitar las operaciones
W = np.array([
    [w13, w14],  # Pesos de la capa de entrada a la capa oculta
    [w23, w24],
])

W_oculta_salida = np.array([w34, w45])  # Pesos de la capa oculta a la capa de salida
B_oculta = np.array([b3, b4])  # Sesgos para las neuronas ocultas
B_salida = b5  # Sesgo para la neurona de salida

# Tasa de aprendizaje
n = 0.01

# Entrenamiento hasta que el error sea menor a 0.1
epoch = 0
error_promedio = float('inf')

while error_promedio > 0.01:
    # Forward Pass
    Z_oculta = sigmoidal(XOR[:, 1:] @ W + B_oculta)  # Capa oculta
    Z_salida = sigmoidal(Z_oculta @ W_oculta_salida + B_salida)  # Neurona de salida

    # Cálculo del error en la neurona de salida
    E_salida = sigmoidal_derivada(Z_salida) * (Y_deseada - Z_salida)

    # Cálculo del error promedio
    error_promedio = np.mean(np.abs(Y_deseada - Z_salida))

    # Cálculo del error en las neuronas ocultas
    E_oculta = sigmoidal_derivada(Z_oculta) * (E_salida[:, np.newaxis] @ W_oculta_salida[np.newaxis, :])

    # Actualización de los pesos y sesgos (Backward Pass)
    W_oculta_salida += n * Z_oculta.T @ E_salida
    W += n * XOR[:, 1:].T @ E_oculta

    B_salida += n * np.sum(E_salida)
    B_oculta += n * np.sum(E_oculta, axis=0)
    
    epoch += 1

    if epoch % 100 == 0:  # Imprime las salidas cada 100 épocas
        print(f"Época {epoch}: Salida = {Z_salida}, Error promedio = {error_promedio}")

# Prueba del modelo final con los datos XOR
def probar_XOR(XOR):
    Z_oculta = sigmoidal(XOR[:, 1:] @ W + B_oculta)  # Capa oculta
    Z_salida = sigmoidal(Z_oculta @ W_oculta_salida + B_salida)  # Neurona de salida
    return Z_salida

# Prueba final
salidas_finales = probar_XOR(XOR)

# Imprimir las salidas finales de manera formateada
print(f"Para [0, 0]: {salidas_finales[0]:.4f}")
print(f"Para [0, 1]: {salidas_finales[1]:.4f}")
print(f"Para [1, 0]: {salidas_finales[2]:.4f}")
print(f"Para [1, 1]: {salidas_finales[3]:.4f}")