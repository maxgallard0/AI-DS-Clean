import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Puntos (x, y) dados
x = np.array([2, 3, 5, 4, 3, 6, 1])
y = np.array([4, 6, 8, 4, 2, 3, 1])

# Convertimos los valores de x para un polinomio de grado 3
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(x.reshape(-1, 1))

# Creamos el modelo de regresión lineal
model = LinearRegression()
model.fit(X_poly, y)

# Obtenemos los coeficientes (theta_0, theta_1, theta_2, theta_3)
theta_0 = model.intercept_
theta_1, theta_2, theta_3 = model.coef_[1:]

# Imprimir los coeficientes
print(f"θ0 (intercepto): {theta_0}")
print(f"θ1: {theta_1}")
print(f"θ2: {theta_2}")
print(f"θ3: {theta_3}")

# Graficar los puntos y el polinomio ajustado
x_range = np.linspace(0, 7, 100)
y_poly_pred = model.predict(poly.fit_transform(x_range.reshape(-1, 1)))

plt.scatter(x, y, color='red', label='Datos')
plt.plot(x_range, y_poly_pred, color='blue', label='Regresión polinomial grado 3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
