ventas_netas_anuales = 10753824.53  # en pesos mexicanos
gastos_totales_anuales = 9950325.70  # en pesos mexicanos

# Cálculo de la utilidad anual antes de impuestos
utilidad_anual_antes_impuestos = ventas_netas_anuales - gastos_totales_anuales

print(utilidad_anual_antes_impuestos)

# Distribución de la utilidad entre los 4 fundadores
utilidad_anual_por_fundador = utilidad_anual_antes_impuestos / 4

print(utilidad_anual_por_fundador)

# Cálculo de la utilidad mensual promedio por fundador
utilidad_mensual_promedio_por_fundador = utilidad_anual_por_fundador / 12

print(utilidad_mensual_promedio_por_fundador)