from flask import Flask, request, render_template
import numpy as np
import pandas as pd  # Añadir pandas
from tensorflow.keras.models import load_model # type: ignore
import joblib  # Usar joblib para cargar el scaler

# Crear la app Flask
app = Flask(__name__)

# Cargar el modelo guardado
model = load_model('/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/Proyecto Final/Catan_Model.h5')

# Cargar el scaler guardado
scaler = joblib.load('/Users/maxgallardo/Documents/TEC/Semestres/Semestre 7/TC3006C/AI-DS/Proyecto Final/scaler.pkl')

# Ruta principal para el formulario
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para predecir basado en los inputs del formulario
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario (las 14 características)
        input_features = [float(x) for x in request.form.values()]
        
        # Convertir los datos a un DataFrame de pandas (incluyendo los nombres de las columnas)
        columns = ['settlement1_dice1', 'settlement1_dice2', 'settlement1_dice3', 
                   'settlement2_dice1', 'settlement2_dice2', 'settlement2_dice3', 
                   'num_lumber', 'num_wheat', 'num_clay', 'num_sheep', 'num_ore', 
                   'num_3G', 'num_2(X)', 'num_D']
        
        features_df = pd.DataFrame([input_features], columns=columns)
        
        # Estandarizar los datos con el scaler de las 14 características
        features_scaled = scaler.transform(features_df)
        
        # Hacer la predicción
        prediction = model.predict(features_scaled)
        prediction_label = 'Ganará' if prediction[0] > 0.5 else 'No ganará'

        return render_template('index.html', prediction_text=f'Predicción: {prediction_label}')
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)