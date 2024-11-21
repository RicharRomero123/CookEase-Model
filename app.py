from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os  # Para obtener el puerto dinámico

app = Flask(__name__)

# Cargar el modelo y las columnas
modelo = joblib.load('modelo.pkl')
columnas = joblib.load('columnas.pkl')

def preprocesar_datos(df_input):
    # Lista de variables categóricas
    categorical_vars = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
                        'SCC', 'CALC', 'MTRANS']

    # Convertir variables categóricas en variables dummy
    df_input = pd.get_dummies(df_input, columns=categorical_vars, drop_first=True)

    # Asegurarse de que las columnas del input coincidan con las del modelo
    missing_cols = set(columnas) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0
    df_input = df_input[columnas]

    return df_input

def predecir_obesidad(modelo, df_input):
    prediccion = modelo.predict(df_input)
    return prediccion[0]

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del request en formato JSON
    data = request.get_json(force=True)
    
    # Convertir los datos en un DataFrame
    df_usuario = pd.DataFrame([data])
    
    # Preprocesar los datos
    df_usuario_procesado = preprocesar_datos(df_usuario)
    
    # Realizar la predicción
    resultado = predecir_obesidad(modelo, df_usuario_procesado)
    
    # Retornar el resultado en formato JSON
    return jsonify({'prediccion': resultado})

if __name__ == '__main__':
    # Obtener el puerto dinámico del entorno o usar 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    # Ejecutar la aplicación Flask en el host 0.0.0.0
    app.run(host='0.0.0.0', port=port, debug=True)
