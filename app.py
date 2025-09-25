
# pip install tensorflow==2.17.0
#pip install streamlit
#!pip install streamlit-drawable-canvas
#
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Cargar el logo
st.image('logo.gif', width="container")  # Asegúrate de que el logo esté en la misma carpeta que tu script

# Título
st.title("ALFREDO DIAZ CLARO 2024")

# Lista de modelos disponibles
modelos_disponibles = ['numerosD1.keras', 'numerosC2.keras', 'numerosC3.keras']

# Función para cargar los modelos
def load_model_from_file(modelo_path):
    try:
        modelobien = load_model(modelo_path)
        modelobien.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
        return modelobien
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Cargar los tres modelos
modelo_d1 = load_model_from_file('numerosD1.keras')
modelo_c2 = load_model_from_file('numerosC2.keras')
modelo_c3 = load_model_from_file('numerosC3.keras')

# Configuración del lienzo
st.title("Dibuja un número")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def mostrar_mensaje(probabilidad, modelo_nombre):
    """Mostrar mensaje si la probabilidad es baja"""
    if probabilidad < 0.8:
        return f" ({modelo_nombre}): No identificado adecuadamente, escribe nuevamente"
    else:
        return f" ({modelo_nombre}): Predicción alta probabilidad, verificar: {probabilidad:.2f}"

if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # Procesar la imagen dibujada
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img = img.convert('L')  # Convertir a escala de grises
        img = img.resize((28, 28))  # Redimensionar a 28x28

        img_array = np.array(img)

        # Invertir los colores (blanco a negro) si es necesario
        img_array = 255 - img_array  # Cambiar blanco a negro, depende de cómo entrenaste el modelo

        # Normalizar y ajustar la forma
        img_array = img_array.reshape((28, 28, 1)) / 255.0  # Normalización a rango [0, 1]
        img_array = img_array.reshape((1, 28, 28, 1))  # Ajustar a la forma (1, 28, 28, 1)

        # Realizar las predicciones para los tres modelos
        with st.spinner('Realizando predicciones...'):
            prediction_d1 = modelo_d1.predict(img_array)
            predicted_class_d1 = np.argmax(prediction_d1)
            predicted_probability_d1 = np.max(prediction_d1)

            prediction_c2 = modelo_c2.predict(img_array)
            predicted_class_c2 = np.argmax(prediction_c2)
            predicted_probability_c2 = np.max(prediction_c2)

            prediction_c3 = modelo_c3.predict(img_array)
            predicted_class_c3 = np.argmax(prediction_c3)
            predicted_probability_c3 = np.max(prediction_c3)

        # Crear 3 columnas para mostrar las predicciones de los modelos
        col1, col2, col3 = st.columns(3)

        # Mostrar las predicciones de cada modelo
        with col1:
            st.subheader("Modelo D1")
            st.write(f"Predicción: {predicted_class_c2}" + mostrar_mensaje(predicted_probability_d1, "Modelo D1"))

        with col2:
            st.subheader("Modelo C2")
            st.write(f"Predicción: {predicted_class_c2}" + mostrar_mensaje(predicted_probability_c2, "Modelo C2"))

        with col3:
            st.subheader("Modelo C3")
            st.write(f"Predicción: {predicted_class_c3}" + mostrar_mensaje(predicted_probability_c3, "Modelo C3"))

    else:
        st.warning("Por favor, dibuja un número antes de predecir.")


