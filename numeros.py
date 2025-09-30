#pip install streamlit
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# Mostrar logo si existe
if os.path.exists('logo.gif'):
    st.image('logo.gif', use_container_width=True)
else:
    st.warning("No se encontró 'logo.gif'. Asegúrate de que esté en la carpeta.")

# Título
st.title("ALFREDO DIAZ CLARO 2024")

# Detectar automáticamente archivos .keras en la carpeta
modelos_disponibles = [f for f in os.listdir() if f.endswith('.keras')]

# Selección del modelo
modelo_seleccionado = st.selectbox("Selecciona un modelo", modelos_disponibles)

# Cargar el modelo seleccionado
@st.cache_resource
def load_model_from_file(modelo_path):
    modelobien = load_model(modelo_path)
    return modelobien

modelo = load_model_from_file(modelo_seleccionado)

# Sección para dibujar
st.subheader("Dibuja un número")
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

# Botón para predecir
if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # Convertir la imagen
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img = img.convert('L')
        img = img.resize((28, 28))

        img_array = np.array(img)
        img_array = 255 - img_array  # Invertir colores

        img_array = img_array.reshape((1, 28, 28, 1)) / 255.0  # Normalizar

        # Mostrar imagen procesada
        st.image(img, caption="Imagen procesada", width=150)

        # Realizar la predicción
        prediction = modelo.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_probability = np.max(prediction)

        st.success(f"Predicción: **{predicted_class}** con probabilidad: **{predicted_probability:.2f}**")
    else:
        st.warning("Por favor, dibuja un número antes de predecir.")
