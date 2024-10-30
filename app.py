
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

# Cargar el modelo
def load_model_from_file():
    try:
        modelobien = load_model(r'C:\Users\USUARIO\Documents\numeros\numerosC.keras')
        modelobien.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
        return modelobien
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

modelo = load_model_from_file()

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

if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # Procesar la imagen dibujada
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img = img.convert('L')  # Convertir a escala de grises
        img = img.resize((28, 28))  # Redimensionar a 28x28

        img_array = np.array(img)

        # Invertir los colores (blanco a negro)
        img_array = 255 - img_array  # Cambiar blanco a negro

        # Normalizar y ajustar la forma
        img_array = img_array.reshape((28, 28, 1)) / 255.0  # Normalizar y añadir canal
        img_array = img_array.reshape((1, 28, 28, 1))  # Ajustar a la forma (1, 28, 28, 1)

        # Mostrar la imagen dibujada en el tamaño correcto
        st.image(img, caption="Imagen dibujada", use_column_width=False)

        # Realizar la predicción
        prediction = modelo.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.write(f"La predicción es: {predicted_class}")
    else:
        st.warning("Por favor, dibuja un número antes de predecir.")
