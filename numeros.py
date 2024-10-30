#pip install streamlit
import streamlit as st
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Cargar el modelo
def load_model():
    # Cargar el modelo JSON
    with open('numeros1.weghts.json', 'r') as json_file:
        model_json = json_file.read()
    modelo = model_from_json(model_json)
    
    # Cargar los pesos
    modelo.load_weights('numeros1.weights.h5')  # Actualizado a .weights.h5
    
    return modelo

modelo = load_model()

# Configuración del lienzo
st.title("Dibuja un número")
canvas_result = st_canvas(
    fill_color="white",  # Color de fondo
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
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGB')
        img = img.convert('L')  # Convertir a escala de grises
        img = img.resize((28, 28))  # Redimensionar
        img_array = np.array(img)
        img_array = img_array.reshape((1, 28, 28, 1)) / 255.0  # Normalizar y ajustar forma

        # Realizar la predicción
        prediction = modelo.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.write(f"La predicción es: {predicted_class}")

