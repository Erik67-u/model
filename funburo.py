import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

st.set_page_config(page_title="Fundbüro KI", page_icon="🔍")
st.title("🔎 Fundbüro KI Bild-Erkennung")
st.write("Lade ein Bild hoch und die KI zeigt, welchem Fundstück es ähnlich ist.")

# Absolute Pfade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_Model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")

# Modell laden
if not os.path.exists(MODEL_PATH):
    st.error(f"Modell nicht gefunden: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH, compile=False)

# Labels laden
if not os.path.exists(LABELS_PATH):
    st.error(f"Labels nicht gefunden: {LABELS_PATH}")
    st.stop()

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Bild Upload
uploaded_file = st.file_uploader("Bild hochladen (jpg, jpeg, png)", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    st.subheader("Ergebnis")
    st.write(f"Erkanntes Objekt: **{class_name[2:] if len(class_name)>2 else class_name}**")
    st.write(f"Confidence Score: {confidence_score*100:.2f}%")
    st.write("BASE_DIR:", BASE_DIR)
    st.write("MODEL_PATH:", MODEL_PATH)
    st.write(os.listdir(BASE_DIR))
    st.write(os.listdir(os.path.join(BASE_DIR, "model")))
