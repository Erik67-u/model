import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

np.set_printoptions(suppress=True)

st.title("🔎 KI Fundbüro")
st.write("Lade ein Bild hoch und die KI erkennt das Objekt.")

# Pfade definieren
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_Model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")

# Modell laden
model = load_model(MODEL_PATH, compile=False)

# Labels laden
with open(LABELS_PATH, "r") as f:
    class_names = f.readlines()

# Bild Upload
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)

    index = np.argmax(prediction)

    class_name = class_names[index]

    confidence_score = prediction[0][index]

    st.subheader("Ergebnis")

    st.write("Erkanntes Objekt:", class_name[2:])
    st.write("Confidence Score:", float(confidence_score))
