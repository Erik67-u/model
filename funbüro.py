import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

st.title("🔍 KI Fundbüro")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_Model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")

model = load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild")

    size = (224,224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    data[0] = normalized

    prediction = model.predict(data)

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = prediction[0][index]

    st.subheader("Ergebnis")

    st.write("Objekt:", class_name)
    st.write("Sicherheit:", f"{confidence*100:.2f}%")
