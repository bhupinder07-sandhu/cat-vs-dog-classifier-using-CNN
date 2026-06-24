import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="wide"
)

st.title("🐱 Cat vs Dog Classifier")

st.success("APP STARTED")

try:
    from tensorflow.keras.models import load_model
    st.success("TensorFlow Imported Successfully")
except Exception as e:
    st.error(f"TensorFlow Import Error: {e}")
    st.stop()

@st.cache_resource
def load_cnn_model():
    st.warning("Loading CNN Model...")
    model = load_model("cats_dogs_model.keras")
    st.success("Model Loaded Successfully")
    return model

try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

st.success("APP READY")

uploaded_file = st.file_uploader(
    "Upload Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    if st.button("Predict"):

        img = image.resize((200, 200))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        score = prediction[0][0]

        if score > 0.5:
            label = "DOG 🐶"
            confidence = score * 100
        else:
            label = "CAT 🐱"
            confidence = (1 - score) * 100

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}%")

st.markdown("---")
st.write("Developed by Bhupinder Sandhu")