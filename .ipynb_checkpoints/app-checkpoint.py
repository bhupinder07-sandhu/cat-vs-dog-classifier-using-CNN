import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_dogs_model.h5")

model = load_model()

# Title
st.markdown(
    """
    <h1 style='text-align:center;'>🐱 Cat vs Dog Classifier 🐶</h1>
    <p style='text-align:center;'>Upload an image and let the CNN model predict.</p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Upload Image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((200, 200))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        label = "🐱 Cat"
        confidence = (1 - prediction) * 100
    else:
        label = "🐶 Dog"
        confidence = prediction * 100

    with col2:
        st.success(f"Prediction: {label}")
        st.metric("Confidence", f"{confidence:.2f}%")

    st.divider()

    st.subheader("Prediction Details")

    st.write(f"Raw Model Output: **{prediction:.4f}**")

    st.progress(float(confidence / 100))

else:
    st.info("Upload an image to start prediction.")

st.divider()

st.markdown(
    """
    ### About Project
    This application uses a Convolutional Neural Network (CNN) trained on
    Cat and Dog images.

    **Model Architecture**
    - Conv2D (16 Filters)
    - Conv2D (32 Filters)
    - Conv2D (64 Filters)
    - Dense (512 Neurons)
    - Sigmoid Output Layer

    **Frameworks Used**
    - TensorFlow / Keras
    - Streamlit
    - NumPy
    - Pillow
    """
)