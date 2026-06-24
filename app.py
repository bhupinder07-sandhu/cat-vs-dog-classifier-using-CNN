import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# ==================================
# CUSTOM CSS
# ==================================
st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
}

.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: 800;
    color: white;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.metric-card {
    background: #1e293b;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #334155;
}

.result-card {
    background: #1e293b;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #334155;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# LOAD MODEL
# ==================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_dogs_model.keras")

model = load_model()

# ==================================
# HEADER
# ==================================
st.markdown("""
<div class="main-title">
🐱 Cat vs Dog Classifier 🐶
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Deep Learning Based Image Classification Using CNN
</div>
""", unsafe_allow_html=True)

# ==================================
# INFO CARDS
# ==================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>96%</h3>
        <p>Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>CNN</h3>
        <p>Model</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>200×200</h3>
        <p>Input Size</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================================
# FILE UPLOAD
# ==================================
uploaded_file = st.file_uploader(
    "Upload Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)

# ==================================
# SHOW IMAGE
# ==================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    # ==================================
    # PREDICT BUTTON
    # ==================================
    if st.button("🔍 Predict"):

        img = image.resize((200, 200))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        score = prediction[0][0]

        if score > 0.5:
            label = "DOG"
            emoji = "🐶"
            confidence = score * 100
        else:
            label = "CAT"
            emoji = "🐱"
            confidence = (1 - score) * 100

        st.markdown(
            f"""
            <div class="result-card">
                <h1>{emoji} {label}</h1>
                <h2 style="color:#38bdf8;">
                    Confidence: {confidence:.2f}%
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(confidence / 100))

# ==================================
# ABOUT PROJECT
# ==================================
st.markdown("---")

st.subheader("📌 About Project")

st.write("""
This project uses a Convolutional Neural Network (CNN) trained to classify images as either Cat or Dog.

### Technologies Used
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow

### Model Details
- CNN Architecture
- Input Size: 200 × 200
- Binary Classification
""")

# ==================================
# FOOTER
# ==================================
st.markdown("---")

st.markdown("""
<div class="footer">
<b>Developed by Bhupinder Sandhu</b>
</div>
""", unsafe_allow_html=True)