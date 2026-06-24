import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    background-color: #1e293b;
    color: white;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_cnn_model():
    return load_model("cats_dogs_model.keras")

with st.spinner("Loading CNN Model... Please wait ⏳"):
    model = load_cnn_model()

# ---------------- HEADER ----------------
st.markdown(
    '<p class="main-title">🐱 Cat vs Dog Classifier 🐶</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Deep Learning Image Classification using CNN</p>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:

        if st.button("🔍 Predict", use_container_width=True):

            img = image.resize((200, 200))

            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)[0][0]

            # Change if labels are reversed
            if prediction > 0.5:
                label = "🐶 DOG"
                confidence = prediction * 100
            else:
                label = "🐱 CAT"
                confidence = (1 - prediction) * 100

            st.markdown(
                f"""
                <div class="result-box">
                Prediction: {label}<br>
                Confidence: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()

# ---------------- ABOUT ----------------
st.subheader("📌 About Project")

st.write("""
This project uses a Convolutional Neural Network (CNN) trained on Cat and Dog image datasets.

- Deep Learning Model: CNN
- Framework: TensorFlow & Keras
- Frontend: Streamlit
- Image Size: 200 × 200
- Binary Classification: Cat vs Dog
""")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
    <h4>Developed by Bhupinder Sandhu</h4>
    B.Tech AI & Data Science Project
    </div>
    """,
    unsafe_allow_html=True
)