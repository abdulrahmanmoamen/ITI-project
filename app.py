import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# تحميل النموذج
model = load_model(r"E:\machine learning\iti project\final_mobilenetv2_model.h5")
class_names = ['Authentic', 'Spliced']

# تخصيص التصميم العام
st.set_page_config(page_title="Image Forgery Detector", layout="centered")

st.markdown("""
    <style>
    .title {
        font-size:40px;
        font-weight:bold;
        color:#2C3E50;
        text-align:center;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #7f8c8d;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# عنوان التطبيق
st.markdown('<div class="title">🕵️‍♂️ Image Forgery Detection</div>', unsafe_allow_html=True)

# رفع الصورة
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

# عند رفع صورة
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (128, 128)) / 255.0
    image_input = np.expand_dims(image_resized, axis=0)

    # مؤشر تحميل
    with st.spinner("🔍 Analyzing image..."):
        prediction = model.predict(image_input)

    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    # عرض الصورة والنتيجة في صف
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.metric(label="Prediction", value=class_names[predicted_class], delta=f"{confidence:.2f} %")

    # رسالة أخيرة
    if class_names[predicted_class] == "Spliced":
        st.warning("⚠️ This image might be FORGED!")
    else:
        st.success("✅ This image appears to be AUTHENTIC.")

# الفوتر
st.markdown('<div class="footer">Developed by abdulrahman - ITI Project 2025</div>', unsafe_allow_html=True)
