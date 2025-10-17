import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Image Filtering with OpenCV")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (400, 400))
    st.image(img, caption="Original Image", channels="BGR")

    # Filters
    kernel = np.ones((5, 5), np.float32) / 25
    h_filter = cv2.filter2D(img, -1, kernel)
    blur = cv2.blur(img, (8, 8))
    gau = cv2.GaussianBlur(img, (5, 5), 0)
    med = cv2.medianBlur(img, 5)
    bi_f = cv2.bilateralFilter(img, 9, 75, 75)

    # Display filters
    st.image(h_filter, caption="Homogeneous Filter", channels="BGR")
    st.image(blur, caption="Blurred", channels="BGR")
    st.image(gau, caption="Gaussian Blur", channels="BGR")
    st.image(med, caption="Median Blur", channels="BGR")
    st.image(bi_f, caption="Bilateral Filter", channels="BGR")
