import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Image Filter Customizer", layout="centered")
st.title("🖼️ Image Filter by RUSHO")
st.markdown("Upload an image and apply filters interactively using sliders and switches.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (400, 400))
    st.image(img, caption="Original Image", channels="BGR")

    def convert_to_bytes(image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    st.markdown("---")
    st.subheader("🎛️ Filter Controls")

    # Homogeneous Filter
    if st.checkbox("Apply Homogeneous Filter"):
        kernel_size = st.slider("Kernel Size (Homogeneous)", 1, 15, 5, step=2)
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        h_filter = cv2.filter2D(img, -1, kernel)
        st.image(h_filter, caption="Homogeneous Filter", channels="BGR")
        st.download_button("Download Homogeneous Filter", data=convert_to_bytes(h_filter),
                           file_name="homogeneous.png", mime="image/png")

    # Blur Filter
    if st.checkbox("Apply Blur Filter"):
        blur_size = st.slider("Kernel Size (Blur)", 1, 15, 8, step=1)
        blur = cv2.blur(img, (blur_size, blur_size))
        st.image(blur, caption="Blurred Image", channels="BGR")
        st.download_button("Download Blurred Image", data=convert_to_bytes(blur),
                           file_name="blurred.png", mime="image/png")

    # Gaussian Filter
    if st.checkbox("Apply Gaussian Filter"):
        gau_size = st.slider("Kernel Size (Gaussian)", 1, 15, 5, step=2)
        gau = cv2.GaussianBlur(img, (gau_size, gau_size), 0)
        st.image(gau, caption="Gaussian Blur", channels="BGR")
        st.download_button("Download Gaussian Blur", data=convert_to_bytes(gau),
                           file_name="gaussian.png", mime="image/png")

    # Median Filter
    if st.checkbox("Apply Median Filter"):
        med_size = st.slider("Kernel Size (Median)", 1, 15, 5, step=2)
        med = cv2.medianBlur(img, med_size)
        st.image(med, caption="Median Blur", channels="BGR")
        st.download_button("Download Median Blur", data=convert_to_bytes(med),
                           file_name="median.png", mime="image/png")

    # Bilateral Filter
    if st.checkbox("Apply Bilateral Filter"):
        d = st.slider("Diameter", 1, 15, 9)
        sigma_color = st.slider("Sigma Color", 10, 150, 75)
        sigma_space = st.slider("Sigma Space", 10, 150, 75)
        bi_f = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        st.image(bi_f, caption="Bilateral Filter", channels="BGR")
        st.download_button("Download Bilateral Filter", data=convert_to_bytes(bi_f),
                           file_name="bilateral.png", mime="image/png")

    st.markdown("---")
    st.info("Tip: Use sliders to fine-tune each filter. You can apply multiple filters and download your favorites!")

else:
    st.warning("Please upload an image to begin.")
