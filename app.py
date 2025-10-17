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

    final_img = img.copy()

    # Brightness Filter
    if st.checkbox("Adjust Brightness"):
        brightness = st.slider("Brightness Level", -100, 100, 0)
        final_img = cv2.convertScaleAbs(final_img, alpha=1, beta=brightness)
        st.image(final_img, caption="After Brightness Adjustment", channels="BGR")

    # Black & White Filter
    if st.checkbox("Convert to Black & White"):
        gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        final_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        st.image(final_img, caption="Black & White", channels="BGR")

    # Background Blur Simulation
    if st.checkbox("Apply Background Blur"):
        mask = np.zeros(final_img.shape[:2], dtype=np.uint8)
        center = (final_img.shape[1] // 2, final_img.shape[0] // 2)
        radius = st.slider("Focus Radius", 50, 200, 100)
        cv2.circle(mask, center, radius, 255, -1)
        blurred = cv2.GaussianBlur(final_img, (21, 21), 0)
        final_img = np.where(mask[:, :, None] == 255, final_img, blurred)
        st.image(final_img, caption="Background Blur", channels="BGR")

    # Homogeneous Filter
    if st.checkbox("Apply Homogeneous Filter"):
        kernel_size = st.slider("Kernel Size (Homogeneous)", 1, 15, 5, step=2)
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        final_img = cv2.filter2D(final_img, -1, kernel)
        st.image(final_img, caption="After Homogeneous Filter", channels="BGR")

    # Blur Filter
    if st.checkbox("Apply Blur Filter"):
        blur_size = st.slider("Kernel Size (Blur)", 1, 15, 8, step=1)
        final_img = cv2.blur(final_img, (blur_size, blur_size))
        st.image(final_img, caption="After Blur Filter", channels="BGR")

    # Gaussian Filter
    if st.checkbox("Apply Gaussian Filter"):
        gau_size = st.slider("Kernel Size (Gaussian)", 1, 15, 5, step=2)
        final_img = cv2.GaussianBlur(final_img, (gau_size, gau_size), 0)
        st.image(final_img, caption="After Gaussian Filter", channels="BGR")

    # Median Filter
    if st.checkbox("Apply Median Filter"):
        med_size = st.slider("Kernel Size (Median)", 1, 15, 5, step=2)
        final_img = cv2.medianBlur(final_img, med_size)
        st.image(final_img, caption="After Median Filter", channels="BGR")

    # Bilateral Filter
    if st.checkbox("Apply Bilateral Filter"):
        d = st.slider("Diameter", 1, 15, 9)
        sigma_color = st.slider("Sigma Color", 10, 150, 75)
        sigma_space = st.slider("Sigma Space", 10, 150, 75)
        final_img = cv2.bilateralFilter(final_img, d, sigma_color, sigma_space)
        st.image(final_img, caption="After Bilateral Filter", channels="BGR")

    st.markdown("---")
    st.subheader("📥 Download Final Image")
    st.download_button("Download Final Filtered Image",
                       data=convert_to_bytes(final_img),
                       file_name="final_filtered.png",
                       mime="image/png")

    st.markdown("---")
    st.info("Tip: You can apply multiple filters in sequence and download the final result!")

else:
    st.warning("Please upload an image to begin.")
