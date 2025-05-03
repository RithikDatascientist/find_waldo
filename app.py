import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.markdown("""
<style>
.header-text {
    font-size:24px;
    font-weight:bold;
}
</style>
<div class='header-text'>
👨‍💻 Rithik — Data Scientist
</div>
""", unsafe_allow_html=True)
st.title("Find :red[Waldo]!")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and convert to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Load YOLO model
    model = YOLO(r"c:\works\runs\detect\train10\weights\best.pt")
    results = model.predict(source=image_bgr)
    result = results[0]

    # Plot result image with boxes
    result_image_np = result.plot()
    result_pil = Image.fromarray(result_image_np[:, :, ::-1])  # Convert BGR to RGB for PIL

    # Convert result image to bytes for download
    img_byte_arr = io.BytesIO()
    result_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Display original and result images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detection Results")
        st.image(result_pil, use_container_width=True)

    # Display detection message
    if len(result.boxes) > 0:
        st.success("🎉 Waldo found!")
    else:
        st.warning("😞 Waldo not found.")

    # Download button
    st.download_button(
        label="📥 Download Detection Image",
        data=img_byte_arr,
        file_name="waldo_detection.png",
        mime="image/png"
    )
st.markdown("""---""")
st.markdown("""
<center>
    Made with ❤️ by Rithik | 
    [GitHub](https://github.com/your-github-username) | 
    [LinkedIn](https://www.linkedin.com/in/your-linkedin-username/)
</center>
""", unsafe_allow_html=True)
