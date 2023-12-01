import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from license_plate_recognition import segment_characters, predict_characters

# Your provided functions: detect_plate, segment_characters, predict_characters

html = """
<div style="background-color: #608397; padding: 10px; color: white;">
    <h2>Automatic License Plate Recognition (ALPR)</h2>
</div>
"""

# Display the styled text
st.markdown(html, unsafe_allow_html=True)

st.write("### By Genon, Twinkle S. & Murao, Christian Ivan P.")
st.write("#### CPE027 - CPE41S4  Digital Signal and Processing")

uploaded_image = st.file_uploader("Upload an image of a license plate", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Recognize License Plate"):
        # Segment the characters from the image
        char = segment_characters(image)

        if char is not None and len(char) > 0:
            # Load your pre-trained model here
            model_path = './Numberplate.h5'  # Update with your model path
            custom_f1score = None  # Define or import your custom_f1score if used in the model
            loaded_model = tf.keras.models.load_model(model_path, custom_objects={'custom_f1score': custom_f1score})

            # Predict characters
            predicted_characters = predict_characters(char, loaded_model)
            plate_number = ''.join(predicted_characters)
            st.write("Predicted Plate Number:", plate_number)
        else:
            st.write("No characters detected in the license plate.")