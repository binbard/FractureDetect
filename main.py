import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import io

model = YOLO('./models/best.pt')

st.title("Fracture Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image_np = np.array(image)

    if st.button('Detect Objects'):
        with st.spinner('Detecting...'):
            results = model.predict(source=image_np, conf=0.25)

            print(results)
            
            for result in results:
                st.image(result.plot(), caption='Detected Image', use_column_width=True)
                
                st.write("Detection Details:")
                st.write(f"Image Path: {result.path}")
                st.write(f"Original Image Shape: {result.orig_shape}")
                st.write(f"Class Names: {result.names}")
                st.write(f"Prediction Speed: {result.speed}")

                if 'fracture' in result.names.values():
                    st.success("Fracture detected!")
                else:
                    st.warning("No fracture detected.")

                if result.boxes:
                    st.write("Bounding Boxes:")
                    for box in result.boxes.xyxy:
                        st.write(f"Box: {box}")

                total_time = sum(result.speed.values())
                st.write(f"Total Time for Prediction: {total_time:.2f} ms")
