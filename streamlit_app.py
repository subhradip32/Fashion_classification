import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np



model=tf.keras.models.load_model("fashion_detector.h")

st.write("""
         # Items Classification
         """
        )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

def import_and_predict(image_data, model): 
    image = np.array(image_data)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    loc = np.argmax(pred)
    return class_names[loc]

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    st.write(f"## {predictions}")
