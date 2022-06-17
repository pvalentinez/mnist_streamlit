import streamlit as st
import datetime, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import h5py
from keras.models import load_model


def load_savemodel():
    model = load_model('mnist_model.h5') 
    return model

def draw_canvas():
    drawing_mode = "freedraw"
    st.sidebar.subheader("Choose indicator")
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 20)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#eee")
    bg_color = st.sidebar.color_picker("Background color hex: ")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    #realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=True,
        height=200,
        width=200,
        drawing_mode=drawing_mode, 
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        #st.image(canvas_result.image_data)
        image = canvas_result.image_data
        image1 = image.copy()
        image1 = image1.astype('uint8')
        image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        image1 = cv2.resize(image1,(28,28))
        
        st.image(image1, caption='computer vision')
        return image1
        
def guide():
    st.sidebar.subheader('Introduction')
    st.sidebar.write('• Adjust indicator as like')
    st.sidebar.write('• Draw on the drawing board')
    st.sidebar.write('• Click on result to see result')


def onClick(pic):
    if pic is not None:
        model = load_model('mnist_model.h5')
        pic.resize(1,28,28,1)
        pic.astype(float)
        pic = pic / 255.0
        st.title("Predict value :"+str(np.argmax(model.predict(pic))))

def main():
    st.title('Handwritten Digit Recognition')
    guide()
    pic = draw_canvas()
    result = st.button('Predict')
    if result:
        onClick(pic)
    

if __name__ == '__main__':
    main()
