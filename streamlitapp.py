pip install streamlit
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("DR Image Classification Using Pytorch and Streamlit")
st.write("")

file_up = st.file_uploader("Upload an image", type=('jpg' , 'png'))

#file_up_2 = st.file_uploader("Upload model", type=('pt' , 'pth'))

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels = predict(file_up)

    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
streamlit run app.py
