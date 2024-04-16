import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Load your PyTorch model here
def load_model():
    # Replace 'model.pth' with the path to your trained PyTorch model
    model = torch.load('streamlitapp.py', map_location=torch.device('cpu'))
    model.eval()
    return model 

# Define transformations for the input image
def transform_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to predict image class
def predict_image(image):
    model = load_model()
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Streamlit app
def main():
    st.title(' DR Image Classification with PyTorch and Streamlit')
    st.text('Upload an image for classification')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        image_tensor = transform_image(image)

        # Classify the image
        class_index = predict_image(image_tensor)

        # Display the result
        class_names = ['Class 1', 'Class 2', 'Class 3']  # Replace with your class names
        st.success(f'Predicted Class: {class_names[class_index]}')

if __name__ == '__main__':
    main()
